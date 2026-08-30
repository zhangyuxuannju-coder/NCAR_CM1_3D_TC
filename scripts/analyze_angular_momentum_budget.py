#!/usr/bin/env python3
"""Close the storm-centred absolute-angular-momentum and inertial-stability budgets.

The script uses a fixed centre for each centred time derivative, so CM1's
Cartesian momentum tendencies and the diagnostic local tendency refer to the
same control volume.  Reynolds and Favre forms are both evaluated.  CM1's
native ub_*/vb_* tendencies provide the primary closure; direct 3-D eddy flux
convergences provide an independent cross-check.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src._se_pipeline_single import (  # noqa: E402
    PipelineConfig,
    _azimuthal_average_by_radius,
    _compute_radial_bin_index,
    _destagger_to_scalar_grid,
    _find_center,
    _get_time_slice,
    _open_dataset_robust,
    _resolve_core_var_names,
)
from src.environmental_eddy import diagnose_eddy_momentum_forcing  # noqa: E402


BUDGET_SUFFIXES = (
    "hadv", "vadv", "cor", "pgrad", "hidiff", "vidiff",
    "hturb", "vturb", "rdamp",
)
DIFF_SUFFIXES = ("hidiff", "vidiff", "hturb", "vturb")
DOMAINS = {
    "core_outflow": (50.0, 200.0, 10.0, 16.0),
    "inner_outflow": (50.0, 350.0, 10.0, 16.0),
    "outer_outflow": (200.0, 500.0, 10.0, 16.0),
    "jet_region": (750.0, 1100.0, 10.0, 17.0),
}
COLORS = {
    "mean_r": "#0072B2", "mean_z": "#56B4E9",
    "eddy_r": "#D55E00", "eddy_z": "#E69F00",
    "pgrad": "#CC79A7", "diffusion": "#009E73",
    "rdamp": "#999999", "residual": "#000000",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nojet", required=True)
    p.add_argument("--jet", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--hours", default="25,55,72,80,110")
    p.add_argument("--difference-hours", type=float, default=1.0)
    p.add_argument("--time-derivative", choices=("centered", "backward", "forward"),
                   default="centered",
                   help="Match instantaneous or interval-averaged CM1 budget output timing")
    p.add_argument("--max-r-km", type=float, default=1200.0)
    p.add_argument("--dr-km", type=float, default=12.0)
    p.add_argument("--max-z-km", type=float, default=20.0)
    p.add_argument("--f", type=float, default=5.464e-5)
    p.add_argument("--radial-smooth-sigma", type=float, default=1.0)
    return p.parse_args()


def safe_grad(a, x, axis):
    return np.gradient(a, x, axis=axis, edge_order=2 if len(x) >= 3 else 1)


def smooth_r(a, sigma):
    if sigma <= 0:
        return np.asarray(a, dtype=float)
    return gaussian_filter1d(np.asarray(a, dtype=float), sigma=sigma, axis=-1, mode="nearest")


def bin_mean(a, bins, valid, nr):
    return _azimuthal_average_by_radius(np.asarray(a), bins, valid, nr)


def weighted_mean(a, w, mask):
    use = mask & np.isfinite(a) & np.isfinite(w)
    return float(np.sum(a[use] * w[use]) / np.sum(w[use])) if np.any(use) else np.nan


def weighted_rms(a, w, mask):
    use = mask & np.isfinite(a) & np.isfinite(w)
    return float(np.sqrt(np.sum(a[use] ** 2 * w[use]) / np.sum(w[use]))) if np.any(use) else np.nan


def weighted_corr(a, b, w, mask):
    use = mask & np.isfinite(a) & np.isfinite(b) & np.isfinite(w)
    if np.count_nonzero(use) < 3:
        return np.nan
    x, y, ww = a[use], b[use], w[use]
    ww = ww / np.sum(ww)
    xa, ya = np.sum(ww * x), np.sum(ww * y)
    cov = np.sum(ww * (x - xa) * (y - ya))
    den = np.sqrt(np.sum(ww * (x - xa) ** 2) * np.sum(ww * (y - ya) ** 2))
    return float(cov / den) if den > 0 else np.nan


def load_3d(ds, name, idx, z_keep):
    data, dims = _get_time_slice(ds[name], idx)
    data, dims = _destagger_to_scalar_grid(data, dims)
    if tuple(dims) != ("zh", "yh", "xh"):
        raise ValueError(f"{name}: got {dims}, expected scalar (zh,yh,xh)")
    return np.asarray(data[z_keep], dtype=np.float64)


def load_2d(ds, name, idx):
    data, dims = _get_time_slice(ds[name], idx)
    data, dims = _destagger_to_scalar_grid(data, dims)
    if tuple(dims) != ("yh", "xh"):
        raise ValueError(f"{name}: got {dims}, expected (yh,xh)")
    return np.asarray(data, dtype=np.float64)


def resolve_index(time_s, hour):
    return int(np.argmin(np.abs(np.asarray(time_s) - hour * 3600.0)))


def geometry(xh, yh, xc, yc, max_r, dr):
    x, y = np.meshgrid(xh, yh)
    dx = (x - xc) * 1000.0
    dy = (y - yc) * 1000.0
    r2d_m = np.hypot(dx, dy)
    ang = np.arctan2(dy, dx)
    rb = np.arange(0.0, max_r + dr, dr)
    rc = 0.5 * (rb[:-1] + rb[1:])
    bi, valid = _compute_radial_bin_index(r2d_m / 1000.0, rb)
    return r2d_m, np.cos(ang), np.sin(ang), rc, bi, valid


def state_at(ds, vm, idx, z_keep, r2d, ct, st, bi, valid, nr, f, need_flow=True):
    u = load_3d(ds, vm["u"], idx, z_keep)
    v = load_3d(ds, vm["v"], idx, z_keep)
    rho = load_3d(ds, vm["rho"], idx, z_keep)
    ur = u * ct[None] + v * st[None]
    ut = -u * st[None] + v * ct[None]
    m3 = r2d[None] * ut + 0.5 * f * r2d[None] ** 2
    rho_bar = bin_mean(rho, bi, valid, nr)
    rho_safe = np.maximum(rho_bar, 1e-10)
    out = {
        "M_reynolds": bin_mean(m3, bi, valid, nr),
        "M_favre": bin_mean(rho * m3, bi, valid, nr) / rho_safe,
        "ut_reynolds": bin_mean(ut, bi, valid, nr),
        "ut_favre": bin_mean(rho * ut, bi, valid, nr) / rho_safe,
        "rho": rho_bar,
    }
    if need_flow:
        w = load_3d(ds, vm["w"], idx, z_keep)
        out.update({
            "u3": ur, "v3": ut, "w3": w, "rho3": rho,
            "ur_reynolds": bin_mean(ur, bi, valid, nr),
            "w_reynolds": bin_mean(w, bi, valid, nr),
            "ur_favre": bin_mean(rho * ur, bi, valid, nr) / rho_safe,
            "w_favre": bin_mean(rho * w, bi, valid, nr) / rho_safe,
        })
    return out


def budget_at(ds, idx, z_keep, r2d, ct, st, bi, valid, nr, rho3, rho_bar):
    reyn, favre = {}, {}
    rho_safe = np.maximum(rho_bar, 1e-10)
    for suf in BUDGET_SUFFIXES:
        un, vn = f"ub_{suf}", f"vb_{suf}"
        if un not in ds.variables or vn not in ds.variables:
            continue
        ub = load_3d(ds, un, idx, z_keep)
        vb = load_3d(ds, vn, idx, z_keep)
        tt = -ub * st[None] + vb * ct[None]
        tm3 = r2d[None] * tt
        reyn[suf] = bin_mean(tm3, bi, valid, nr)
        favre[suf] = bin_mean(rho3 * tm3, bi, valid, nr) / rho_safe
    return reyn, favre


def inertia_from_m(M, ut, r, f, sigma):
    ms = smooth_r(M, sigma)
    eta = safe_grad(ms, r, axis=1) / np.maximum(r[None], 1.0)
    xi = f + 2.0 * smooth_r(ut, sigma) / np.maximum(r[None], 1.0)
    return xi * eta, xi, eta


def map_m_tendency_to_i(tm, xi, eta, r, sigma):
    ts = smooth_r(tm, sigma)
    t_eta = safe_grad(ts, r, axis=1) / np.maximum(r[None], 1.0)
    t_xi = 2.0 * ts / np.maximum(r[None] ** 2, 1.0)
    return eta * t_xi + xi * t_eta


def decompose(form, center, minus, plus, budgets, direct, r, z, dt_minus,
              dt_plus, derivative_mode, f, sigma):
    M = center[f"M_{form}"]
    ut = center[f"ut_{form}"]
    ur = center[f"ur_{form}"]
    w = center[f"w_{form}"]
    if derivative_mode == "backward":
        local_m = (M - minus[f"M_{form}"]) / dt_minus
    elif derivative_mode == "forward":
        local_m = (plus[f"M_{form}"] - M) / dt_plus
    else:
        local_m = (plus[f"M_{form}"] - minus[f"M_{form}"]) / (dt_minus + dt_plus)
    dm_dr = safe_grad(smooth_r(M, sigma), r, axis=1)
    dm_dz = safe_grad(smooth_r(M, sigma), z, axis=0)
    mean_r = -smooth_r(ur, sigma) * dm_dr
    mean_z = -smooth_r(w, sigma) * dm_dz
    zero = np.zeros_like(M)
    hadv_cor = budgets.get("hadv", zero) + budgets.get("cor", zero)
    vadv = budgets.get("vadv", zero)
    eddy_r = hadv_cor - mean_r
    eddy_z = vadv - mean_z
    pgrad = budgets.get("pgrad", zero)
    diffusion = sum((budgets.get(k, zero) for k in DIFF_SUFFIXES), zero.copy())
    rdamp = budgets.get("rdamp", zero)
    model_sum = hadv_cor + vadv + pgrad + diffusion + rdamp
    residual_m = local_m - model_sum
    I0, xi, eta = inertia_from_m(M, ut, r, f, sigma)
    Im, _, _ = inertia_from_m(minus[f"M_{form}"], minus[f"ut_{form}"], r, f, sigma)
    Ip, _, _ = inertia_from_m(plus[f"M_{form}"], plus[f"ut_{form}"], r, f, sigma)
    if derivative_mode == "backward":
        local_i = (I0 - Im) / dt_minus
    elif derivative_mode == "forward":
        local_i = (Ip - I0) / dt_plus
    else:
        local_i = (Ip - Im) / (dt_minus + dt_plus)
    terms_m = {
        "mean_r": mean_r, "mean_z": mean_z, "eddy_r": eddy_r,
        "eddy_z": eddy_z, "pgrad": pgrad, "diffusion": diffusion,
        "rdamp": rdamp,
    }
    terms_i = {k: map_m_tendency_to_i(v, xi, eta, r, sigma) for k, v in terms_m.items()}
    model_i = sum(terms_i.values(), np.zeros_like(M))
    residual_i = local_i - model_i
    direct_m = {
        "direct_eddy_r": r[None] * direct["F_lambda_eddy_radial"],
        "direct_eddy_z": r[None] * direct["F_lambda_eddy_vertical"],
    }
    direct_i = {k: map_m_tendency_to_i(v, xi, eta, r, sigma) for k, v in direct_m.items()}
    return {
        "I": I0, "xi": xi, "eta": eta,
        "local_m": local_m, "model_m": model_sum, "residual_m": residual_m,
        "local_i": local_i, "model_i": model_i, "residual_i": residual_i,
        **{f"M_{k}": v for k, v in terms_m.items()},
        **{f"I_{k}": v for k, v in terms_i.items()},
        **{f"M_{k}": v for k, v in direct_m.items()},
        **{f"I_{k}": v for k, v in direct_i.items()},
    }


def instantaneous_decomposition(form, state, budgets, direct, r, z, f, sigma):
    """Return one-time budget terms before temporal averaging."""
    M = state[f"M_{form}"]
    ut = state[f"ut_{form}"]
    ur = state[f"ur_{form}"]
    w = state[f"w_{form}"]
    dm_dr = safe_grad(smooth_r(M, sigma), r, axis=1)
    dm_dz = safe_grad(smooth_r(M, sigma), z, axis=0)
    mean_r = -smooth_r(ur, sigma) * dm_dr
    mean_z = -smooth_r(w, sigma) * dm_dz
    zero = np.zeros_like(M)
    hadv_cor = budgets.get("hadv", zero) + budgets.get("cor", zero)
    vadv = budgets.get("vadv", zero)
    terms_m = {
        "mean_r": mean_r,
        "mean_z": mean_z,
        "eddy_r": hadv_cor - mean_r,
        "eddy_z": vadv - mean_z,
        "pgrad": budgets.get("pgrad", zero),
        "diffusion": sum((budgets.get(k, zero) for k in DIFF_SUFFIXES), zero.copy()),
        "rdamp": budgets.get("rdamp", zero),
    }
    model_m = sum(terms_m.values(), zero.copy())
    I0, xi, eta = inertia_from_m(M, ut, r, f, sigma)
    terms_i = {k: map_m_tendency_to_i(v, xi, eta, r, sigma) for k, v in terms_m.items()}
    direct_m = {
        "direct_eddy_r": r[None] * direct["F_lambda_eddy_radial"],
        "direct_eddy_z": r[None] * direct["F_lambda_eddy_vertical"],
    }
    direct_i = {k: map_m_tendency_to_i(v, xi, eta, r, sigma) for k, v in direct_m.items()}
    return {
        "M": M, "I": I0, "xi": xi, "eta": eta,
        "ur": smooth_r(ur, sigma), "model_m": model_m,
        "model_i": sum(terms_i.values(), zero.copy()),
        **{f"M_{k}": v for k, v in terms_m.items()},
        **{f"I_{k}": v for k, v in terms_i.items()},
        **{f"M_{k}": v for k, v in direct_m.items()},
        **{f"I_{k}": v for k, v in direct_i.items()},
    }


def temporal_average(arrays, times):
    stack = np.stack(arrays)
    if len(times) == 1:
        return stack[0]
    return np.trapezoid(stack, x=np.asarray(times), axis=0) / (times[-1] - times[0])


def window_decomposition(instants, times, r, sigma):
    """Compare endpoint tendency with the time-integrated instantaneous budget."""
    span = float(times[-1] - times[0])
    local_m = (instants[-1]["M"] - instants[0]["M"]) / span
    local_i_actual = (instants[-1]["I"] - instants[0]["I"]) / span
    keys = [k for k in instants[0] if k not in {"M", "I"}]
    avg = {k: temporal_average([q[k] for q in instants], times) for k in keys}
    avg["I"] = temporal_average([q["I"] for q in instants], times)
    avg["local_m"] = local_m
    # Use one linearized operator for every M-budget term.  This preserves the
    # exact additive attribution of M tendencies; the endpoint finite-
    # difference of xi*eta is retained as a nonlinear sensitivity diagnostic.
    avg["local_i"] = map_m_tendency_to_i(local_m, avg["xi"], avg["eta"], r, sigma)
    avg["local_i_actual"] = local_i_actual
    for name in ("mean_r", "mean_z", "eddy_r", "eddy_z", "pgrad", "diffusion", "rdamp"):
        avg[f"I_{name}"] = map_m_tendency_to_i(
            avg[f"M_{name}"], avg["xi"], avg["eta"], r, sigma)
    for name in ("direct_eddy_r", "direct_eddy_z"):
        avg[f"I_{name}"] = map_m_tendency_to_i(
            avg[f"M_{name}"], avg["xi"], avg["eta"], r, sigma)
    avg["model_i"] = sum((avg[f"I_{name}"] for name in
                          ("mean_r", "mean_z", "eddy_r", "eddy_z", "pgrad", "diffusion", "rdamp")),
                         np.zeros_like(local_m))
    avg["residual_m"] = local_m - avg["model_m"]
    avg["residual_i"] = avg["local_i"] - avg["model_i"]
    avg["nonlinear_i_difference"] = local_i_actual - avg["local_i"]
    return avg


def analyze_file(path, hours, args, label):
    cfg = PipelineConfig(input_file=path, max_r_km=args.max_r_km,
                         dr_km=args.dr_km, max_z_km=args.max_z_km,
                         coriolis_f=args.f)
    ds, _, _, _ = _open_dataset_robust(path)
    out = {}
    try:
        vm = _resolve_core_var_names(ds, cfg)
        xh = np.asarray(ds["xh"], float); yh = np.asarray(ds["yh"], float)
        zh_all = np.asarray(ds["zh"], float); z_keep = zh_all <= args.max_z_km
        z = zh_all[z_keep] * 1000.0
        time_s = np.asarray(ds["time"], float)
        for hour in hours:
            i0 = resolve_index(time_s, hour)
            im = resolve_index(time_s, hour - args.difference_hours)
            ip = resolve_index(time_s, hour + args.difference_hours)
            ps = load_2d(ds, vm["psfc"], i0)
            xc, yc = _find_center(ps, xh, yh, cfg)
            r2d, ct, st, r_km, bi, valid = geometry(
                xh, yh, xc, yc, args.max_r_km, args.dr_km)
            r = r_km * 1000.0; nr = len(r)
            inst_r, inst_f, window_times = [], [], []
            for j in range(im, ip + 1):
                state = state_at(ds, vm, j, z_keep, r2d, ct, st, bi, valid,
                                 nr, args.f, True)
                br, bf = budget_at(ds, j, z_keep, r2d, ct, st, bi, valid, nr,
                                   state["rho3"], state["rho"])
                direct_r = diagnose_eddy_momentum_forcing(
                    state["u3"], state["v3"], state["w3"], state["rho3"],
                    bi, valid, nr, r, z, averaging="reynolds")
                direct_f = diagnose_eddy_momentum_forcing(
                    state["u3"], state["v3"], state["w3"], state["rho3"],
                    bi, valid, nr, r, z, averaging="favre")
                inst_r.append(instantaneous_decomposition(
                    "reynolds", state, br, direct_r, r, z, args.f,
                    args.radial_smooth_sigma))
                inst_f.append(instantaneous_decomposition(
                    "favre", state, bf, direct_f, r, z, args.f,
                    args.radial_smooth_sigma))
                window_times.append(float(time_s[j]))
                del state, br, bf, direct_r, direct_f
            out[hour] = {
                "center": (xc, yc), "r_km": r_km, "z_km": z / 1000.0,
                "reynolds": window_decomposition(inst_r, window_times, r, args.radial_smooth_sigma),
                "favre": window_decomposition(inst_f, window_times, r, args.radial_smooth_sigma),
            }
            print(f"[{label}] completed {hour:g} h, window={2*args.difference_hours:g} h, "
                  f"center=({xc:.1f},{yc:.1f}) km")
    finally:
        ds.close()
    return out


def masks_and_weights(r_km, z_km):
    rr, zz = np.meshgrid(r_km, z_km)
    w = np.maximum(rr, 0.5 * np.nanmedian(np.diff(r_km)))
    masks = {k: ((rr >= b[0]) & (rr <= b[1]) & (zz >= b[2]) & (zz <= b[3]))
             for k, b in DOMAINS.items()}
    return masks, w


def collect_metrics(results, hours):
    r = results["noJET"][hours[0]]["r_km"]
    z = results["noJET"][hours[0]]["z_km"]
    masks, w = masks_and_weights(r, z)
    rows = []
    terms = ("mean_r", "mean_z", "eddy_r", "eddy_z", "pgrad", "diffusion", "rdamp")
    for case in ("noJET", "JET"):
        for hour in hours:
            for form in ("reynolds", "favre"):
                q = results[case][hour][form]
                for domain, mask in masks.items():
                    obs_rms = weighted_rms(q["local_i"], w, mask)
                    row = {
                        "case": case, "hour": hour, "form": form, "domain": domain,
                        "M_closure_corr": weighted_corr(q["local_m"], q["model_m"], w, mask),
                        "M_normalized_residual": weighted_rms(q["residual_m"], w, mask) /
                            max(weighted_rms(q["local_m"], w, mask), 1e-30),
                        "I_closure_corr": weighted_corr(q["local_i"], q["model_i"], w, mask),
                        "I_normalized_residual": weighted_rms(q["residual_i"], w, mask) / max(obs_rms, 1e-30),
                        "I_actual_vs_linear_corr": weighted_corr(q["local_i_actual"], q["local_i"], w, mask),
                        "I_nonlinear_normalized_difference": weighted_rms(q["nonlinear_i_difference"], w, mask) /
                            max(weighted_rms(q["local_i_actual"], w, mask), 1e-30),
                        "eddy_direct_r_corr": weighted_corr(q["M_eddy_r"], q["M_direct_eddy_r"], w, mask),
                        "eddy_direct_z_corr": weighted_corr(q["M_eddy_z"], q["M_direct_eddy_z"], w, mask),
                        "I_local_mean": weighted_mean(q["local_i"], w, mask),
                        "I_local_rms": obs_rms,
                        "M_local_mean": weighted_mean(q["local_m"], w, mask),
                        "M_local_rms": weighted_rms(q["local_m"], w, mask),
                        "M_residual_mean": weighted_mean(q["residual_m"], w, mask),
                        "M_residual_rms": weighted_rms(q["residual_m"], w, mask),
                    }
                    for t in terms:
                        row[f"I_{t}_mean"] = weighted_mean(q[f"I_{t}"], w, mask)
                        row[f"I_{t}_rms"] = weighted_rms(q[f"I_{t}"], w, mask)
                        row[f"M_{t}_mean"] = weighted_mean(q[f"M_{t}"], w, mask)
                        row[f"M_{t}_rms"] = weighted_rms(q[f"M_{t}"], w, mask)
                    row["I_residual_mean"] = weighted_mean(q["residual_i"], w, mask)
                    rows.append(row)
    return rows, masks, w


def nature_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("nature", ["#214478", "#f7f7f3", "#b2182b"])


def shared_limit(fields, pct=98):
    v = np.concatenate([np.ravel(np.abs(x[np.isfinite(x)])) for x in fields])
    return float(np.percentile(v, pct)) if v.size else 1.0


def figure_closure(results, hour, out):
    fields = []
    for case in ("noJET", "JET"):
        q = results[case][hour]["reynolds"]
        fields += [q["local_m"], q["model_m"], q["residual_m"]]
    r0 = results["JET"][hour]["r_km"]
    z0 = results["JET"][hour]["z_km"]
    view = (z0[:, None] >= 8) & (z0[:, None] <= 18) & (r0[None, :] >= 50) & (r0[None, :] <= 500)
    lim = shared_limit([np.where(view, f, np.nan) for f in fields])
    fig, ax = plt.subplots(2, 3, figsize=(15, 7.4), sharex=True, sharey=True)
    cmap = nature_cmap(); im = None
    for i, case in enumerate(("noJET", "JET")):
        q = results[case][hour]["reynolds"]
        r, z = results[case][hour]["r_km"], results[case][hour]["z_km"]
        for j, (key, title) in enumerate((("local_m", "Observed ∂M/∂t"),
                                           ("model_m", "Sum of all budget terms"),
                                           ("residual_m", "Observed − sum"))):
            im = ax[i, j].pcolormesh(r, z, q[key], cmap=cmap,
                                     vmin=-lim, vmax=lim, shading="auto")
            ax[i, j].set_xlim(50, 500); ax[i, j].set_ylim(8, 18)
            ax[i, j].set_title(f"{case}: {title}")
            ax[i, j].set_xlabel("Radius (km)")
        ax[i, 0].set_ylabel("Height (km)")
    fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.06, pad=0.10,
                 label=r"Absolute-angular-momentum tendency (m$^2$ s$^{-2}$)")
    fig.suptitle(f"Absolute-angular-momentum budget closure at {hour:g} h", fontweight="bold")
    fig.savefig(out, dpi=220, bbox_inches="tight"); plt.close(fig)


def figure_mechanisms(results, hour, out):
    keys = ("mean_r", "mean_z", "eddy_r", "eddy_z", "diffusion", "residual")
    titles = ("Mean radial advection", "Mean vertical advection", "Eddy radial transport",
              "Eddy vertical transport", "Diffusion/turbulence", "Unclosed residual")
    qj = results["JET"][hour]["reynolds"]; qc = results["noJET"][hour]["reynolds"]
    fs = [(qj[f"I_{k}"] - qc[f"I_{k}"]) if k != "residual" else
          (qj["residual_i"] - qc["residual_i"]) for k in keys]
    r, z = results["JET"][hour]["r_km"], results["JET"][hour]["z_km"]
    view = (z[:, None] >= 8) & (z[:, None] <= 18) & (r[None, :] >= 50) & (r[None, :] <= 500)
    lim = shared_limit([np.where(view, f, np.nan) for f in fs])
    fig, ax = plt.subplots(2, 3, figsize=(15, 7.4), sharex=True, sharey=True)
    im = None
    for a, f, title in zip(ax.flat, fs, titles):
        im = a.pcolormesh(r, z, f * 1e15, cmap=nature_cmap(), vmin=-lim*1e15,
                          vmax=lim*1e15, shading="auto")
        a.set_xlim(50, 500); a.set_ylim(8, 18); a.set_title(title); a.set_xlabel("Radius (km)")
    ax[0, 0].set_ylabel("Height (km)"); ax[1, 0].set_ylabel("Height (km)")
    fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.06, pad=0.10,
                 label=r"JET−noJET contribution to ∂I/∂t ($10^{-15}$ s$^{-3}$)")
    fig.suptitle(f"Mechanistic decomposition of inertial-stability tendency at {hour:g} h",
                 fontweight="bold")
    fig.savefig(out, dpi=220, bbox_inches="tight"); plt.close(fig)


def figure_timeseries(rows, hours, out):
    use = [r for r in rows if r["form"] == "reynolds" and r["domain"] == "inner_outflow"]
    by = {(r["case"], float(r["hour"])): r for r in use}
    terms = ("mean_r", "mean_z", "eddy_r", "eddy_z", "diffusion")
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for t in terms:
        vals = [(by[("JET", h)][f"I_{t}_mean"] - by[("noJET", h)][f"I_{t}_mean"]) * 1e15
                for h in hours]
        ax[0].plot(hours, vals, marker="o", label=t.replace("_", " "), color=COLORS[t])
    ax[0].axhline(0, color="0.4", lw=0.8); ax[0].set_xlabel("Time (h)")
    ax[0].set_ylabel(r"JET−noJET domain mean ($10^{-15}$ s$^{-3}$)")
    ax[0].set_title("Inner-outflow mechanism contributions"); ax[0].legend(frameon=False, fontsize=8)
    for form, ls in (("reynolds", "-"), ("favre", "--")):
        u = [r for r in rows if r["form"] == form and r["domain"] == "inner_outflow"]
        b = {(r["case"], float(r["hour"])): r for r in u}
        for case, color in (("noJET", "#0072B2"), ("JET", "#D55E00")):
            ax[1].plot(hours, [b[(case, h)]["I_normalized_residual"] for h in hours],
                       marker="o", ls=ls, color=color, label=f"{case} {form}")
    ax[1].axhline(1, color="0.5", lw=0.8); ax[1].set_xlabel("Time (h)")
    ax[1].set_ylabel("RMS closure residual / RMS observed")
    ax[1].set_title("Inertial-stability budget closure"); ax[1].legend(frameon=False, fontsize=8)
    for a in ax: a.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out, dpi=220, bbox_inches="tight"); plt.close(fig)


def figure_eddy_validation(rows, hours, out):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for j, component in enumerate(("r", "z")):
        for form, marker in (("reynolds", "o"), ("favre", "s")):
            for case, color in (("noJET", "#0072B2"), ("JET", "#D55E00")):
                sel = [r for r in rows if r["form"] == form and r["case"] == case
                       and r["domain"] == "inner_outflow"]
                ax[j].plot(hours, [r[f"eddy_direct_{component}_corr"] for r in sel],
                           marker=marker, color=color, ls="-" if form == "reynolds" else "--",
                           label=f"{case} {form}")
        ax[j].axhline(0, color="0.5", lw=0.8); ax[j].set_ylim(-1, 1)
        ax[j].set_xlabel("Time (h)"); ax[j].set_title(f"{component.upper()} eddy: CM1 residual vs direct flux")
        ax[j].grid(alpha=0.2)
    ax[0].set_ylabel("Spatial correlation in 50–350 km, 10–16 km")
    ax[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(out, dpi=220, bbox_inches="tight"); plt.close(fig)


def save_outputs(results, rows, hours, out):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "angular_momentum_budget_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    h = 72.0 if 72.0 in hours else hours[len(hours)//2]
    payload = {}
    for case in ("noJET", "JET"):
        for form in ("reynolds", "favre"):
            for key, val in results[case][h][form].items():
                if isinstance(val, np.ndarray): payload[f"{case}_{form}_{key}"] = val
    payload["r_km"] = results["JET"][h]["r_km"]; payload["z_km"] = results["JET"][h]["z_km"]
    np.savez_compressed(out / "angular_momentum_budget_72h.npz", **payload)

    # Save compact Reynolds fields for every requested time.  These fields are
    # needed to connect the M budget to eta=(1/r)dM/dr and raw Bui I2 without
    # re-reading the full 3-D CM1 files during post-processing.
    all_payload = {
        "hours": np.asarray(hours, dtype=float),
        "r_km": results["JET"][hours[0]]["r_km"],
        "z_km": results["JET"][hours[0]]["z_km"],
    }
    keep = (
        "I", "xi", "eta", "ur", "local_m", "model_m", "residual_m",
        "M_mean_r", "M_mean_z", "M_eddy_r", "M_eddy_z",
        "M_pgrad", "M_diffusion", "M_rdamp",
    )
    for case in ("noJET", "JET"):
        for hour in hours:
            q = results[case][hour]["reynolds"]
            tag = f"{case}_{hour:g}h"
            for key in keep:
                all_payload[f"{tag}_{key}"] = q[key]
    np.savez_compressed(out / "angular_momentum_budget_all_times.npz", **all_payload)
    figure_closure(results, h, out / "figure1_budget_closure_72h.png")
    figure_mechanisms(results, h, out / "figure2_mechanism_terms_72h.png")
    figure_timeseries(rows, hours, out / "figure3_mechanism_time_evolution.png")
    figure_eddy_validation(rows, hours, out / "figure4_direct_eddy_validation.png")


def summary(rows, out):
    inner = [r for r in rows if r["domain"] == "inner_outflow"]
    def med(key, form="reynolds"):
        x = [float(r[key]) for r in inner if r["form"] == form and np.isfinite(float(r[key]))]
        return float(np.median(x)) if x else np.nan
    s = {
        "verification_status": "ANALYZED",
        "interpretation": "mechanistic budget attribution; not exclusive causal identification",
        "inner_outflow_median": {
            "reynolds_M_normalized_residual": med("M_normalized_residual"),
            "reynolds_I_normalized_residual": med("I_normalized_residual"),
            "favre_M_normalized_residual": med("M_normalized_residual", "favre"),
            "favre_I_normalized_residual": med("I_normalized_residual", "favre"),
            "reynolds_direct_radial_eddy_corr": med("eddy_direct_r_corr"),
            "reynolds_direct_vertical_eddy_corr": med("eddy_direct_z_corr"),
            "favre_direct_radial_eddy_corr": med("eddy_direct_r_corr", "favre"),
            "favre_direct_vertical_eddy_corr": med("eddy_direct_z_corr", "favre"),
        },
        "warning": "Strength is a jet-mediated variable; one experiment pair cannot isolate an exclusive direct effect.",
    }
    (out / "angular_momentum_budget_summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")


def main():
    a = parse_args(); hours = [float(x) for x in a.hours.split(",")]
    out = Path(a.output_dir)
    print("[PLAN] fixed-centre CM1 budget closure + Reynolds/Favre direct-flux cross-check")
    results = {
        "noJET": analyze_file(a.nojet, hours, a, "noJET"),
        "JET": analyze_file(a.jet, hours, a, "JET"),
    }
    rows, _, _ = collect_metrics(results, hours)
    save_outputs(results, rows, hours, out)
    summary(rows, out)
    print(f"[DONE] outputs: {out}")


if __name__ == "__main__":
    main()
