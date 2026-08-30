#!/usr/bin/env python3
"""Multi-angle attribution of JET-minus-CTRL inertial-stability changes.

The analysis deliberately separates three descriptive quantities:
1) the same-time total JET-minus-CTRL difference;
2) an intensity/evolution-mediated estimate obtained by matching JET states to
   CTRL states in (Pmin, low-level Vtmax, RMW) space;
3) the remaining matched-intensity structural difference.

It also diagnoses the instantaneous contribution of the environmental eddy
tangential acceleration to absolute-vorticity and classical inertial-stability
tendencies.  These are mechanism diagnostics, not a proof of exclusive cause.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import netCDF4
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src._se_pipeline_single import PipelineConfig, azimuthal_average_from_3d
from src.se_applicability import compute_case_stability


DOMAINS = {
    "core_outflow": (50.0, 200.0, 10.0, 16.0),
    "inner_outflow": (50.0, 350.0, 10.0, 16.0),
    "outer_outflow": (200.0, 500.0, 10.0, 16.0),
    "jet_region": (750.0, 1100.0, 10.0, 17.0),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nojet", required=True)
    p.add_argument("--jet", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--start-hour", type=float, default=20.0)
    p.add_argument("--end-hour", type=float, default=120.0)
    p.add_argument("--step-hour", type=float, default=5.0)
    p.add_argument("--extra-hours", type=float, nargs="*", default=[72.0])
    p.add_argument("--max-r-km", type=float, default=1200.0)
    p.add_argument("--dr-km", type=float, default=12.0)
    p.add_argument("--max-z-km", type=float, default=20.0)
    p.add_argument("--f", type=float, default=5.464e-5)
    p.add_argument("--eddy-average", choices=["reynolds", "favre"], default="reynolds")
    p.add_argument("--radial-smooth-sigma", type=float, default=1.0)
    p.add_argument("--center-window", type=int, default=21)
    p.add_argument("--center-method", choices=["min", "mean"], default="min")
    return p.parse_args()


def selected_hours(a):
    regular = np.arange(a.start_hour, a.end_hour + 0.1 * a.step_hour, a.step_hour)
    hours = np.unique(np.concatenate((regular, np.asarray(a.extra_hours, dtype=float))))
    return hours[(hours >= a.start_hour) & (hours <= a.end_hour)], regular


def read_pmin(path, hours):
    result = np.full(hours.size, np.nan)
    with netCDF4.Dataset(path) as ds:
        time_h = np.asarray(ds.variables["time"][:], dtype=float) / 3600.0
        ps = ds.variables["psfc"]
        for i, h in enumerate(hours):
            it = int(np.argmin(np.abs(time_h - h)))
            field = np.asarray(ps[it], dtype=float)
            result[i] = np.nanmin(gaussian_filter(field, sigma=2.0, mode="nearest")) / 100.0
    return result


def analyze_case(path, hours, pmin, a, label):
    records = []
    for i, hour in enumerate(hours):
        print(f"[ATTR] {label}: {hour:g} h ({i + 1}/{len(hours)})", flush=True)
        cfg = PipelineConfig(
            input_file=path,
            output_dir=str(Path(a.output_dir) / "scratch"),
            target_time_hours=float(hour),
            max_r_km=a.max_r_km,
            dr_km=a.dr_km,
            max_z_km=a.max_z_km,
            center_window=a.center_window,
            center_method=a.center_method,
            coriolis_f=a.f,
            eddy_average=a.eddy_average,
            include_model_budget_terms=False,
            write_netcdf=False,
            write_ieee=False,
            plot_solution=False,
        )
        avg = azimuthal_average_from_3d(cfg)
        r_km = np.asarray(avg["r_km"], dtype=float)
        z_km = np.asarray(avg["z_km"], dtype=float)
        stability, _ = compute_case_stability(
            avg["ut"], avg["theta"], avg["rho"], r_km * 1000.0,
            z_km * 1000.0, a.f,
        )
        low = np.where(z_km <= 2.0)[0]
        radial = np.where((r_km >= max(a.dr_km, 6.0)) & (r_km <= 300.0))[0]
        low_ut = np.nanmean(np.asarray(avg["ut"])[np.ix_(low, radial)], axis=0)
        imax = int(np.nanargmax(low_ut))
        vtmax = float(low_ut[imax])
        rmw = float(r_km[radial[imax]])
        xi = np.asarray(stability["xi_raw"], dtype=float)
        eta = np.asarray(stability["zeta_abs_raw"], dtype=float)
        classic = xi * eta
        rec = {
            "hour": float(np.asarray(avg["time_seconds_used"])[0] / 3600.0),
            "pmin_hpa": float(pmin[i]), "vtmax_m_s": vtmax, "rmw_km": rmw,
            "center_x_km": float(np.asarray(avg["center_x_km"])[0]),
            "center_y_km": float(np.asarray(avg["center_y_km"])[0]),
            "r_km": r_km, "z_km": z_km,
            "ut": np.asarray(avg["ut"], dtype=float),
            "ur": np.asarray(avg["ur"], dtype=float),
            "F_eddy": np.asarray(avg["F_lambda_eddy"], dtype=float),
            "F_eddy_r": np.asarray(avg["F_lambda_eddy_radial"], dtype=float),
            "F_eddy_z": np.asarray(avg["F_lambda_eddy_vertical"], dtype=float),
            "I2": np.asarray(stability["I2_raw"], dtype=float),
            "I2_vort": np.asarray(stability["I2_vorticity_component_raw"], dtype=float),
            "I2_baroc": np.asarray(stability["I2_baroclinic_component_raw"], dtype=float),
            "D": np.asarray(stability["D_raw"], dtype=float),
            "classes": np.asarray(stability["stability_class"], dtype=np.int8),
            "xi": xi, "eta": eta,
            "chi": np.asarray(stability["chi_raw"], dtype=float),
            "classic": classic,
        }
        records.append(rec)
    return records


def stack(records, key):
    return np.stack([r[key] for r in records], axis=0)


def cell_weights(r_km, z_km):
    r = np.asarray(r_km) * 1000.0
    z = np.asarray(z_km) * 1000.0
    dr = np.abs(np.gradient(r))
    dz = np.abs(np.gradient(z))
    return np.maximum(r, 1.0)[None, :] * dr[None, :] * dz[:, None]


def domain_mask(r_km, z_km, bounds):
    r0, r1, z0, z1 = bounds
    return ((z_km[:, None] >= z0) & (z_km[:, None] <= z1)
            & (r_km[None, :] >= r0) & (r_km[None, :] <= r1))


def wmean(field, weights, mask):
    valid = mask & np.isfinite(field)
    den = np.sum(weights[valid])
    return float(np.sum(weights[valid] * field[valid]) / den) if den > 0 else np.nan


def wrms(field, weights, mask):
    valid = mask & np.isfinite(field)
    den = np.sum(weights[valid])
    return float(np.sqrt(np.sum(weights[valid] * field[valid] ** 2) / den)) if den > 0 else np.nan


def wfraction(condition, weights, mask):
    valid = mask & np.isfinite(condition)
    den = np.sum(weights[valid])
    return float(np.sum(weights[valid] * condition[valid]) / den) if den > 0 else np.nan


def wcorr(x, y, weights, mask):
    valid = mask & np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 4:
        return np.nan
    w = weights[valid]
    xv, yv = x[valid], y[valid]
    w = w / np.sum(w)
    xm, ym = np.sum(w * xv), np.sum(w * yv)
    num = np.sum(w * (xv - xm) * (yv - ym))
    den = np.sqrt(np.sum(w * (xv - xm) ** 2) * np.sum(w * (yv - ym) ** 2))
    return float(num / den) if den > 0 else np.nan


def strength_match(ctrl, jet, hours):
    pc = np.array([r["pmin_hpa"] for r in ctrl])
    pj = np.array([r["pmin_hpa"] for r in jet])
    vc = np.array([r["vtmax_m_s"] for r in ctrl])
    vj = np.array([r["vtmax_m_s"] for r in jet])
    rc = np.array([r["rmw_km"] for r in ctrl])
    rj = np.array([r["rmw_km"] for r in jet])
    dp_c = np.gradient(pc, hours)
    dp_j = np.gradient(pj, hours)
    scales = np.array([
        max(2.0, np.nanstd(np.r_[pc, pj]) * 0.35),
        max(2.0, np.nanstd(np.r_[vc, vj]) * 0.35),
        max(18.0, np.nanstd(np.r_[rc, rj]) * 0.50),
    ])
    weights = np.array([0.45, 0.45, 0.10])
    matches = []
    for i, h in enumerate(hours):
        candidates = np.where(np.abs(hours - h) <= 35.0)[0]
        same_stage = candidates[np.sign(dp_c[candidates]) == np.sign(dp_j[i])]
        if same_stage.size:
            candidates = same_stage
        target = np.array([pj[i], vj[i], rj[i]])
        values = np.column_stack((pc[candidates], vc[candidates], rc[candidates]))
        cost = np.sum(weights * ((values - target) / scales) ** 2, axis=1)
        cost += 0.035 * ((hours[candidates] - h) / 10.0) ** 2
        j = int(candidates[np.argmin(cost)])
        matches.append({
            "jet_index": i, "ctrl_index": j,
            "jet_hour": float(h), "ctrl_hour": float(hours[j]),
            "cost": float(np.min(cost)),
            "delta_pmin_hpa": float(pj[i] - pc[j]),
            "delta_vtmax_m_s": float(vj[i] - vc[j]),
            "delta_rmw_km": float(rj[i] - rc[j]),
        })
    return matches


def direct_tendencies(ctrl, jet, r_km, sigma):
    r_m = r_km * 1000.0
    r_safe = np.maximum(r_m, 0.5 * np.nanmin(np.diff(r_m)))
    f_env = stack(jet, "F_eddy") - stack(ctrl, "F_eddy")
    if sigma > 0:
        f_use = gaussian_filter1d(f_env, sigma=sigma, axis=-1, mode="nearest")
    else:
        f_use = f_env.copy()
    t_eta = np.gradient(f_use, r_m, axis=-1, edge_order=2) + f_use / r_safe[None, None, :]
    t_xi = 2.0 * f_use / r_safe[None, None, :]
    xi_mid = 0.5 * (stack(ctrl, "xi") + stack(jet, "xi"))
    eta_mid = 0.5 * (stack(ctrl, "eta") + stack(jet, "eta"))
    chi_mid = 0.5 * (stack(ctrl, "chi") + stack(jet, "chi"))
    t_classic = eta_mid * t_xi + xi_mid * t_eta
    t_bui_vort = chi_mid * t_classic
    return f_env, t_eta, t_xi, t_classic, t_bui_vort


def nature_diverging():
    return LinearSegmentedColormap.from_list(
        "nature_div", ["#2F4B7C", "#7192BE", "#D1DEEB", "#F7F7F4",
                       "#F2C1AE", "#E56B4E", "#9D2933"], N=256)


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.linewidth": 0.75, "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.75, "ytick.major.width": 0.75,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def add_letter(ax, letter):
    ax.text(-0.12, 1.04, f"({letter})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom")


def figure_intensity(ctrl, jet, hours, matches, out):
    style()
    pc, pj = [np.array([r[k] for r in x]) for x, k in ((ctrl, "pmin_hpa"), (jet, "pmin_hpa"))]
    vc, vj = [np.array([r[k] for r in x]) for x, k in ((ctrl, "vtmax_m_s"), (jet, "vtmax_m_s"))]
    rc, rj = [np.array([r[k] for r in x]) for x, k in ((ctrl, "rmw_km"), (jet, "rmw_km"))]
    mi = np.array([m["ctrl_index"] for m in matches], dtype=int)
    fig, ax = plt.subplots(2, 3, figsize=(13.0, 7.2), constrained_layout=True)
    colors = {"CTRL": "#3B6FB6", "JET": "#D44B3E"}
    for y1, y2, title, ylabel, a0 in [
        (pc, pj, "Minimum surface pressure", "Pressure (hPa)", ax[0, 0]),
        (vc, vj, "Low-level azimuthal Vt max", r"Wind (m s$^{-1}$)", ax[0, 1]),
        (rc, rj, "Radius of maximum Vt", "RMW (km)", ax[0, 2]),
    ]:
        a0.plot(hours, y1, "o-", ms=3, lw=1.4, color=colors["CTRL"], label="noJET")
        a0.plot(hours, y2, "s-", ms=3, lw=1.4, color=colors["JET"], label="JET")
        a0.set_title(title); a0.set_xlabel("Time (h)"); a0.set_ylabel(ylabel); a0.grid(alpha=0.2)
    ax[0, 0].invert_yaxis(); ax[0, 0].legend(frameon=False)
    ax[1, 0].plot(pc, vc, "o-", ms=3, lw=1.1, color=colors["CTRL"], label="noJET trajectory")
    ax[1, 0].plot(pj, vj, "s-", ms=3, lw=1.1, color=colors["JET"], label="JET trajectory")
    for i in range(0, len(hours), max(1, len(hours) // 7)):
        ax[1, 0].plot([pj[i], pc[mi[i]]], [vj[i], vc[mi[i]]], color="0.65", lw=0.7)
    ax[1, 0].invert_xaxis(); ax[1, 0].set_xlabel("Pmin (hPa)"); ax[1, 0].set_ylabel(r"Vt max (m s$^{-1}$)")
    ax[1, 0].set_title("Intensity-space matching"); ax[1, 0].legend(frameon=False, fontsize=8); ax[1, 0].grid(alpha=0.2)
    ax[1, 1].axhline(0, color="0.35", lw=0.8)
    ax[1, 1].plot(hours, pj - pc[mi], "o-", label="Pmin mismatch (hPa)", color="#7A5195")
    ax[1, 1].plot(hours, vj - vc[mi], "s-", label="Vtmax mismatch (m/s)", color="#EF8354")
    ax[1, 1].set_title("Residual mismatch after matching"); ax[1, 1].set_xlabel("JET time (h)"); ax[1, 1].legend(frameon=False, fontsize=8); ax[1, 1].grid(alpha=0.2)
    ax[1, 2].plot(hours, hours, color="0.6", lw=1, ls="--", label="same time")
    ax[1, 2].plot(hours, hours[mi], "o-", color="#00876C", ms=3, label="matched CTRL time")
    ax[1, 2].set_title("Time mapping for matched intensity"); ax[1, 2].set_xlabel("JET time (h)"); ax[1, 2].set_ylabel("CTRL matched time (h)"); ax[1, 2].legend(frameon=False, fontsize=8); ax[1, 2].grid(alpha=0.2)
    for i, a0 in enumerate(ax.flat): add_letter(a0, "abcdef"[i])
    fig.suptitle("Intensity evolution and matched-state counterfactual", fontsize=14, fontweight="bold")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)


def figure_stability(ctrl, jet, hours, matches, r_km, z_km, weights, masks, out):
    style(); cmap = nature_diverging(); mi = np.array([m["ctrl_index"] for m in matches], int)
    Ic, Ij = stack(ctrl, "I2"), stack(jet, "I2")
    classic_c, classic_j = stack(ctrl, "classic"), stack(jet, "classic")
    xi_c, xi_j = stack(ctrl, "xi"), stack(jet, "xi")
    eta_c, eta_j = stack(ctrl, "eta"), stack(jet, "eta")
    total = Ij - Ic
    mediated = Ic[mi] - Ic
    residual = Ij - Ic[mi]
    eta_bar = 0.5 * (eta_j + eta_c); xi_bar = 0.5 * (xi_j + xi_c)
    wind_comp = eta_bar * (xi_j - xi_c)
    vort_comp = xi_bar * (eta_j - eta_c)
    mask = masks["inner_outflow"]
    unstable_c = np.array([wfraction(x <= 0, weights, mask) for x in Ic])
    unstable_j = np.array([wfraction(x <= 0, weights, mask) for x in Ij])
    mean_total = np.array([wmean(x, weights, mask) for x in total])
    mean_med = np.array([wmean(x, weights, mask) for x in mediated])
    mean_res = np.array([wmean(x, weights, mask) for x in residual])
    mean_wind = np.array([wmean(x, weights, mask) for x in wind_comp])
    mean_vort = np.array([wmean(x, weights, mask) for x in vort_comp])
    mean_classic_total = np.array([
        wmean(x, weights, mask) for x in (classic_j - classic_c)
    ])
    k = int(np.argmin(np.abs(hours - 72.0)))
    fig, ax = plt.subplots(2, 3, figsize=(13.2, 7.5), constrained_layout=True)
    ax[0, 0].plot(hours, unstable_c * 100, "o-", color="#3B6FB6", label="noJET")
    ax[0, 0].plot(hours, unstable_j * 100, "s-", color="#D44B3E", label="JET")
    ax[0, 0].set_title("Inertially unstable fraction"); ax[0, 0].set_ylabel("Area fraction (%)"); ax[0, 0].set_xlabel("Time (h)"); ax[0, 0].legend(frameon=False); ax[0, 0].grid(alpha=0.2)
    ax[0, 1].plot(hours, mean_total * 1e12, "k-", lw=1.6, label="same-time total")
    ax[0, 1].plot(hours, mean_med * 1e12, "--", lw=1.4, color="#E69F00", label="intensity/evolution mediated")
    ax[0, 1].plot(hours, mean_res * 1e12, "-.", lw=1.4, color="#009E73", label="matched-intensity residual")
    ax[0, 1].axhline(0, color="0.5", lw=0.7); ax[0, 1].set_title("Bui I² counterfactual decomposition")
    ax[0, 1].set_ylabel(r"Domain mean ($10^{-12}$)"); ax[0, 1].set_xlabel("Time (h)"); ax[0, 1].legend(frameon=False, fontsize=7.5); ax[0, 1].grid(alpha=0.2)
    ax[0, 2].plot(hours, mean_classic_total * 1e9, color="k", lw=1.4, label="total")
    ax[0, 2].plot(hours, mean_wind * 1e9, color="#CC6677", lw=1.3, label=r"wind factor: $\bar\eta\Delta\xi$")
    ax[0, 2].plot(hours, mean_vort * 1e9, color="#4477AA", lw=1.3, label=r"vorticity: $\bar\xi\Delta\eta$")
    ax[0, 2].axhline(0, color="0.5", lw=0.7); ax[0, 2].set_title("Classic I exact decomposition")
    ax[0, 2].set_ylabel(r"Contribution ($10^{-9}$ s$^{-2}$)"); ax[0, 2].set_xlabel("Time (h)"); ax[0, 2].legend(frameon=False, fontsize=7.5); ax[0, 2].grid(alpha=0.2)
    fields = [total[k] * 1e12, mediated[k] * 1e12, residual[k] * 1e12]
    finite = np.concatenate([np.abs(x[masks["inner_outflow"]]) for x in fields])
    vmax = max(float(np.nanpercentile(finite, 98)), 1e-6)
    titles = ["72 h total JET−noJET", "CTRL evolution/strength estimate", "Matched-intensity jet residual"]
    rr, zz = np.meshgrid(r_km, z_km)
    for j, (a0, field, title) in enumerate(zip(ax[1], fields, titles)):
        pm = a0.pcolormesh(rr, zz, field, shading="auto", cmap=cmap,
                           norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), rasterized=True)
        a0.contour(rr, zz, jet[k]["ur"], levels=[2, 5, 10], colors="#009E73", linewidths=0.7)
        a0.set_xlim(0, 500); a0.set_ylim(8, 18); a0.set_title(title); a0.set_xlabel("Radius (km)")
        if j == 0: a0.set_ylabel("Height (km)")
    cb = fig.colorbar(pm, ax=ax[1], orientation="horizontal", shrink=0.82, pad=0.12, aspect=35, extend="both")
    cb.set_label(r"Bui $\Delta I^2$ ($10^{-12}$; raw state)")
    for i, a0 in enumerate(ax.flat): add_letter(a0, "abcdef"[i])
    fig.suptitle("Inertial-stability attribution: same time versus matched intensity", fontsize=14, fontweight="bold")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    return {"total": total, "mediated": mediated, "residual": residual,
            "wind_comp": wind_comp, "vort_comp": vort_comp,
            "unstable_ctrl": unstable_c, "unstable_jet": unstable_j}


def lag_metrics(hours, regular_hours, t_direct, observed, weights, masks):
    out = {}
    hour_to_idx = {round(float(h), 6): i for i, h in enumerate(hours)}
    for name, mask in masks.items():
        rows = []
        for lag in (0.0, 5.0, 10.0, 15.0):
            pairs = []
            for h in regular_hours:
                if round(float(h), 6) in hour_to_idx and round(float(h + lag), 6) in hour_to_idx:
                    pairs.append((hour_to_idx[round(float(h), 6)], hour_to_idx[round(float(h + lag), 6)]))
            spatial = [wcorr(t_direct[i], observed[j], weights, mask) for i, j in pairs]
            x = np.array([wmean(t_direct[i], weights, mask) for i, _ in pairs])
            y = np.array([wmean(observed[j], weights, mask) for _, j in pairs])
            valid = np.isfinite(x) & np.isfinite(y)
            tcorr = float(np.corrcoef(x[valid], y[valid])[0, 1]) if np.count_nonzero(valid) >= 4 and np.std(x[valid]) > 0 and np.std(y[valid]) > 0 else np.nan
            rows.append({"lag_hours": lag, "median_spatial_correlation": float(np.nanmedian(spatial)), "domain_mean_time_correlation": tcorr, "n_pairs": len(pairs)})
        out[name] = rows
    return out


def figure_torque(ctrl, jet, hours, regular_hours, r_km, z_km, weights, masks, sigma, out):
    style(); cmap = nature_diverging()
    f_env, t_eta, t_xi, t_classic, t_direct = direct_tendencies(ctrl, jet, r_km, sigma)
    delta_vort = stack(jet, "I2_vort") - stack(ctrl, "I2_vort")
    observed = np.gradient(delta_vort, hours * 3600.0, axis=0, edge_order=2)
    lag = lag_metrics(hours, regular_hours, t_direct, observed, weights, masks)
    k = int(np.argmin(np.abs(hours - 72.0)))
    rr, zz = np.meshgrid(r_km, z_km)
    fig, ax = plt.subplots(2, 3, figsize=(13.2, 7.5), constrained_layout=True)
    map_fields = [f_env[k] * 1e3, t_direct[k] * 1e15, observed[k] * 1e15]
    titles = [r"$F_{\lambda,env}$", "Direct eddy I² tendency", "Observed tendency of ΔI² vorticity part"]
    labels = [r"$10^{-3}$ m s$^{-2}$", r"Scaled Bui-I² tendency ($10^{-15}$)", r"Scaled Bui-I² tendency ($10^{-15}$)"]
    for j, (a0, field, title, label) in enumerate(zip(ax[0], map_fields, titles, labels)):
        subset = np.abs(field[masks["inner_outflow"]])
        vmax = max(float(np.nanpercentile(subset, 98)), 1e-9)
        pm = a0.pcolormesh(rr, zz, field, shading="auto", cmap=cmap,
                           norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), rasterized=True)
        a0.contour(rr, zz, jet[k]["ur"], levels=[2, 5, 10], colors="#009E73", linewidths=0.7)
        a0.set_xlim(0, 500); a0.set_ylim(8, 18); a0.set_title(title); a0.set_xlabel("Radius (km)")
        if j == 0: a0.set_ylabel("Height (km)")
        cb = fig.colorbar(pm, ax=a0, orientation="horizontal", pad=0.12, aspect=28, extend="both"); cb.set_label(label)
    colors = ["#0072B2", "#D55E00", "#009E73", "#7A5195"]
    for color, (name, rows) in zip(colors, lag.items()):
        ax[1, 0].plot([x["lag_hours"] for x in rows], [x["median_spatial_correlation"] for x in rows], "o-", color=color, label=name)
        ax[1, 1].plot([x["lag_hours"] for x in rows], [x["domain_mean_time_correlation"] for x in rows], "o-", color=color, label=name)
    for a0, title in ((ax[1, 0], "Median spatial pattern correlation"), (ax[1, 1], "Domain-mean time correlation")):
        a0.axhline(0, color="0.5", lw=0.7); a0.set_xlabel("Lag after eddy forcing (h)"); a0.set_ylabel("Correlation"); a0.set_title(title); a0.grid(alpha=0.2)
    ax[1, 0].legend(frameon=False, fontsize=7.2, ncol=2)
    mask = masks["inner_outflow"]
    valid = mask & np.isfinite(t_direct[k]) & np.isfinite(observed[k])
    x = (t_direct[k][valid] * 1e15); y = (observed[k][valid] * 1e15)
    if x.size > 3000:
        idx = np.linspace(0, x.size - 1, 3000).astype(int); x, y = x[idx], y[idx]
    ax[1, 2].scatter(x, y, s=7, alpha=0.35, color="#4C78A8", edgecolors="none")
    corr72 = wcorr(t_direct[k], observed[k], weights, mask)
    ax[1, 2].set_title(f"72 h spatial agreement (r={corr72:.2f})")
    ax[1, 2].set_xlabel("Direct eddy tendency"); ax[1, 2].set_ylabel("Observed ΔI² tendency"); ax[1, 2].grid(alpha=0.2)
    for i, a0 in enumerate(ax.flat): add_letter(a0, "abcdef"[i])
    fig.suptitle("Does environmental eddy torque directly reorganize inertial stability?", fontsize=14, fontweight="bold")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    return {"F_env": f_env, "T_eta": t_eta, "T_xi": t_xi, "T_classic": t_classic,
            "T_bui_vort": t_direct, "observed_tendency": observed, "lag": lag,
            "corr72_inner": corr72}


def figure_profiles(ctrl, jet, hours, matches, r_km, z_km, masks, weights, attribution, torque, out):
    style(); k = int(np.argmin(np.abs(hours - 72.0))); km = matches[k]["ctrl_index"]
    zmask = (z_km >= 10) & (z_km <= 16)
    def zavg(a): return np.nanmean(a[zmask], axis=0)
    f = 5.464e-5; rmw_c = ctrl[k]["rmw_km"]; rmw_j = jet[k]["rmw_km"]; rmw_m = ctrl[km]["rmw_km"]
    M_c = r_km * 1000 * zavg(ctrl[k]["ut"]) + 0.5 * f * (r_km * 1000) ** 2
    M_j = r_km * 1000 * zavg(jet[k]["ut"]) + 0.5 * f * (r_km * 1000) ** 2
    M_m = r_km * 1000 * zavg(ctrl[km]["ut"]) + 0.5 * f * (r_km * 1000) ** 2
    eta_c, eta_j, eta_m = zavg(ctrl[k]["eta"]), zavg(jet[k]["eta"]), zavg(ctrl[km]["eta"])
    fig, ax = plt.subplots(2, 2, figsize=(10.8, 7.8), constrained_layout=True)
    for x, y, label, color in ((r_km / rmw_c, M_c / 1e6, "noJET same time", "#3B6FB6"), (r_km / rmw_j, M_j / 1e6, "JET", "#D44B3E"), (r_km / rmw_m, M_m / 1e6, f"noJET matched ({hours[km]:g} h)", "#009E73")):
        ax[0, 0].plot(x, y, lw=1.5, label=label, color=color)
    ax[0, 0].set_xlim(0, 12); ax[0, 0].set_title("Absolute angular momentum"); ax[0, 0].set_xlabel("r/RMW"); ax[0, 0].set_ylabel(r"M ($10^6$ m$^2$ s$^{-1}$)"); ax[0, 0].legend(frameon=False, fontsize=8); ax[0, 0].grid(alpha=0.2)
    for x, y, label, color in ((r_km / rmw_c, eta_c * 1e4, "noJET same time", "#3B6FB6"), (r_km / rmw_j, eta_j * 1e4, "JET", "#D44B3E"), (r_km / rmw_m, eta_m * 1e4, "noJET matched", "#009E73")):
        ax[0, 1].plot(x, y, lw=1.5, label=label, color=color)
    ax[0, 1].axhline(0, color="0.5", lw=0.7); ax[0, 1].set_xlim(0, 12); ax[0, 1].set_title(r"Absolute vorticity = $(1/r)\partial M/\partial r$"); ax[0, 1].set_xlabel("r/RMW"); ax[0, 1].set_ylabel(r"$f+\zeta$ ($10^{-4}$ s$^{-1}$)"); ax[0, 1].grid(alpha=0.2)
    names = list(DOMAINS)
    total_rms = [wrms(attribution["total"][k], weights, masks[n]) * 1e12 for n in names]
    med_rms = [wrms(attribution["mediated"][k], weights, masks[n]) * 1e12 for n in names]
    res_rms = [wrms(attribution["residual"][k], weights, masks[n]) * 1e12 for n in names]
    xx = np.arange(len(names)); w = 0.25
    ax[1, 0].bar(xx - w, total_rms, w, label="total", color="#4C78A8")
    ax[1, 0].bar(xx, med_rms, w, label="intensity/evolution", color="#F2B134")
    ax[1, 0].bar(xx + w, res_rms, w, label="matched residual", color="#59A14F")
    ax[1, 0].set_xticks(xx, [n.replace("_", "\n") for n in names]); ax[1, 0].set_ylabel(r"RMS $\Delta I^2$ ($10^{-12}$)"); ax[1, 0].set_title("72 h attribution magnitude by domain"); ax[1, 0].legend(frameon=False, fontsize=8); ax[1, 0].grid(axis="y", alpha=0.2)
    smooths = [0.0, 1.0, 2.0]
    for s, color in zip(smooths, ("#CC6677", "#4477AA", "#228833")):
        _, _, _, _, td = direct_tendencies(ctrl, jet, r_km, s)
        obs = torque["observed_tendency"]
        vals = [wcorr(td[k], obs[k], weights, masks[n]) for n in names]
        ax[1, 1].plot(xx, vals, "o-", label=f"radial smooth σ={s:g}", color=color)
    ax[1, 1].axhline(0, color="0.5", lw=0.7); ax[1, 1].set_xticks(xx, [n.replace("_", "\n") for n in names]); ax[1, 1].set_ylabel("Spatial correlation"); ax[1, 1].set_title("72 h eddy-tendency sensitivity"); ax[1, 1].legend(frameon=False, fontsize=8); ax[1, 1].grid(axis="y", alpha=0.2)
    for i, a0 in enumerate(ax.flat): add_letter(a0, "abcd"[i])
    fig.suptitle("Structural and robustness checks", fontsize=14, fontweight="bold")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)


def write_summary_csv(path, ctrl, jet, hours, matches, attribution, torque, weights, masks):
    fields = ["hour", "ctrl_match_hour", "pmin_ctrl", "pmin_jet", "vtmax_ctrl", "vtmax_jet", "rmw_ctrl", "rmw_jet"]
    for d in DOMAINS:
        fields += [f"{d}_I2_total_mean", f"{d}_I2_mediated_mean", f"{d}_I2_matched_residual_mean", f"{d}_unstable_ctrl", f"{d}_unstable_jet", f"{d}_eddy_direct_tendency_mean", f"{d}_observed_I2vort_tendency_mean"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for i, h in enumerate(hours):
            row = {"hour": h, "ctrl_match_hour": matches[i]["ctrl_hour"], "pmin_ctrl": ctrl[i]["pmin_hpa"], "pmin_jet": jet[i]["pmin_hpa"], "vtmax_ctrl": ctrl[i]["vtmax_m_s"], "vtmax_jet": jet[i]["vtmax_m_s"], "rmw_ctrl": ctrl[i]["rmw_km"], "rmw_jet": jet[i]["rmw_km"]}
            for d, mask in masks.items():
                row[f"{d}_I2_total_mean"] = wmean(attribution["total"][i], weights, mask)
                row[f"{d}_I2_mediated_mean"] = wmean(attribution["mediated"][i], weights, mask)
                row[f"{d}_I2_matched_residual_mean"] = wmean(attribution["residual"][i], weights, mask)
                row[f"{d}_unstable_ctrl"] = wfraction(ctrl[i]["I2"] <= 0, weights, mask)
                row[f"{d}_unstable_jet"] = wfraction(jet[i]["I2"] <= 0, weights, mask)
                row[f"{d}_eddy_direct_tendency_mean"] = wmean(torque["T_bui_vort"][i], weights, mask)
                row[f"{d}_observed_I2vort_tendency_mean"] = wmean(torque["observed_tendency"][i], weights, mask)
            writer.writerow(row)


def main():
    a = parse_args(); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    hours, regular_hours = selected_hours(a)
    pmin_c = read_pmin(a.nojet, hours); pmin_j = read_pmin(a.jet, hours)
    ctrl = analyze_case(a.nojet, hours, pmin_c, a, "noJET")
    jet = analyze_case(a.jet, hours, pmin_j, a, "JET")
    r_km, z_km = ctrl[0]["r_km"], ctrl[0]["z_km"]
    weights = cell_weights(r_km, z_km)
    masks = {name: domain_mask(r_km, z_km, bounds) for name, bounds in DOMAINS.items()}
    matches = strength_match(ctrl, jet, hours)
    figure_intensity(ctrl, jet, hours, matches, out / "figure1_intensity_matching.png")
    attribution = figure_stability(ctrl, jet, hours, matches, r_km, z_km, weights, masks, out / "figure2_stability_attribution.png")
    torque = figure_torque(ctrl, jet, hours, regular_hours, r_km, z_km, weights, masks, a.radial_smooth_sigma, out / "figure3_eddy_torque.png")
    figure_profiles(ctrl, jet, hours, matches, r_km, z_km, masks, weights, attribution, torque, out / "figure4_structural_robustness.png")
    write_summary_csv(out / "attribution_timeseries.csv", ctrl, jet, hours, matches, attribution, torque, weights, masks)
    np.savez_compressed(out / "attribution_products.npz", hours=hours, regular_hours=regular_hours,
        r_km=r_km, z_km=z_km, I2_ctrl=stack(ctrl, "I2"), I2_jet=stack(jet, "I2"),
        I2_total=attribution["total"], I2_mediated=attribution["mediated"], I2_matched_residual=attribution["residual"],
        classic_wind_component=attribution["wind_comp"], classic_vorticity_component=attribution["vort_comp"],
        F_lambda_env=torque["F_env"], eddy_direct_I2_tendency=torque["T_bui_vort"], observed_I2vort_tendency=torque["observed_tendency"])
    k72 = int(np.argmin(np.abs(hours - 72.0)))
    domain_summary = {}
    for name, mask in masks.items():
        domain_summary[name] = {
            "at_72h": {
                "total_I2_mean": wmean(attribution["total"][k72], weights, mask),
                "total_I2_rms": wrms(attribution["total"][k72], weights, mask),
                "intensity_mediated_I2_mean": wmean(attribution["mediated"][k72], weights, mask),
                "intensity_mediated_I2_rms": wrms(attribution["mediated"][k72], weights, mask),
                "matched_residual_I2_mean": wmean(attribution["residual"][k72], weights, mask),
                "matched_residual_I2_rms": wrms(attribution["residual"][k72], weights, mask),
                "unstable_fraction_ctrl": wfraction(ctrl[k72]["I2"] <= 0, weights, mask),
                "unstable_fraction_jet": wfraction(jet[k72]["I2"] <= 0, weights, mask),
                "classic_wind_component_mean_s-2": wmean(attribution["wind_comp"][k72], weights, mask),
                "classic_vorticity_component_mean_s-2": wmean(attribution["vort_comp"][k72], weights, mask),
                "eddy_direct_vs_observed_spatial_corr": wcorr(torque["T_bui_vort"][k72], torque["observed_tendency"][k72], weights, mask),
            },
            "lag_correlations": torque["lag"][name],
        }
    summary = {
        "analysis_type": "descriptive multi-angle attribution; not exclusive causal identification",
        "hours": hours.tolist(), "regular_hours": regular_hours.tolist(),
        "domains_km": {k: list(v) for k, v in DOMAINS.items()},
        "definitions": {
            "total": "I2_JET(t)-I2_CTRL(t)",
            "intensity_mediated": "I2_CTRL(t_match)-I2_CTRL(t)",
            "matched_residual": "I2_JET(t)-I2_CTRL(t_match)",
            "direct_eddy_eta_tendency": "(1/r)*d[r*F_lambda_env]/dr",
            "direct_eddy_classic_I_tendency": "eta_bar*(2F/r)+xi_bar*(1/r*d[rF]/dr)",
            "warning": "matching controls measured intensity/evolution only; eddy tendency omits thermodynamic and other momentum pathways",
        },
        "matches": matches, "domains": domain_summary,
        "outputs": ["figure1_intensity_matching.png", "figure2_stability_attribution.png", "figure3_eddy_torque.png", "figure4_structural_robustness.png", "attribution_timeseries.csv", "attribution_products.npz"],
    }
    (out / "attribution_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = f"""# Jet-related inertial-stability attribution\n\n## Scope\n\nThis is a descriptive multi-angle attribution using raw, unregularized Bui stability fields. It does not claim that two simulations alone identify an exclusive causal pathway.\n\n## Analysis design\n\n1. Same-time total difference.\n2. Matched-intensity CTRL counterfactual using Pmin, low-level azimuthal Vt maximum, and RMW.\n3. Exact classic-I wind-factor and absolute-vorticity decomposition.\n4. Environmental eddy-torque tendency and comparison with the observed tendency of the Bui vorticity component.\n5. Multiple spatial domains, lags, and radial-smoothing sensitivity.\n\n## Figures\n\n![Intensity matching](figure1_intensity_matching.png)\n\n![Stability attribution](figure2_stability_attribution.png)\n\n![Eddy torque](figure3_eddy_torque.png)\n\n![Robustness](figure4_structural_robustness.png)\n\n## Interpretation boundary\n\nThe matched residual is a strength-conditioned structural difference, not a pure direct-jet effect. A positive spatial/lag agreement between eddy torque and observed stability tendency supports dynamical consistency, not exclusive causation. Raw D and I2 determine SE applicability; regularized fields are excluded from this attribution.\n"""
    (out / "analysis_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "n_times": len(hours), "match_72h": matches[k72], "domains_72h": domain_summary}, indent=2))


if __name__ == "__main__":
    main()
