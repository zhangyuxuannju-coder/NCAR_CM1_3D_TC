#!/usr/bin/env python3
"""Solve the matched-intensity equivalent SE-operator response.

The JET-minus-CTRL operator perturbation is evaluated between independently
selected (intensity-matched) times.  Only RHS values at r >= mask_radius_km are
retained, while the regularized CTRL SE problem is solved over the full r-z
domain.  The solution is a balanced projection, not the full CM1 response.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src._se_pipeline_single import (
    PipelineConfig,
    _rho_ext_from_rho_zr,
    _to_solver_layout_zr_to_rz,
    azimuthal_average_from_3d,
    psi_to_uw,
    solve_se_sparse,
)
from src.se_bui import (
    assemble_operator, build_basic_state, invert_balanced_theta,
    regularize_ellipticity,
)


CMAP = LinearSegmentedColormap.from_list(
    "nature_div", ["#2F4B7C", "#7895B7", "#D6E0EA", "#F7F7F4",
                   "#F2C1AE", "#DD7055", "#9D2933"], N=256,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ctrl", required=True)
    p.add_argument("--jet", required=True)
    p.add_argument("--ctrl-hour", type=float, required=True)
    p.add_argument("--jet-hour", type=float, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-r-km", type=float, default=1200.0)
    p.add_argument("--dr-km", type=float, default=12.0)
    p.add_argument("--max-z-km", type=float, default=20.0)
    p.add_argument("--plot-max-z-km", type=float, default=18.0)
    p.add_argument("--mask-radius-km", type=float, default=100.0)
    p.add_argument("--f", type=float, default=5.464e-5)
    p.add_argument("--smooth-sigma-z", type=float, default=0.75)
    p.add_argument("--smooth-sigma-r", type=float, default=1.0)
    p.add_argument("--eps-ratio", type=float, default=1.0e-3)
    return p.parse_args()


def read_case(path, hour, a):
    cfg = PipelineConfig(
        input_file=path, output_dir=a.output_dir, target_time_hours=hour,
        max_r_km=a.max_r_km, dr_km=a.dr_km, max_z_km=a.max_z_km,
        coriolis_f=a.f, include_model_budget_terms=False,
        write_netcdf=False, write_ieee=False, plot_solution=False,
    )
    return azimuthal_average_from_3d(cfg)


def grad(field, coord, axis):
    return np.gradient(np.asarray(field, float), coord, axis=axis, edge_order=2)


def smooth(field, a):
    return gaussian_filter(
        np.asarray(field, float),
        sigma=(a.smooth_sigma_z, a.smooth_sigma_r), mode="nearest",
    )


def forcing_terms(d_a, d_b, d_i2, u0, w0, r_m, z_m):
    static = -grad(d_a * w0, r_m, axis=1)
    inertial = grad(d_i2 * u0, z_m, axis=0)
    baroclinic = -grad(d_b * u0, r_m, axis=1) + grad(d_b * w0, z_m, axis=0)
    return {
        "static": static,
        "inertial": inertial,
        "baroclinic": baroclinic,
        "total": static + inertial + baroclinic,
    }


def build_ctrl_operator(basic, r_m, z_m, a):
    k1r = np.asarray(basic["K1_raw"], float)
    k2r = np.asarray(basic["K2_raw"], float)
    k3r = np.asarray(basic["K3_raw"], float)
    raw_d = k1r * k3r - k2r ** 2
    k1, k2, k3, info = regularize_ellipticity(
        k1r, k2r, k3r, eps_ratio=a.eps_ratio, margin=0.0,
    )
    changed = (
        (np.abs(k1-k1r) > 1e-12 * max(np.nanmax(np.abs(k1r)), 1e-30)) |
        (np.abs(k2-k2r) > 1e-12 * max(np.nanmax(np.abs(k2r)), 1e-30)) |
        (np.abs(k3-k3r) > 1e-12 * max(np.nanmax(np.abs(k3r)), 1e-30))
    )
    return basic, assemble_operator(basic, k1, k2, k3, r_m, z_m), {
        **info,
        "raw_nonelliptic_fraction": float(np.mean(
            (~np.isfinite(raw_d)) | (k1r <= 0) | (k3r <= 0) | (raw_d <= 0)
        )),
        "changed_coefficient_fraction": float(np.mean(changed)),
        "min_regularized_discriminant": float(np.nanmin(k1*k3-k2**2)),
    }


def solve(operator, forcing_zr, rho_zr, r_m, z_m):
    arrays = {k: _to_solver_layout_zr_to_rz(operator[k]) for k in ("A", "B", "C", "D", "E")}
    dr = float(np.mean(np.diff(r_m)))
    dz = float(np.mean(np.diff(z_m)))
    psi = solve_se_sparse(
        A=arrays["A"], B=arrays["B"], C=arrays["C"],
        D=arrays["D"], E=arrays["E"],
        Fin=_to_solver_layout_zr_to_rz(forcing_zr), dr=dr, dz=dz,
    )
    rho_ext = _rho_ext_from_rho_zr(rho_zr)
    u, w = psi_to_uw(psi, rho_ext, r_m, dr, dz)
    return {
        "psi": psi[:, 1:-1].T,
        "u": u[:, 1:-1].T,
        "w": w[:, 1:-1].T,
    }


def robust_limit(fields, percentile=99.0):
    vals = np.concatenate([np.abs(np.asarray(f)[np.isfinite(f)]) for f in fields])
    return max(float(np.nanpercentile(vals, percentile)), 1e-30) if vals.size else 1.0


def exponent_scale(fields):
    lim = robust_limit(fields)
    exp = int(np.floor(np.log10(lim))) if lim > 0 else 0
    return 10.0 ** exp, lim / (10.0 ** exp), exp


def draw_field(ax, r, z, field, scale, lim, title, xlim):
    im = ax.pcolormesh(
        r, z, field / scale, shading="auto", cmap=CMAP,
        norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim), rasterized=True,
    )
    finite = field[np.isfinite(field)]
    if finite.size and np.nanmin(finite) < 0 < np.nanmax(finite):
        ax.contour(r, z, field, levels=[0], colors="0.25", linewidths=0.45)
    ax.set_xlim(*xlim)
    ax.set_title(title, fontsize=9.5)
    return im


def plot_forcing(r, z, forcing, a, out):
    keys = ("static", "inertial", "baroclinic", "total")
    maskz = z <= a.plot_max_z_km
    fields = [forcing[k][np.ix_(maskz, r >= a.mask_radius_km)] for k in keys]
    scale, lim, exp = exponent_scale(fields)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), sharex=True, sharey=True,
                             constrained_layout=True)
    titles = ["Static-stability term", "Inertial-stability term",
              "Baroclinic/shear term", "Total equivalent forcing"]
    ims = []
    for ax, key, title in zip(axes, keys, titles):
        rms = np.sqrt(np.nanmean(fields[keys.index(key)] ** 2))
        ims.append(draw_field(ax, r, z, forcing[key], scale, lim,
                              f"{title}\nouter-domain RMS={rms:.2e}",
                              (a.mask_radius_km, a.max_r_km)))
        ax.axvline(a.mask_radius_km, color="k", ls="--", lw=0.8)
        ax.set_xlabel("Radius (km)")
    axes[0].set_ylabel("Height (km)")
    axes[0].set_ylim(0, a.plot_max_z_km)
    cb = fig.colorbar(ims[-1], ax=axes, orientation="horizontal", shrink=0.72, pad=0.13)
    cb.set_label(rf"Equivalent SE forcing ($\times10^{{{exp}}}$ K$^{{-1}}$ s$^{{-3}}$)")
    fig.suptitle(
        f"Matched-state operator forcing: JET {a.jet_hour:g} h - CTRL {a.ctrl_hour:g} h\n"
        f"RHS retained only at r ≥ {a.mask_radius_km:g} km", fontweight="bold",
    )
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_responses(r, z, responses, a, out):
    keys = ("static", "inertial", "baroclinic", "total")
    row_vars = ("psi", "u", "w")
    row_labels = ("Mass streamfunction", "Radial wind (outward +)", "Vertical velocity (upward +)")
    units = ("kg s$^{-1}$", "m s$^{-1}$", "m s$^{-1}$")
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), sharex=True, sharey=True,
                             constrained_layout=True)
    zmask = z <= a.plot_max_z_km
    for row, (var, row_label, unit) in enumerate(zip(row_vars, row_labels, units)):
        fields = [responses[k][var][zmask] for k in keys]
        scale, lim, exp = exponent_scale(fields)
        ims = []
        for col, key in enumerate(keys):
            fld = responses[key][var]
            rms = np.sqrt(np.nanmean(fld[zmask] ** 2))
            ims.append(draw_field(
                axes[row, col], r, z, fld, scale, lim,
                f"{key.capitalize()} response\nRMS={rms:.2e}", (0, a.max_r_km),
            ))
            axes[row, col].axvline(a.mask_radius_km, color="k", ls="--", lw=0.65)
        cb = fig.colorbar(ims[-1], ax=axes[row, :], orientation="vertical", shrink=0.78, pad=0.015)
        cb.set_label(rf"{row_label} ($\times10^{{{exp}}}$ {unit})")
    for ax in axes[-1]: ax.set_xlabel("Radius (km)")
    for ax in axes[:, 0]: ax.set_ylabel("Height (km)")
    axes[0, 0].set_ylim(0, a.plot_max_z_km)
    fig.suptitle(
        "Full-domain regularized SE response to outer-only equivalent operator forcing\n"
        "Dashed line marks the 100-km forcing cutoff; response is solved at all radii",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)


def weighted_metrics(pred, actual, r, z, bounds):
    r0, r1, z0, z1 = bounds
    mask = ((r[None, :] >= r0) & (r[None, :] <= r1) &
            (z[:, None] >= z0) & (z[:, None] <= z1) &
            np.isfinite(pred) & np.isfinite(actual))
    weight = np.maximum(r[None, :], 0.5)
    if not np.any(mask):
        return {"pred_rms": np.nan, "actual_rms": np.nan, "correlation": np.nan,
                "projection_on_actual": np.nan, "mean_pred": np.nan}
    w = np.broadcast_to(weight, pred.shape)[mask]
    p, q = pred[mask], actual[mask]
    pred_rms = np.sqrt(np.sum(w*p*p)/np.sum(w))
    actual_rms = np.sqrt(np.sum(w*q*q)/np.sum(w))
    pm, qm = np.sum(w*p)/np.sum(w), np.sum(w*q)/np.sum(w)
    cov = np.sum(w*(p-pm)*(q-qm))
    den = np.sqrt(np.sum(w*(p-pm)**2)*np.sum(w*(q-qm)**2))
    return {
        "pred_rms": float(pred_rms), "actual_rms": float(actual_rms),
        "correlation": float(cov/den) if den > 0 else np.nan,
        "projection_on_actual": float(np.sum(w*p*q)/np.sum(w*q*q)) if np.sum(w*q*q) > 0 else np.nan,
        "mean_pred": float(pm),
    }


def plot_benchmark(r, z, predicted, actual_u, actual_w, metrics, a, out):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True,
                             constrained_layout=True)
    rows = [("u", predicted["u"], actual_u, "Radial wind", "m s$^{-1}$"),
            ("w", predicted["w"], actual_w, "Vertical velocity", "m s$^{-1}$")]
    for row, (key, pred, actual, title, unit) in enumerate(rows):
        residual = actual - pred
        scale, lim, exp = exponent_scale([pred, actual, residual])
        for col, (fld, subtitle) in enumerate(((pred, "Operator-only SE projection"),
                                                (actual, "Actual matched CM1 difference"),
                                                (residual, "CM1 difference minus projection"))):
            draw_field(axes[row, col], r, z, fld, scale, lim, subtitle, (0, a.max_r_km))
            axes[row, col].axvline(a.mask_radius_km, color="k", ls="--", lw=0.65)
        cb = fig.colorbar(axes[row, 2].collections[0], ax=axes[row, :], orientation="vertical",
                          shrink=0.78, pad=0.015)
        cb.set_label(rf"{title} ($\times10^{{{exp}}}$ {unit})")
        core = metrics[key]["inner_0_100km"]
        axes[row, 0].text(
            0.02, 0.03,
            f"r<100 km: corr={core['correlation']:.2f}, projection={core['projection_on_actual']:.2f}",
            transform=axes[row, 0].transAxes, fontsize=8,
            bbox=dict(facecolor="white", alpha=0.78, edgecolor="0.6"),
        )
    for ax in axes[-1]: ax.set_xlabel("Radius (km)")
    for ax in axes[:, 0]: ax.set_ylabel("Height (km)")
    axes[0, 0].set_ylim(0, a.plot_max_z_km)
    fig.suptitle(
        "Outer-forced balanced projection versus the full matched CM1 difference\n"
        "The CM1 difference contains every pathway and is only a descriptive benchmark",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main():
    a = parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Reading CTRL {a.ctrl_hour:g} h", flush=True)
    ctrl = read_case(a.ctrl, a.ctrl_hour, a)
    print(f"Reading JET {a.jet_hour:g} h", flush=True)
    jet = read_case(a.jet, a.jet_hour, a)
    r = np.asarray(ctrl["r_km"], float)
    z = np.asarray(ctrl["z_km"], float)
    if not np.allclose(r, jet["r_km"]) or not np.allclose(z, jet["z_km"]):
        raise ValueError("CTRL and JET SE grids differ")
    r_m, z_m = r*1000.0, z*1000.0

    # A coherent SE inversion requires a balanced basic state. Apply the same
    # thermal-wind projection to both matched states before forming delta-L.
    theta_c_bal, tw_c = invert_balanced_theta(
        ctrl["ut"], ctrl["theta"], r_m, z_m, a.f, outer_smooth_window=1,
    )
    theta_j_bal, tw_j = invert_balanced_theta(
        jet["ut"], jet["theta"], r_m, z_m, a.f, outer_smooth_window=1,
    )
    bc = build_basic_state(ctrl["ut"], theta_c_bal, ctrl["rho"], r_m, z_m, a.f)
    bj = build_basic_state(jet["ut"], theta_j_bal, jet["rho"], r_m, z_m, a.f)
    # Compact derivation: A=K1, B=-K2, I2=K3.
    d_a = smooth(np.asarray(bj["K1_raw"])-np.asarray(bc["K1_raw"]), a)
    d_b = smooth(-np.asarray(bj["K2_raw"])+np.asarray(bc["K2_raw"]), a)
    d_i2 = smooth(np.asarray(bj["K3_raw"])-np.asarray(bc["K3_raw"]), a)
    u0, w0 = smooth(ctrl["ur"], a), smooth(ctrl["w"], a)
    forcing_raw = forcing_terms(d_a, d_b, d_i2, u0, w0, r_m, z_m)
    outer_mask = r[None, :] >= a.mask_radius_km
    forcing = {k: np.where(outer_mask, v, 0.0) for k, v in forcing_raw.items()}

    basic_ctrl, operator, reg_info = build_ctrl_operator(bc, r_m, z_m, a)
    responses = {}
    for key in ("static", "inertial", "baroclinic", "total"):
        print(f"Solving {key} response", flush=True)
        responses[key] = solve(operator, forcing[key], basic_ctrl["rho"], r_m, z_m)

    # Linearity closure of independently solved components.
    closure = {}
    for var in ("psi", "u", "w"):
        summed = responses["static"][var] + responses["inertial"][var] + responses["baroclinic"][var]
        closure[var] = float(np.nanmax(np.abs(summed-responses["total"][var])))

    actual_u = np.asarray(jet["ur"], float) - np.asarray(ctrl["ur"], float)
    actual_w = np.asarray(jet["w"], float) - np.asarray(ctrl["w"], float)
    domains = {
        "inner_0_100km": (0.0, 100.0, 0.0, a.plot_max_z_km),
        "inner_lowlevel": (0.0, 100.0, 0.0, 2.0),
        "inner_outflow": (0.0, 300.0, 10.0, 18.0),
        "outer_forced": (100.0, a.max_r_km, 0.0, a.plot_max_z_km),
        "jet_annulus": (650.0, min(1150.0, a.max_r_km), 10.0, 16.0),
    }
    comparison = {"u": {}, "w": {}}
    for name, bounds in domains.items():
        comparison["u"][name] = weighted_metrics(responses["total"]["u"], actual_u, r, z, bounds)
        comparison["w"][name] = weighted_metrics(responses["total"]["w"], actual_w, r, z, bounds)

    forcing_rms = {}
    omask = np.broadcast_to(outer_mask, forcing["total"].shape) & (z[:, None] <= a.plot_max_z_km)
    for key in ("static", "inertial", "baroclinic", "total"):
        forcing_rms[key] = float(np.sqrt(np.nanmean(forcing[key][omask]**2)))

    plot_forcing(r, z, forcing, a, out/"figure1_outer100_operator_forcing.png")
    plot_responses(r, z, responses, a, out/"figure2_fullradius_component_responses.png")
    plot_benchmark(r, z, responses["total"], actual_u, actual_w, comparison, a,
                   out/"figure3_projection_vs_matched_cm1.png")

    np.savez_compressed(
        out/"matched_operator_outer100_products.npz", r_km=r, z_km=z,
        delta_A=d_a, delta_B=d_b, delta_I2=d_i2,
        **{f"S_{k}": v for k, v in forcing.items()},
        **{f"psi_{k}": v["psi"] for k, v in responses.items()},
        **{f"u_{k}": v["u"] for k, v in responses.items()},
        **{f"w_{k}": v["w"] for k, v in responses.items()},
        actual_matched_du=actual_u, actual_matched_dw=actual_w,
    )
    summary = {
        "inputs": {"ctrl": a.ctrl, "jet": a.jet},
        "matched_times_h": {"ctrl": a.ctrl_hour, "jet": a.jet_hour},
        "matching_metadata_from_existing_attribution": {
            "pmin_hpa": {"ctrl": 977.6663375370576, "jet": 978.0754008328647},
            "lowlevel_vtmax_m_s": {"ctrl": 37.873664853107464, "jet": 38.61687191133356},
            "rmw_km": {"ctrl": 30.0, "jet": 18.0},
        },
        "definition": "L_CTRL70(delta_psi) = -deltaL(JET75-CTRL70) psi_CTRL70",
        "forcing_mask": f"hard zero for r < {a.mask_radius_km:g} km; full SE domain retained",
        "operator": "thermal-wind-balanced CTRL operator, ellipticity-regularized, homogeneous Dirichlet streamfunction boundary",
        "coefficient_difference": "JET75 balanced projection minus CTRL70 balanced projection",
        "thermal_wind_projection": {"ctrl": tw_c, "jet": tw_j},
        "interpretation": "regularized balanced projection of the leading matched-state operator perturbation only",
        "forcing_outer_rms_K-1_s-3": forcing_rms,
        "regularization": reg_info,
        "linearity_closure_max_abs": closure,
        "comparison_with_actual_matched_cm1_difference": comparison,
        "warning": "matching conditions on measured intensity/evolution only; actual CM1 difference includes all pathways and the RMW mismatch is 12 km",
    }
    (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
