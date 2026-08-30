#!/usr/bin/env python3
"""Diagnose the equivalent SE forcing caused by JET-induced operator changes.

No SE inversion is performed.  The diagnostic follows the first-order identity

    L_CTRL(delta psi) = -delta L psi_CTRL

using the actual azimuthal-mean CTRL secondary circulation from CM1.  Bui's
mass-weighted streamfunction convention is retained:

    u = -(rho r)^-1 psi_z,    w = (rho r)^-1 psi_r.

The project's raw K2 coefficient is minus the B used in the compact derivation,
so this script explicitly sets B = -K2 before evaluating the forcing terms.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src._se_pipeline_single import PipelineConfig, azimuthal_average_from_3d
from src.se_applicability import compute_case_stability


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--nojet", required=True)
    p.add_argument("--jet", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--time-hours", type=float, default=0.0)
    p.add_argument("--max-r-km", type=float, default=1200.0)
    p.add_argument("--dr-km", type=float, default=12.0)
    p.add_argument("--max-z-km", type=float, default=20.0)
    p.add_argument("--f", type=float, default=6.2e-5)
    p.add_argument("--smooth-sigma-z", type=float, default=0.75)
    p.add_argument("--smooth-sigma-r", type=float, default=1.0)
    p.add_argument("--plot-max-z-km", type=float, default=18.0)
    p.add_argument("--jet-axis-r-km", type=float, default=888.0)
    p.add_argument("--jet-axis-z-km", type=float, default=12.0)
    return p.parse_args()


def grad(a: np.ndarray, coord: np.ndarray, axis: int) -> np.ndarray:
    return np.gradient(np.asarray(a, dtype=float), coord, axis=axis, edge_order=2)


def smooth(a: np.ndarray, sigz: float, sigr: float) -> np.ndarray:
    if sigz <= 0.0 and sigr <= 0.0:
        return np.asarray(a, dtype=float)
    return gaussian_filter(np.asarray(a, dtype=float), sigma=(sigz, sigr), mode="nearest")


def forcing_terms(
    d_a: np.ndarray,
    d_b: np.ndarray,
    d_i2: np.ndarray,
    u0: np.ndarray,
    w0: np.ndarray,
    r_m: np.ndarray,
    z_m: np.ndarray,
) -> dict[str, np.ndarray]:
    """Equation (1), with B=+d(chi*Cg)/dz (not the code's raw K2)."""
    static = -grad(d_a * w0, r_m, axis=1)
    inertial = grad(d_i2 * u0, z_m, axis=0)
    baroclinic = -grad(d_b * u0, r_m, axis=1) + grad(d_b * w0, z_m, axis=0)
    return {
        "static": static,
        "inertial": inertial,
        "baroclinic": baroclinic,
        "total": static + inertial + baroclinic,
    }


def cell_weights(r_m: np.ndarray, z_m: np.ndarray) -> np.ndarray:
    dr = np.abs(np.gradient(r_m))
    dz = np.abs(np.gradient(z_m))
    return np.maximum(r_m, 1.0)[None, :] * dr[None, :] * dz[:, None]


def domain_mask(r_km: np.ndarray, z_km: np.ndarray, bounds: tuple[float, ...]) -> np.ndarray:
    r0, r1, z0, z1 = bounds
    return (
        (r_km[None, :] >= r0) & (r_km[None, :] <= r1)
        & (z_km[:, None] >= z0) & (z_km[:, None] <= z1)
    )


def wrms(a: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    good = mask & np.isfinite(a) & np.isfinite(weights)
    den = float(np.sum(weights[good]))
    if den <= 0.0:
        return float("nan")
    return float(np.sqrt(np.sum(weights[good] * a[good] ** 2) / den))


def wcorr(a: np.ndarray, b: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    good = mask & np.isfinite(a) & np.isfinite(b) & np.isfinite(weights)
    if np.count_nonzero(good) < 4:
        return float("nan")
    w = weights[good]
    w = w / np.sum(w)
    x, y = a[good], b[good]
    xm, ym = np.sum(w * x), np.sum(w * y)
    num = np.sum(w * (x - xm) * (y - ym))
    den = np.sqrt(np.sum(w * (x - xm) ** 2) * np.sum(w * (y - ym) ** 2))
    return float(num / den) if den > 0.0 else float("nan")


def ratio(num: float, den: float) -> float:
    return float(num / den) if np.isfinite(den) and den > 1.0e-30 else float("nan")


def read_case(path: str, a: argparse.Namespace) -> dict[str, np.ndarray]:
    cfg = PipelineConfig(
        input_file=path,
        output_dir=a.output_dir,
        target_time_hours=a.time_hours,
        max_r_km=a.max_r_km,
        dr_km=a.dr_km,
        max_z_km=a.max_z_km,
        coriolis_f=a.f,
        include_model_budget_terms=False,
        write_netcdf=False,
        write_ieee=False,
        plot_solution=False,
    )
    return azimuthal_average_from_3d(cfg)


def stability(avg: dict[str, np.ndarray], r_m: np.ndarray, z_m: np.ndarray, f: float):
    fields, _ = compute_case_stability(
        avg["ut"], avg["theta"], avg["rho"], r_m, z_m, f,
        outer_smooth_window=1,
    )
    return fields


def scale_for_fields(fields: list[np.ndarray], mask: np.ndarray) -> tuple[float, float]:
    vals = np.concatenate([np.abs(q[mask & np.isfinite(q)]) for q in fields])
    vmax = float(np.nanpercentile(vals, 99.0)) if vals.size else 1.0
    vmax = max(vmax, 1.0e-30)
    scale = 10.0 ** np.floor(np.log10(vmax))
    return scale, vmax / scale


def div_cmap():
    return LinearSegmentedColormap.from_list(
        "operator_force",
        ["#24486E", "#6F98BE", "#C8D9E7", "#F7F7F4", "#F4C6AF", "#D96B4C", "#8E2C3A"],
        N=256,
    )


def panel(ax, r, z, field, scale, lim, title, letter, marker=True):
    plot = field / scale
    p = ax.pcolormesh(
        r, z, plot, cmap=div_cmap(),
        norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
        shading="auto", rasterized=True,
    )
    finite = field[np.isfinite(field)]
    if finite.size and np.nanmin(finite) <= 0.0 <= np.nanmax(finite):
        ax.contour(r, z, field, levels=[0.0], colors="0.25", linewidths=0.45)
    if marker:
        ax.plot(888.0, 12.0, marker="*", ms=9, color="#7A1FA2", mec="white", mew=0.5)
    rms = np.sqrt(np.nanmean(field**2))
    ax.set_title(f"{title}\nRMS={rms:.2e}", fontsize=9.3)
    ax.text(-0.11, 1.03, f"({letter})", transform=ax.transAxes, fontweight="bold")
    return p


def plot_forcing(r, z, terms, density_corr, second, output, plot_zmax):
    zmask = z[:, None] <= plot_zmax
    mask = np.broadcast_to(zmask, terms["total"].shape) & (r[None, :] >= 30.0)
    common = [terms[k] for k in ("static", "inertial", "baroclinic", "total")]
    s1, l1 = scale_for_fields(common, mask)
    s2, l2 = scale_for_fields([density_corr, second], mask)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.3), sharex=True, sharey=True,
                             constrained_layout=True)
    fields = common + [density_corr, second]
    titles = [
        r"$S_{static}=-\partial_r(\delta A\,w_0)$",
        r"$S_{inertial}=\partial_z(\delta I^2\,u_0)$",
        r"$S_{baroclinic}=-\partial_r(\delta B\,u_0)+\partial_z(\delta B\,w_0)$",
        r"$S_{jet}^{stab}$ (sum)",
        r"Neglected density-metric correction",
        r"Neglected cross term proxy $-\delta L\,\delta\psi_{CM1}$",
    ]
    mappables = []
    for i, (ax, fld, title) in enumerate(zip(axes.flat, fields, titles)):
        scale, lim = (s1, l1) if i < 4 else (s2, l2)
        mappables.append(panel(ax, r, z, fld, scale, lim, title, "abcdef"[i]))
        ax.set_xlim(0.0, r[-1]); ax.set_ylim(0.0, plot_zmax)
    for ax in axes[-1]: ax.set_xlabel("Radius from TC centre (km)")
    for ax in axes[:, 0]: ax.set_ylabel("Height (km)")
    cb1 = fig.colorbar(mappables[0], ax=axes[0, :], orientation="horizontal", shrink=0.72, pad=0.03)
    cb1.set_label(rf"Forcing ($\times 10^{{{int(np.log10(s1))}}}$ K$^{{-1}}$ s$^{{-3}}$)")
    cb2 = fig.colorbar(mappables[-1], ax=axes[1, :], orientation="horizontal", shrink=0.72, pad=0.03)
    cb2.set_label(rf"Correction ($\times 10^{{{int(np.log10(s2))}}}$ K$^{{-1}}$ s$^{{-3}}$)")
    fig.suptitle("Equivalent SE forcing from JET-induced operator perturbations", fontsize=14, fontweight="bold")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_i2(r, z, d_i2, d_i2_lin, fi, fi_lin, output, plot_zmax):
    ri = d_i2_lin - d_i2
    rf = fi_lin - fi
    zmask = np.broadcast_to(z[:, None] <= plot_zmax, d_i2.shape) & (r[None, :] >= 30.0)
    s1, l1 = scale_for_fields([d_i2, d_i2_lin, ri], zmask)
    s2, l2 = scale_for_fields([fi, fi_lin, rf], zmask)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.3), sharex=True, sharey=True,
                             constrained_layout=True)
    fields = [d_i2, d_i2_lin, ri, fi, fi_lin, rf]
    titles = [
        r"Exact $\delta I^2=I^2_J-I^2_C$",
        r"First-order state expansion of $\delta I^2$",
        r"Linearization residual",
        r"$\partial_z(u_0\,\delta I^2)$",
        r"$\partial_z(u_0\,\delta I^2_{linear})$",
        r"Inertial-forcing linearization residual",
    ]
    maps = []
    for i, (ax, fld, title) in enumerate(zip(axes.flat, fields, titles)):
        scale, lim = (s1, l1) if i < 3 else (s2, l2)
        maps.append(panel(ax, r, z, fld, scale, lim, title, "abcdef"[i]))
        ax.set_xlim(0.0, r[-1]); ax.set_ylim(0.0, plot_zmax)
    for ax in axes[-1]: ax.set_xlabel("Radius from TC centre (km)")
    for ax in axes[:, 0]: ax.set_ylabel("Height (km)")
    cb1 = fig.colorbar(maps[0], ax=axes[0, :], orientation="horizontal", shrink=0.72, pad=0.03)
    cb1.set_label(rf"$\delta I^2$ ($\times 10^{{{int(np.log10(s1))}}}$ K$^{{-1}}$ s$^{{-2}}$)")
    cb2 = fig.colorbar(maps[-1], ax=axes[1, :], orientation="horizontal", shrink=0.72, pad=0.03)
    cb2.set_label(rf"Forcing ($\times 10^{{{int(np.log10(s2))}}}$ K$^{{-1}}$ s$^{{-3}}$)")
    fig.suptitle("Exact versus first-order generalized-inertial-stability perturbation", fontsize=14, fontweight="bold")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scales(metrics: dict, output: Path):
    names = list(metrics["domains"])
    terms = ["static_rms", "inertial_rms", "baroclinic_rms", "total_rms"]
    labels = ["static", "inertial", "baroclinic", "total"]
    colors = ["#4C78A8", "#E45756", "#72B7B2", "#333333"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    x = np.arange(len(names)); width = 0.18
    for j, (term, label, color) in enumerate(zip(terms, labels, colors)):
        vals = [metrics["domains"][n][term] for n in names]
        axes[0].bar(x + (j - 1.5) * width, vals, width, label=label, color=color)
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel(r"Cylindrical RMS (K$^{-1}$ s$^{-3}$)")
    axes[0].set_title("Equivalent-forcing component scales")
    axes[0].legend(frameon=False)
    approx = ["density_correction_ratio", "second_order_proxy_ratio", "i2_linear_residual_ratio", "smoothing_change_ratio"]
    alabels = [r"$\delta q$ correction", "cross-term proxy", r"$\delta I^2$ nonlinear residual", "smoothing sensitivity"]
    for j, (key, label, color) in enumerate(zip(approx, alabels, colors)):
        vals = [metrics["domains"][n][key] for n in names]
        axes[1].plot(x, vals, "o-", lw=1.6, ms=5, label=label, color=color)
    axes[1].axhline(0.1, color="0.55", ls=":", lw=1)
    axes[1].axhline(0.3, color="0.45", ls="--", lw=1)
    axes[1].set_xticks(x, names)
    axes[1].set_ylim(bottom=0.0)
    axes[1].set_ylabel("RMS ratio to retained leading term")
    axes[1].set_title("Approximation checks (smaller is better)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    a = parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ctrl = read_case(a.nojet, a)
    jet = read_case(a.jet, a)
    tc = float(ctrl["time_seconds_used"][0] / 3600.0)
    tj = float(jet["time_seconds_used"][0] / 3600.0)
    if abs(tc - tj) > 1.0e-6:
        raise ValueError(f"Selected times differ: CTRL={tc} h, JET={tj} h")

    r_km = np.asarray(ctrl["r_km"], float)
    z_km = np.asarray(ctrl["z_km"], float)
    r_m, z_m = r_km * 1000.0, z_km * 1000.0
    sc = stability(ctrl, r_m, z_m, a.f)
    sj = stability(jet, r_m, z_m, a.f)

    # Compact derivation coefficients: A=K1, B=-K2, I2=K3.
    ac = np.asarray(sc["static_stability_raw"], float)
    aj = np.asarray(sj["static_stability_raw"], float)
    bc = -np.asarray(sc["shear_coefficient_raw"], float)
    bj = -np.asarray(sj["shear_coefficient_raw"], float)
    ic = np.asarray(sc["I2_raw"], float)
    ij = np.asarray(sj["I2_raw"], float)
    d_a, d_b, d_i2 = aj - ac, bj - bc, ij - ic
    u0, w0 = np.asarray(ctrl["ur"], float), np.asarray(ctrl["w"], float)
    du = np.asarray(jet["ur"], float) - u0
    dw = np.asarray(jet["w"], float) - w0

    sigz, sigr = a.smooth_sigma_z, a.smooth_sigma_r
    arrays_to_smooth = [d_a, d_b, d_i2, u0, w0, du, dw]
    d_as, d_bs, d_is, u0s, w0s, dus, dws = [smooth(q, sigz, sigr) for q in arrays_to_smooth]
    raw_terms = forcing_terms(d_a, d_b, d_i2, u0, w0, r_m, z_m)
    terms = forcing_terms(d_as, d_bs, d_is, u0s, w0s, r_m, z_m)

    # Finite density-metric correction. Since q=(rho*r)^-1 and r is common,
    # q_J/q_C=rho_C/rho_J.  This is the exact coefficient change acting on
    # psi_CTRL, not merely a differential estimate of delta q.
    rho_ratio = np.asarray(ctrl["rho"], float) / np.maximum(np.asarray(jet["rho"], float), 1.0e-10)
    d_a_eff = rho_ratio * aj - ac
    d_b_eff = rho_ratio * bj - bc
    d_i_eff = rho_ratio * ij - ic
    d_a_eff, d_b_eff, d_i_eff = [smooth(q, sigz, sigr) for q in (d_a_eff, d_b_eff, d_i_eff)]
    metric_terms = forcing_terms(d_a_eff, d_b_eff, d_i_eff, u0s, w0s, r_m, z_m)
    density_corr = metric_terms["total"] - terms["total"]

    # Observable proxy for the formally neglected delta-L delta-psi term,
    # using the CM1 JET-CTRL secondary-circulation difference.
    second = forcing_terms(d_as, d_bs, d_is, dus, dws, r_m, z_m)["total"]

    # First-order state expansion of generalized I2 (attachment Eq. 3/4).
    chic = np.asarray(sc["chi_raw"], float)
    chij = np.asarray(sj["chi_raw"], float)
    xic = np.asarray(sc["xi_raw"], float)
    xij = np.asarray(sj["xi_raw"], float)
    etac = np.asarray(sc["zeta_abs_raw"], float)
    etaj = np.asarray(sj["zeta_abs_raw"], float)
    cc = np.asarray(sc["Cg_raw"], float)
    dv = np.asarray(jet["ut"], float) - np.asarray(ctrl["ut"], float)
    dchi, dxi, deta = chij - chic, xij - xic, etaj - etac
    chi_r = grad(chic, r_m, axis=1)
    d_c_linear = xic * dv
    d_i2_linear = (
        xic * etac * dchi
        + chic * etac * dxi
        + chic * xic * deta
        + chi_r * d_c_linear
        + cc * grad(dchi, r_m, axis=1)
    )
    d_i2_linear_s = smooth(d_i2_linear, sigz, sigr)
    inertial_linear = forcing_terms(
        np.zeros_like(d_as), np.zeros_like(d_bs), d_i2_linear_s,
        u0s, w0s, r_m, z_m,
    )["inertial"]

    weights = cell_weights(r_m, z_m)
    domains = {
        "full": (30.0, a.max_r_km, 0.5, min(18.0, a.max_z_km)),
        "inner_outflow": (50.0, 350.0, 10.0, 16.0),
        "jet_annulus": (650.0, min(1150.0, a.max_r_km), 10.0, 16.0),
    }
    metrics = {
        "inputs": {"nojet": a.nojet, "jet": a.jet},
        "requested_time_hours": a.time_hours,
        "selected_time_hours": tc,
        "f_s-1": a.f,
        "grid": {"max_r_km": a.max_r_km, "dr_km": float(ctrl["dr_km_used"][0]), "max_z_km": a.max_z_km},
        "smoothing_sigma_gridpoints": {"z": sigz, "r": sigr},
        "definitions": {
            "S_static": "-d_r(deltaA*w_CTRL)",
            "S_inertial": "d_z(deltaI2*u_CTRL)",
            "S_baroclinic": "-d_r(deltaB*u_CTRL)+d_z(deltaB*w_CTRL)",
            "B_sign": "B=+d_z(chi*Cg)=-project_K2_raw",
            "density_correction": "q_J/q_C=rho_C/rho_J retained exactly while acting on psi_CTRL",
            "second_order_proxy": "-deltaL*deltaPsi using observed CM1 delta-u and delta-w",
        },
        "ctrl_secondary_circulation": {
            "ur_rms_m_s": float(np.sqrt(np.nanmean(u0**2))),
            "w_rms_m_s": float(np.sqrt(np.nanmean(w0**2))),
            "ur_max_abs_m_s": float(np.nanmax(np.abs(u0))),
            "w_max_abs_m_s": float(np.nanmax(np.abs(w0))),
        },
        "domains": {},
    }
    for name, bounds in domains.items():
        mask = domain_mask(r_km, z_km, bounds)
        total_rms = wrms(terms["total"], weights, mask)
        exact_i_rms = wrms(d_is, weights, mask)
        metrics["domains"][name] = {
            "bounds_r0_r1_z0_z1_km": list(bounds),
            "static_rms": wrms(terms["static"], weights, mask),
            "inertial_rms": wrms(terms["inertial"], weights, mask),
            "baroclinic_rms": wrms(terms["baroclinic"], weights, mask),
            "total_rms": total_rms,
            "component_cancellation_factor": ratio(
                sum(wrms(terms[k], weights, mask) for k in ("static", "inertial", "baroclinic")), total_rms
            ),
            "density_correction_ratio": ratio(wrms(density_corr, weights, mask), total_rms),
            "second_order_proxy_ratio": ratio(wrms(second, weights, mask), total_rms),
            "smoothing_change_ratio": ratio(wrms(terms["total"] - raw_terms["total"], weights, mask), total_rms),
            "i2_linear_residual_ratio": ratio(wrms(d_i2_linear_s - d_is, weights, mask), exact_i_rms),
            "i2_linear_exact_correlation": wcorr(d_i2_linear_s, d_is, weights, mask),
            "inertial_forcing_linear_residual_ratio": ratio(
                wrms(inertial_linear - terms["inertial"], weights, mask),
                wrms(terms["inertial"], weights, mask),
            ),
            "deltaA_relative_rms": ratio(wrms(d_as, weights, mask), wrms(ac, weights, mask)),
            "deltaB_relative_rms": ratio(wrms(d_bs, weights, mask), wrms(bc, weights, mask)),
            "deltaI2_relative_rms": ratio(exact_i_rms, wrms(ic, weights, mask)),
            "rho_metric_relative_rms": ratio(wrms(rho_ratio - 1.0, weights, mask), 1.0),
        }

    plot_forcing(r_km, z_km, terms, density_corr, second,
                 out / "operator_perturbation_forcing.png", a.plot_max_z_km)
    plot_i2(r_km, z_km, d_is, d_i2_linear_s, terms["inertial"], inertial_linear,
            out / "delta_I2_linearization.png", a.plot_max_z_km)
    plot_scales(metrics, out / "scale_analysis.png")

    np.savez_compressed(
        out / "operator_perturbation_products.npz",
        r_km=r_km, z_km=z_km,
        delta_A=d_as, delta_B=d_bs, delta_I2=d_is,
        delta_I2_linear=d_i2_linear_s,
        S_static=terms["static"], S_inertial=terms["inertial"],
        S_baroclinic=terms["baroclinic"], S_total=terms["total"],
        S_density_correction=density_corr,
        S_second_order_proxy=second,
        S_inertial_linearized=inertial_linear,
        ur_ctrl=u0, w_ctrl=w0, ur_change=du, w_change=dw,
    )
    (out / "scale_analysis.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
