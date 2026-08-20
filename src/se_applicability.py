"""Diagnose where the classical Bui Sawyer--Eliassen problem is applicable.

All primary stability fields in this module are computed from the unmodified
basic state.  Regularized coefficients are written only as explicitly named
comparison fields; they never replace ``I2_raw`` or ``D_raw``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np

from .environmental_eddy import environmental_difference
from .se_bui import build_basic_state, invert_balanced_theta, regularize_ellipticity


def classify_stability(
    static_stability: np.ndarray,
    inertial_stability: np.ndarray,
    discriminant: np.ndarray,
) -> np.ndarray:
    """Return mutually exclusive physical applicability classes.

    Codes are 0=elliptic, 1=inertially unstable, 2=shear/symmetric
    non-elliptic, and 3=statically unstable.  Static instability has highest
    priority so every grid point receives exactly one label.
    """
    k1 = np.asarray(static_stability, dtype=np.float64)
    i2 = np.asarray(inertial_stability, dtype=np.float64)
    disc = np.asarray(discriminant, dtype=np.float64)
    if not (k1.shape == i2.shape == disc.shape):
        raise ValueError("K1, I2 and D must have matching shapes")

    classes = np.full(k1.shape, 3, dtype=np.int8)
    statically_stable = np.isfinite(k1) & (k1 > 0.0)
    inertially_unstable = statically_stable & np.isfinite(i2) & (i2 <= 0.0)
    symmetric_nonelliptic = (
        statically_stable & (i2 > 0.0) & np.isfinite(disc) & (disc <= 0.0)
    )
    elliptic = statically_stable & (i2 > 0.0) & (disc > 0.0)
    classes[inertially_unstable] = 1
    classes[symmetric_nonelliptic] = 2
    classes[elliptic] = 0
    return classes


def compute_case_stability(
    vt_zr: np.ndarray,
    theta_model_zr: np.ndarray,
    rho_zr: np.ndarray,
    r_m: np.ndarray,
    z_m: np.ndarray,
    coriolis_f: float,
    theta_floor: float = 150.0,
    outer_smooth_window: int = 1,
    regularization_eps_ratio: float = 1.0e-3,
    regularization_margin: float = 0.0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Compute raw and explicitly labelled regularized Bui stability fields."""
    # Primary diagnosis: the actual azimuthal-mean CM1 basic state.  No
    # thermal-wind inversion and no ellipticity regularization are applied.
    basic_raw = build_basic_state(
        vt_zr,
        theta_model_zr,
        rho_zr,
        r_m,
        z_m,
        coriolis_f,
        baroclinic_scale=1.0,
    )
    k1_raw = np.asarray(basic_raw["K1_raw"], dtype=np.float64)
    k2_raw = np.asarray(basic_raw["K2_raw"], dtype=np.float64)
    i2_raw = np.asarray(basic_raw["K3_raw"], dtype=np.float64)
    d_raw = k1_raw * i2_raw - k2_raw**2
    classes = classify_stability(k1_raw, i2_raw, d_raw)
    i2_vorticity = (
        np.asarray(basic_raw["chi"])
        * np.asarray(basic_raw["xi"])
        * np.asarray(basic_raw["zeta_abs"])
    )

    # Separate balanced projection.  This is useful for interpreting the
    # fixed-CTRL SE operator, but is not allowed to overwrite the actual CM1
    # stability fields above.
    theta_bal, thermal_wind_info = invert_balanced_theta(
        vt_zr,
        theta_model_zr,
        r_m,
        z_m,
        coriolis_f,
        theta_floor=theta_floor,
        outer_smooth_window=outer_smooth_window,
    )
    # Physical Bui coefficient: baroclinic_scale must remain one for the raw
    # applicability diagnosis.
    basic_balanced = build_basic_state(
        vt_zr,
        theta_bal,
        rho_zr,
        r_m,
        z_m,
        coriolis_f,
        baroclinic_scale=1.0,
    )
    k1_bal = np.asarray(basic_balanced["K1_raw"], dtype=np.float64)
    k2_bal = np.asarray(basic_balanced["K2_raw"], dtype=np.float64)
    i2_bal = np.asarray(basic_balanced["K3_raw"], dtype=np.float64)
    d_bal = k1_bal * i2_bal - k2_bal**2

    k1_reg, k2_reg, i2_reg, regularization_info = regularize_ellipticity(
        k1_raw,
        k2_raw,
        i2_raw,
        eps_ratio=regularization_eps_ratio,
        margin=regularization_margin,
    )
    d_reg = k1_reg * i2_reg - k2_reg**2
    changed = (
        (k1_reg != k1_raw)
        | (k2_reg != k2_raw)
        | (i2_reg != i2_raw)
    )
    fields = {
        "theta_bal": theta_bal,
        "static_stability_raw": k1_raw,
        "shear_coefficient_raw": k2_raw,
        "I2_raw": i2_raw,
        "D_raw": d_raw,
        "I2_vorticity_component_raw": i2_vorticity,
        "I2_baroclinic_component_raw": i2_raw - i2_vorticity,
        "D_static_inertial_product_raw": k1_raw * i2_raw,
        "D_shear_penalty_raw": k2_raw**2,
        "chi_raw": np.asarray(basic_raw["chi"]),
        "Cg_raw": np.asarray(basic_raw["Cg"]),
        "xi_raw": np.asarray(basic_raw["xi"]),
        "zeta_abs_raw": np.asarray(basic_raw["zeta_abs"]),
        "stability_class": classes,
        "elliptic_mask_raw": classes == 0,
        "inertial_unstable_mask_raw": i2_raw <= 0.0,
        "symmetric_nonelliptic_mask_raw": (k1_raw > 0.0) & (i2_raw > 0.0) & (d_raw <= 0.0),
        "static_unstable_mask_raw": k1_raw <= 0.0,
        "static_stability_balanced_projection": k1_bal,
        "shear_coefficient_balanced_projection": k2_bal,
        "I2_balanced_projection": i2_bal,
        "D_balanced_projection": d_bal,
        "stability_class_balanced_projection": classify_stability(k1_bal, i2_bal, d_bal),
        "K1_regularized": k1_reg,
        "K2_regularized": k2_reg,
        "I2_regularized": i2_reg,
        "D_regularized": d_reg,
        "regularization_changed_mask": changed,
        "thermal_wind_residual_raw": basic_raw["thermal_wind_residual"],
        "thermal_wind_residual_balanced_projection": basic_balanced["thermal_wind_residual"],
    }
    info: Dict[str, object] = {
        "thermal_wind": thermal_wind_info,
        "regularization": regularization_info,
        "raw_definition": (
            "actual CM1 azimuthal-mean theta/v: I2=K3_raw; "
            "D=K1_raw*I2-K2_raw^2; no balance projection or regularization"
        ),
        "balanced_projection_definition": (
            "theta from thermal-wind inversion; still unregularized; separate from raw"
        ),
        "raw_thermal_wind_residual_rms": float(
            np.sqrt(np.nanmean(np.asarray(basic_raw["thermal_wind_residual"]) ** 2))
        ),
    }
    return fields, info


def _cylindrical_cell_measure(r_m: np.ndarray, z_m: np.ndarray) -> np.ndarray:
    r = np.asarray(r_m, dtype=np.float64)
    z = np.asarray(z_m, dtype=np.float64)
    dr = np.abs(np.gradient(r)) if r.size > 1 else np.ones_like(r)
    dz = np.abs(np.gradient(z)) if z.size > 1 else np.ones_like(z)
    return np.maximum(r, 1.0)[None, :] * dr[None, :] * dz[:, None]


def _fraction(weights: np.ndarray, mask: np.ndarray) -> float:
    valid = np.isfinite(weights)
    denominator = float(np.sum(weights[valid]))
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(weights[valid & mask]) / denominator)


def summarize_case_overlap(
    case_fields: Mapping[str, np.ndarray],
    f_lambda_env: np.ndarray,
    r_m: np.ndarray,
    z_m: np.ndarray,
) -> Dict[str, float]:
    """Quantify SE applicability and its overlap with environmental forcing."""
    f_env = np.asarray(f_lambda_env, dtype=np.float64)
    i2 = np.asarray(case_fields["I2_raw"], dtype=np.float64)
    disc = np.asarray(case_fields["D_raw"], dtype=np.float64)
    classes = np.asarray(case_fields["stability_class"])
    if not (f_env.shape == i2.shape == disc.shape == classes.shape):
        raise ValueError("forcing and stability fields must have matching (z, r) shapes")

    measure = _cylindrical_cell_measure(r_m, z_m)
    valid_measure = measure * np.isfinite(i2) * np.isfinite(disc)
    forcing_weight = measure * np.nan_to_num(np.abs(f_env), nan=0.0)
    nonelliptic = classes != 0
    abs_finite = np.abs(f_env[np.isfinite(f_env)])
    positive_force = abs_finite[abs_finite > 0.0]
    strong_threshold = (
        float(np.nanpercentile(positive_force, 90.0)) if positive_force.size else 0.0
    )
    strong = (
        np.isfinite(f_env) & (np.abs(f_env) >= strong_threshold)
        if strong_threshold > 0.0
        else np.zeros_like(f_env, dtype=bool)
    )

    return {
        "elliptic_area_fraction": _fraction(valid_measure, classes == 0),
        "inertial_unstable_area_fraction": _fraction(valid_measure, classes == 1),
        "symmetric_nonelliptic_area_fraction": _fraction(valid_measure, classes == 2),
        "static_unstable_area_fraction": _fraction(valid_measure, classes == 3),
        "nonelliptic_abs_forcing_fraction": _fraction(forcing_weight, nonelliptic),
        "strong_forcing_nonelliptic_area_fraction": _fraction(measure * strong, nonelliptic),
        "strong_forcing_threshold_m_s2": strong_threshold,
        "raw_I2_min": float(np.nanmin(i2)),
        "raw_D_min": float(np.nanmin(disc)),
    }


def compute_applicability_diagnostics(
    ctrl: Mapping[str, np.ndarray],
    jet: Mapping[str, np.ndarray],
    coriolis_f: float,
    theta_floor: float = 150.0,
    outer_smooth_window: int = 1,
    regularization_eps_ratio: float = 1.0e-3,
    regularization_margin: float = 0.0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Compute CTRL/JET raw applicability fields and forcing-overlap metrics."""
    r_km = np.asarray(ctrl["r_km"], dtype=np.float64)
    z_km = np.asarray(ctrl["z_km"], dtype=np.float64)
    jet_r_km = np.asarray(jet["r_km"], dtype=np.float64)
    jet_z_km = np.asarray(jet["z_km"], dtype=np.float64)
    if r_km.shape != jet_r_km.shape or not np.allclose(r_km, jet_r_km, rtol=0.0, atol=1.0e-9):
        raise ValueError("CTRL and JET radial grids differ")
    if z_km.shape != jet_z_km.shape or not np.allclose(z_km, jet_z_km, rtol=0.0, atol=1.0e-9):
        raise ValueError("CTRL and JET vertical grids differ")
    r_m = r_km * 1000.0
    z_m = z_km * 1000.0

    ctrl_stability, ctrl_info = compute_case_stability(
        ctrl["ut"], ctrl["theta"], ctrl["rho"], r_m, z_m, coriolis_f,
        theta_floor, outer_smooth_window, regularization_eps_ratio,
        regularization_margin,
    )
    jet_stability, jet_info = compute_case_stability(
        jet["ut"], jet["theta"], jet["rho"], r_m, z_m, coriolis_f,
        theta_floor, outer_smooth_window, regularization_eps_ratio,
        regularization_margin,
    )
    f_env = environmental_difference(jet["F_lambda_eddy"], ctrl["F_lambda_eddy"])
    f_env_r = environmental_difference(
        jet["F_lambda_eddy_radial"], ctrl["F_lambda_eddy_radial"]
    )
    f_env_z = environmental_difference(
        jet["F_lambda_eddy_vertical"], ctrl["F_lambda_eddy_vertical"]
    )
    eke_ctrl = np.asarray(ctrl.get("eddy_kinetic_energy", np.zeros_like(f_env)))
    eke_jet = np.asarray(jet.get("eddy_kinetic_energy", np.zeros_like(f_env)))
    eddy_speed_ctrl = np.sqrt(np.maximum(2.0 * eke_ctrl, 0.0))
    eddy_speed_jet = np.sqrt(np.maximum(2.0 * eke_jet, 0.0))

    arrays: Dict[str, np.ndarray] = {
        "r_km": r_km,
        "z_km": z_km,
        "F_lambda_env": f_env,
        "F_lambda_env_radial": f_env_r,
        "F_lambda_env_vertical": f_env_z,
        "outflow_ctrl": np.asarray(ctrl["ur"], dtype=np.float64),
        "outflow_jet": np.asarray(jet["ur"], dtype=np.float64),
        "eddy_speed_ctrl": eddy_speed_ctrl,
        "eddy_speed_jet": eddy_speed_jet,
        "eddy_speed_env_change": eddy_speed_jet - eddy_speed_ctrl,
    }
    for prefix, source in (("ctrl", ctrl_stability), ("jet", jet_stability)):
        arrays.update({f"{name}_{prefix}": np.asarray(value) for name, value in source.items()})
    arrays["I2_raw_change"] = arrays["I2_raw_jet"] - arrays["I2_raw_ctrl"]
    arrays["D_raw_change"] = arrays["D_raw_jet"] - arrays["D_raw_ctrl"]
    arrays["F_lambda_env_ctrl_raw_elliptic"] = np.where(
        arrays["elliptic_mask_raw_ctrl"], f_env, 0.0
    )
    arrays["F_lambda_env_jet_raw_elliptic"] = np.where(
        arrays["elliptic_mask_raw_jet"], f_env, 0.0
    )
    arrays["F_lambda_env_ctrl_balanced_projection_elliptic"] = np.where(
        arrays["stability_class_balanced_projection_ctrl"] == 0, f_env, 0.0
    )
    arrays["F_lambda_env_jet_balanced_projection_elliptic"] = np.where(
        arrays["stability_class_balanced_projection_jet"] == 0, f_env, 0.0
    )

    finite_force = np.abs(f_env)
    if np.any(np.isfinite(finite_force)):
        max_index = np.unravel_index(np.nanargmax(finite_force), finite_force.shape)
    else:
        max_index = (0, 0)
    k, j = max_index
    ctrl_balanced_view = {
        "I2_raw": ctrl_stability["I2_balanced_projection"],
        "D_raw": ctrl_stability["D_balanced_projection"],
        "stability_class": ctrl_stability["stability_class_balanced_projection"],
    }
    jet_balanced_view = {
        "I2_raw": jet_stability["I2_balanced_projection"],
        "D_raw": jet_stability["D_balanced_projection"],
        "stability_class": jet_stability["stability_class_balanced_projection"],
    }
    ctrl_summary = {**ctrl_info, **summarize_case_overlap(ctrl_stability, f_env, r_m, z_m)}
    jet_summary = {**jet_info, **summarize_case_overlap(jet_stability, f_env, r_m, z_m)}
    ctrl_summary["balanced_projection_overlap"] = summarize_case_overlap(
        ctrl_balanced_view, f_env, r_m, z_m
    )
    jet_summary["balanced_projection_overlap"] = summarize_case_overlap(
        jet_balanced_view, f_env, r_m, z_m
    )
    summary: Dict[str, object] = {
        "definitions": {
            "I2_raw": "chi*xi*(zeta+f) + Cg*dchi/dr",
            "D_raw": "(-g*dchi/dz)*I2_raw - [d(chi*Cg)/dz]^2",
            "raw_basic_state": "actual CM1 azimuthal-mean theta/v; no balance projection",
            "balanced_projection": "thermal-wind-inverted theta; separate comparison fields",
            "elliptic": "static_stability_raw>0 and I2_raw>0 and D_raw>0",
            "environmental_forcing": "F_lambda_eddy(JET)-F_lambda_eddy(CTRL)",
            "eddy_speed": "sqrt(2*azimuthal_eddy_kinetic_energy); jet/asymmetry location proxy",
        },
        "ctrl": ctrl_summary,
        "jet": jet_summary,
        "maximum_abs_environmental_forcing": {
            "r_km": float(r_km[j]),
            "z_km": float(z_km[k]),
            "F_lambda_env_m_s2": float(f_env[k, j]),
            "I2_raw_ctrl": float(arrays["I2_raw_ctrl"][k, j]),
            "D_raw_ctrl": float(arrays["D_raw_ctrl"][k, j]),
            "I2_raw_jet": float(arrays["I2_raw_jet"][k, j]),
            "D_raw_jet": float(arrays["D_raw_jet"][k, j]),
            "I2_balanced_projection_ctrl": float(arrays["I2_balanced_projection_ctrl"][k, j]),
            "D_balanced_projection_ctrl": float(arrays["D_balanced_projection_ctrl"][k, j]),
            "I2_balanced_projection_jet": float(arrays["I2_balanced_projection_jet"][k, j]),
            "D_balanced_projection_jet": float(arrays["D_balanced_projection_jet"][k, j]),
        },
    }
    return arrays, summary


def _display_scale(fields: Tuple[np.ndarray, ...]) -> Tuple[float, float]:
    finite = np.concatenate([np.abs(field[np.isfinite(field)]) for field in fields])
    vmax = float(np.nanpercentile(finite, 99.0)) if finite.size else 1.0
    vmax = max(vmax, 1.0e-30)
    power = float(10.0 ** np.floor(np.log10(vmax)))
    return power, vmax / power


def _contour_if_crossed(ax, rr, zz, field, level, **kwargs):
    finite = field[np.isfinite(field)]
    if finite.size and float(np.nanmin(finite)) <= level <= float(np.nanmax(finite)):
        return ax.contour(rr, zz, field, levels=[level], **kwargs)
    return None


def plot_applicability_fields(
    arrays: Mapping[str, np.ndarray],
    out_file: Path,
    outflow_threshold_ms: float = 2.0,
    jet_speed_threshold_ms: float = 20.0,
    forcing_contour_percentile: float = 90.0,
    jet_axis_r_km: float | None = None,
    jet_axis_z_km: float | None = None,
) -> None:
    """Plot raw I2/D for both cases with forcing, outflow and jet proxies."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not 0.0 <= forcing_contour_percentile <= 100.0:
        raise ValueError("forcing_contour_percentile must be between 0 and 100")
    r = arrays["r_km"]
    z = arrays["z_km"]
    rr, zz = np.meshgrid(r, z)
    force = np.asarray(arrays["F_lambda_env"], dtype=np.float64)
    force_abs = np.abs(force[np.isfinite(force)])
    force_abs = force_abs[force_abs > 0.0]
    force_level = (
        float(np.nanpercentile(force_abs, forcing_contour_percentile))
        if force_abs.size else 0.0
    )

    i2_pair = (arrays["I2_raw_ctrl"], arrays["I2_raw_jet"])
    d_pair = (arrays["D_raw_ctrl"], arrays["D_raw_jet"])
    i2_scale, i2_vmax = _display_scale(i2_pair)
    d_scale, d_vmax = _display_scale(d_pair)
    di_scale, di_vmax = _display_scale((arrays["I2_raw_change"],))
    dd_scale, dd_vmax = _display_scale((arrays["D_raw_change"],))
    fields = (
        arrays["I2_raw_ctrl"] / i2_scale,
        arrays["I2_raw_jet"] / i2_scale,
        arrays["I2_raw_change"] / di_scale,
        arrays["D_raw_ctrl"] / d_scale,
        arrays["D_raw_jet"] / d_scale,
        arrays["D_raw_change"] / dd_scale,
    )
    titles = (
        rf"CTRL raw $I^2$ ($\times10^{{{int(np.log10(i2_scale))}}}$ K$^{{-1}}$ s$^{{-2}}$)",
        rf"JET raw $I^2$ ($\times10^{{{int(np.log10(i2_scale))}}}$ K$^{{-1}}$ s$^{{-2}}$)",
        rf"JET-CTRL raw $I^2$ ($\times10^{{{int(np.log10(di_scale))}}}$ K$^{{-1}}$ s$^{{-2}}$)",
        rf"CTRL raw $D$ ($\times10^{{{int(np.log10(d_scale))}}}$ K$^{{-2}}$ s$^{{-4}}$)",
        rf"JET raw $D$ ($\times10^{{{int(np.log10(d_scale))}}}$ K$^{{-2}}$ s$^{{-4}}$)",
        rf"JET-CTRL raw $D$ ($\times10^{{{int(np.log10(dd_scale))}}}$ K$^{{-2}}$ s$^{{-4}}$)",
    )
    vmaxes = (i2_vmax, i2_vmax, di_vmax, d_vmax, d_vmax, dd_vmax)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True, sharex=True, sharey=True)
    for ax, field, title, vmax in zip(axes.flat, fields, titles, vmaxes):
        levels = np.linspace(-vmax, vmax, 31)
        image = ax.contourf(rr, zz, field, levels=levels, cmap="RdBu_r", extend="both")
        _contour_if_crossed(ax, rr, zz, field, 0.0, colors="k", linewidths=1.0)
        if force_level > 0.0:
            _contour_if_crossed(ax, rr, zz, force, force_level, colors="black", linewidths=1.2)
            _contour_if_crossed(ax, rr, zz, force, -force_level, colors="black", linewidths=1.2, linestyles="--")
        _contour_if_crossed(
            ax, rr, zz, arrays["outflow_jet"], outflow_threshold_ms,
            colors="limegreen", linewidths=1.3,
        )
        _contour_if_crossed(
            ax, rr, zz, arrays["eddy_speed_jet"], jet_speed_threshold_ms,
            colors="darkorange", linewidths=1.3,
        )
        if jet_axis_r_km is not None and jet_axis_z_km is not None:
            ax.scatter(
                [jet_axis_r_km], [jet_axis_z_km], marker="*", s=90,
                facecolor="gold", edgecolor="black", zorder=10,
            )
        ax.set_title(title)
        ax.set_xlabel("Radius (km)")
        ax.set_ylabel("Height (km)")
        fig.colorbar(image, ax=ax, pad=0.015)
    legend = [
        Line2D([0], [0], color="black", lw=1.2, label=f"+Fenv p{forcing_contour_percentile:g}"),
        Line2D([0], [0], color="black", lw=1.2, ls="--", label=f"-Fenv p{forcing_contour_percentile:g}"),
        Line2D([0], [0], color="limegreen", lw=1.3, label=f"JET outflow={outflow_threshold_ms:g} m/s"),
        Line2D([0], [0], color="darkorange", lw=1.3, label=f"JET eddy speed={jet_speed_threshold_ms:g} m/s"),
    ]
    if jet_axis_r_km is not None and jet_axis_z_km is not None:
        legend.append(
            Line2D([0], [0], marker="*", color="none", markerfacecolor="gold",
                   markeredgecolor="black", markersize=10, label="imposed jet axis")
        )
    fig.legend(handles=legend, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_applicability_classes(
    arrays: Mapping[str, np.ndarray],
    out_file: Path,
    outflow_threshold_ms: float = 2.0,
    jet_speed_threshold_ms: float = 20.0,
    jet_axis_r_km: float | None = None,
    jet_axis_z_km: float | None = None,
) -> None:
    """Plot physical stability classes and Fenv overlap with D=0 boundaries."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    r = arrays["r_km"]
    z = arrays["z_km"]
    rr, zz = np.meshgrid(r, z)
    cmap = ListedColormap(["#2ca25f", "#d73027", "#fc8d59", "#756bb1"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    labels = (
        "Elliptic: K1>0, I2>0, D>0",
        "Inertial: K1>0, I2<=0",
        "Symmetric/shear: K1>0, I2>0, D<=0",
        "Static: K1<=0",
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True, sharey=True)
    for ax, key, title in zip(
        axes[:2],
        ("stability_class_ctrl", "stability_class_jet"),
        ("CTRL raw SE applicability", "JET raw SE applicability"),
    ):
        ax.pcolormesh(rr, zz, arrays[key], cmap=cmap, norm=norm, shading="auto")
        ax.set_title(title)
        ax.set_xlabel("Radius (km)")
        ax.set_ylabel("Height (km)")

    force = np.asarray(arrays["F_lambda_env"], dtype=np.float64)
    power, vmax = _display_scale((force,))
    levels = np.linspace(-vmax, vmax, 31)
    image = axes[2].contourf(rr, zz, force / power, levels=levels, cmap="RdBu_r", extend="both")
    _contour_if_crossed(axes[2], rr, zz, arrays["D_raw_ctrl"], 0.0, colors="royalblue", linewidths=1.5)
    _contour_if_crossed(axes[2], rr, zz, arrays["D_raw_jet"], 0.0, colors="red", linewidths=1.5, linestyles="--")
    _contour_if_crossed(axes[2], rr, zz, arrays["outflow_jet"], outflow_threshold_ms, colors="limegreen", linewidths=1.3)
    _contour_if_crossed(axes[2], rr, zz, arrays["eddy_speed_jet"], jet_speed_threshold_ms, colors="darkorange", linewidths=1.3)
    if jet_axis_r_km is not None and jet_axis_z_km is not None:
        for ax in axes:
            ax.scatter(
                [jet_axis_r_km], [jet_axis_z_km], marker="*", s=90,
                facecolor="gold", edgecolor="black", zorder=10,
            )
    axes[2].set_title(rf"$F_{{\lambda,env}}$ ($\times10^{{{int(np.log10(power))}}}$ m s$^{{-2}}$)")
    axes[2].set_xlabel("Radius (km)")
    axes[2].set_ylabel("Height (km)")
    fig.colorbar(image, ax=axes[2], pad=0.015)
    class_legend = [Patch(facecolor=cmap(i), label=label) for i, label in enumerate(labels)]
    axes[0].legend(handles=class_legend, loc="upper right", fontsize=8)
    axes[2].plot([], [], color="royalblue", lw=1.5, label="CTRL D=0")
    axes[2].plot([], [], color="red", lw=1.5, ls="--", label="JET D=0")
    axes[2].plot([], [], color="limegreen", lw=1.3, label="JET outflow")
    axes[2].plot([], [], color="darkorange", lw=1.3, label="JET eddy-speed proxy")
    if jet_axis_r_km is not None and jet_axis_z_km is not None:
        axes[2].scatter([], [], marker="*", s=70, facecolor="gold", edgecolor="black", label="imposed jet axis")
    axes[2].legend(loc="upper right", fontsize=8)
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_applicability_products(
    arrays: Mapping[str, np.ndarray],
    summary: Mapping[str, object],
    output_dir: str | Path,
    write_netcdf: bool = True,
    make_plots: bool = True,
    outflow_threshold_ms: float = 2.0,
    jet_speed_threshold_ms: float = 20.0,
    forcing_contour_percentile: float = 90.0,
    jet_axis_r_km: float | None = None,
    jet_axis_z_km: float | None = None,
) -> Dict[str, str]:
    """Write reusable arrays, summary statistics and server-side figures."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "se_applicability_products.npz"
    np.savez_compressed(npz_path, **arrays)

    nc_path = out_dir / "se_applicability_products.nc"
    if write_netcdf:
        import xarray as xr

        data_vars = {}
        nz = len(arrays["z_km"])
        nr = len(arrays["r_km"])
        for name, value in arrays.items():
            array = np.asarray(value)
            if array.shape == (nz, nr):
                data_vars[name] = (("z", "r"), array)
        dataset = xr.Dataset(
            data_vars=data_vars,
            coords={"z": arrays["z_km"], "r": arrays["r_km"]},
            attrs={
                "primary_fields": "I2_raw and D_raw are unregularized",
                "D_raw_definition": "K1_raw*I2_raw-K2_raw^2",
                "regularized_fields": "comparison only; not original SE applicability",
            },
        )
        for name in dataset.data_vars:
            if name.startswith("I2_") or name.startswith("static_stability_"):
                dataset[name].attrs["units"] = "K-1 s-2"
            elif name.startswith("D_"):
                dataset[name].attrs["units"] = "K-2 s-4"
            elif name.startswith("F_lambda_"):
                dataset[name].attrs["units"] = "m s-2"
            elif name.startswith("outflow_") or name.startswith("eddy_speed_"):
                dataset[name].attrs["units"] = "m s-1"
        dataset["I2_raw_ctrl"].attrs["description"] = "unregularized actual CTRL CM1 basic state"
        dataset["I2_raw_jet"].attrs["description"] = "unregularized actual JET CM1 basic state"
        dataset["D_raw_ctrl"].attrs["description"] = "unregularized actual CTRL Bui discriminant"
        dataset["D_raw_jet"].attrs["description"] = "unregularized actual JET Bui discriminant"
        dataset.to_netcdf(nc_path)

    fields_png = out_dir / "se_applicability_I2_D.png"
    classes_png = out_dir / "se_applicability_classes.png"
    if make_plots:
        plot_applicability_fields(
            arrays,
            fields_png,
            outflow_threshold_ms,
            jet_speed_threshold_ms,
            forcing_contour_percentile,
            jet_axis_r_km,
            jet_axis_z_km,
        )
        plot_applicability_classes(
            arrays,
            classes_png,
            outflow_threshold_ms,
            jet_speed_threshold_ms,
            jet_axis_r_km,
            jet_axis_z_km,
        )

    summary_path = out_dir / "se_applicability_summary.json"
    complete_summary = dict(summary)
    complete_summary["outputs"] = {
        "npz": npz_path.as_posix(),
        "netcdf": nc_path.as_posix() if write_netcdf else "",
        "I2_D_png": fields_png.as_posix() if make_plots else "",
        "classes_png": classes_png.as_posix() if make_plots else "",
    }
    summary_path.write_text(
        json.dumps(complete_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return complete_summary["outputs"]
