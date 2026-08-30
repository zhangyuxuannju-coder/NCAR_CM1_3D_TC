#!/usr/bin/env python3
"""Regularized full-domain four-cell SE attribution in radial wind.

This complements ``run_se_forcing_operator_factorial.py``.  The older script
uses only a common raw-elliptic subdomain and plots streamfunction.  This script
uses the full requested r-z domain, regularizes the CTRL and JET operators
separately, solves all four operator/forcing combinations, converts every
solution to radial/vertical wind, and attributes the JET-minus-CTRL difference.

The resulting circulation is a regularized balanced projection.  Hatched cells
in the figures identify locations where at least one raw operator coefficient
had to be changed to recover ellipticity.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

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
    assemble_operator,
    build_basic_state,
    build_forcing,
    invert_balanced_theta,
    regularize_ellipticity,
)


NATURE_DIVERGING = LinearSegmentedColormap.from_list(
    "nature_diverging",
    ["#3C5488", "#8FAACC", "#F7F7F4", "#EFA287", "#B9363E"],
)
DOMINANCE_CMAP = LinearSegmentedColormap.from_list(
    "forcing_operator",
    ["#3C5488", "#B9C9DF", "#F7F7F4", "#F1B39A", "#B9363E"],
)
COLORS = {
    "forcing": "#B9363E",
    "operator": "#3C5488",
    "interaction": "#009E73",
    "total": "#4D4D4D",
    "CTRL": "#3C5488",
    "JET": "#B9363E",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regularized full-domain SE forcing/operator attribution in radial wind"
    )
    p.add_argument("--nojet", required=True)
    p.add_argument("--jet", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--hours", type=float, nargs="+", default=[20.0, 55.0, 80.0, 110.0])
    p.add_argument("--max-r-km", type=float, default=300.0)
    p.add_argument("--max-z-km", type=float, default=20.0)
    p.add_argument("--dr-km", type=float, default=12.0)
    p.add_argument("--f", type=float, default=5.464e-5)
    p.add_argument("--eps-ratio", type=float, default=1.0e-3)
    p.add_argument("--elliptic-margin", type=float, default=0.0)
    p.add_argument("--baroclinic-scale", type=float, default=1.0)
    return p.parse_args()


def azimuthal_average(path: str, hour: float, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    cfg = PipelineConfig(
        input_file=path,
        output_dir=args.output_dir,
        target_time_hours=hour,
        max_r_km=args.max_r_km,
        dr_km=args.dr_km,
        max_z_km=args.max_z_km,
        coriolis_f=args.f,
        include_model_budget_terms=True,
        write_netcdf=False,
        write_ieee=False,
        plot_solution=False,
        sor_max_iter=80000,
        sor_omega=1.5,
        sor_tol=1.0e-14,
        sor_verbose_every=0,
    )
    return azimuthal_average_from_3d(cfg)


def check_grids(ctrl: Dict[str, np.ndarray], jet: Dict[str, np.ndarray]) -> None:
    for key in ("r_km", "z_km"):
        a = np.asarray(ctrl[key], dtype=np.float64)
        b = np.asarray(jet[key], dtype=np.float64)
        if a.shape != b.shape or not np.allclose(a, b, rtol=0.0, atol=1.0e-9):
            raise ValueError(f"CTRL/JET {key} grids differ")


def build_case(
    avg: Dict[str, np.ndarray], r_m: np.ndarray, z_m: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, object]:
    theta_bal, thermal_wind = invert_balanced_theta(
        avg["ut"], avg["theta"], r_m, z_m, args.f
    )
    basic = build_basic_state(
        avg["ut"], theta_bal, avg["rho"], r_m, z_m, args.f,
        baroclinic_scale=args.baroclinic_scale,
    )
    k1_raw = np.asarray(basic["K1_raw"], dtype=np.float64)
    k2_raw = np.asarray(basic["K2_raw"], dtype=np.float64)
    k3_raw = np.asarray(basic["K3_raw"], dtype=np.float64)
    d_raw = k1_raw * k3_raw - k2_raw**2
    k1, k2, k3, reg_info = regularize_ellipticity(
        k1_raw, k2_raw, k3_raw,
        eps_ratio=args.eps_ratio,
        margin=args.elliptic_margin,
    )
    d_reg = k1 * k3 - k2**2
    scale1 = max(float(np.nanmax(np.abs(k1_raw))), 1.0e-30)
    scale2 = max(float(np.nanmax(np.abs(k2_raw))), 1.0e-30)
    scale3 = max(float(np.nanmax(np.abs(k3_raw))), 1.0e-30)
    changed = (
        np.abs(k1 - k1_raw) > 1.0e-12 * scale1
    ) | (
        np.abs(k2 - k2_raw) > 1.0e-12 * scale2
    ) | (
        np.abs(k3 - k3_raw) > 1.0e-12 * scale3
    )
    raw_nonelliptic = (
        (~np.isfinite(d_raw)) | (~np.isfinite(k1_raw)) | (~np.isfinite(k3_raw))
        | (k1_raw <= 0.0) | (k3_raw <= 0.0) | (d_raw <= 0.0)
    )
    return {
        "basic": basic,
        "operator": assemble_operator(basic, k1, k2, k3, r_m, z_m),
        "theta_bal": theta_bal,
        "thermal_wind": thermal_wind,
        "K1_raw": k1_raw,
        "K2_raw": k2_raw,
        "K3_raw": k3_raw,
        "D_raw": d_raw,
        "K1_reg": k1,
        "K2_reg": k2,
        "K3_reg": k3,
        "D_reg": d_reg,
        "changed": changed,
        "raw_nonelliptic": raw_nonelliptic,
        "regularization": reg_info,
    }


def solve_one(
    case: Dict[str, object], q: np.ndarray, f_lambda: np.ndarray,
    r_m: np.ndarray, z_m: np.ndarray,
) -> Dict[str, np.ndarray]:
    rhs = build_forcing(case["basic"], q, f_lambda, r_m, z_m)
    operator = case["operator"]
    arrays = {
        key: _to_solver_layout_zr_to_rz(operator[key])
        for key in ("A", "B", "C", "D", "E")
    }
    dr = float(np.mean(np.diff(r_m)))
    dz = float(np.mean(np.diff(z_m)))
    psi = solve_se_sparse(
        A=arrays["A"], B=arrays["B"], C=arrays["C"],
        D=arrays["D"], E=arrays["E"],
        Fin=_to_solver_layout_zr_to_rz(rhs["forcing_total"]),
        dr=dr, dz=dz,
    )
    rho_ext = _rho_ext_from_rho_zr(case["basic"]["rho"])
    u, w = psi_to_uw(psi, rho_ext, r_m, dr, dz)
    return {
        "psi": psi[:, 1:-1].T,
        "u": u[:, 1:-1].T,
        "w": w[:, 1:-1].T,
        "forcing_total": rhs["forcing_total"],
        "forcing_thermal": rhs["forcing_thermal"],
        "forcing_momentum": rhs["forcing_momentum"],
    }


def weighted_rms(field: np.ndarray, weight: np.ndarray, mask: np.ndarray) -> float:
    good = mask & np.isfinite(field) & np.isfinite(weight) & (weight > 0.0)
    if not np.any(good):
        return float("nan")
    return float(np.sqrt(np.sum(weight[good] * field[good] ** 2) / np.sum(weight[good])))


def projection_share(
    component: np.ndarray, total: np.ndarray, weight: np.ndarray, mask: np.ndarray,
) -> float:
    good = (
        mask & np.isfinite(component) & np.isfinite(total)
        & np.isfinite(weight) & (weight > 0.0)
    )
    if not np.any(good):
        return float("nan")
    denom = float(np.sum(weight[good] * total[good] ** 2))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(weight[good] * component[good] * total[good]) / denom)


def metric_block(
    effects: Dict[str, np.ndarray], ctrl: Dict[str, np.ndarray], jet: Dict[str, np.ndarray],
    r_km: np.ndarray, z_km: np.ndarray,
) -> Dict[str, object]:
    rr = np.broadcast_to(np.maximum(r_km[None, :], 0.5), effects["total"].shape)
    zz = np.broadcast_to(z_km[:, None], effects["total"].shape)
    finite = np.isfinite(effects["total"])
    upper = finite & (zz >= 10.0) & (zz <= 18.0)
    outflow_union = upper & ((ctrl["ur"] >= 2.0) | (jet["ur"] >= 2.0))
    masks = {"full": finite, "upper_10_18km": upper, "upper_outflow_union": outflow_union}
    result: Dict[str, object] = {}
    for mask_name, mask in masks.items():
        rms = {key: weighted_rms(value, rr, mask) for key, value in effects.items()}
        shares = {
            key: projection_share(effects[key], effects["total"], rr, mask)
            for key in ("forcing", "operator", "interaction")
        }
        ratio = rms["forcing"] / max(rms["operator"], 1.0e-30)
        if ratio > 1.2:
            dominant = "forcing"
        elif ratio < 1.0 / 1.2:
            dominant = "operator"
        else:
            dominant = "comparable"
        result[mask_name] = {
            "cylindrical_r_weighted_rms_m_s": rms,
            "forcing_to_operator_rms_ratio": float(ratio),
            "projection_share_on_total": shares,
            "dominant_by_20pct_rule": dominant,
            "point_count": int(np.count_nonzero(mask)),
        }
    return result


def solve_hour(hour: float, args: argparse.Namespace) -> Dict[str, object]:
    print(f"\n=== full-domain factorial SE: {hour:g} h ===", flush=True)
    ctrl = azimuthal_average(args.nojet, hour, args)
    jet = azimuthal_average(args.jet, hour, args)
    check_grids(ctrl, jet)
    r_km = np.asarray(ctrl["r_km"], dtype=np.float64)
    z_km = np.asarray(ctrl["z_km"], dtype=np.float64)
    r_m = r_km * 1000.0
    z_m = z_km * 1000.0
    cases = {
        "CTRL": build_case(ctrl, r_m, z_m, args),
        "JET": build_case(jet, r_m, z_m, args),
    }
    sources = {
        "CTRL": (np.asarray(ctrl["Q"]), np.asarray(ctrl["Fnu"])),
        "JET": (np.asarray(jet["Q"]), np.asarray(jet["Fnu"])),
    }
    cells: Dict[str, Dict[str, np.ndarray]] = {}
    for op_name in ("CTRL", "JET"):
        for forcing_name in ("CTRL", "JET"):
            key = op_name[0] + forcing_name[0]
            print(f"  solving {key}: {op_name} operator / {forcing_name} forcing", flush=True)
            q, f_lambda = sources[forcing_name]
            cells[key] = solve_one(cases[op_name], q, f_lambda, r_m, z_m)

    effects: Dict[str, Dict[str, np.ndarray]] = {}
    for variable in ("psi", "u", "w"):
        cc = cells["CC"][variable]
        cj = cells["CJ"][variable]
        jc = cells["JC"][variable]
        jj = cells["JJ"][variable]
        effects[variable] = {
            "forcing": cj - cc,
            "operator": jc - cc,
            "interaction": jj - jc - cj + cc,
            "total": jj - cc,
        }
        closure = (
            effects[variable]["forcing"] + effects[variable]["operator"]
            + effects[variable]["interaction"] - effects[variable]["total"]
        )
        print(f"  {variable} factorial closure max={np.nanmax(np.abs(closure)):.3e}")

    metrics = metric_block(effects["u"], ctrl, jet, r_km, z_km)
    summary = {
        "hour": float(hour),
        "domain": {
            "r_min_km": float(r_km[0]), "r_max_km": float(r_km[-1]),
            "z_min_km": float(z_km[0]), "z_max_km": float(z_km[-1]),
        },
        "interpretation": "regularized balanced projection; not the raw unstable-state circulation",
        "regularization": {},
        "radial_wind_attribution": metrics,
    }
    for name in ("CTRL", "JET"):
        case = cases[name]
        summary["regularization"][name] = {
            **case["regularization"],
            "raw_nonelliptic_fraction": float(np.mean(case["raw_nonelliptic"])),
            "changed_coefficient_fraction": float(np.mean(case["changed"])),
            "min_D_regularized": float(np.nanmin(case["D_reg"])),
        }
    return {
        "hour": hour, "r_km": r_km, "z_km": z_km,
        "ctrl": ctrl, "jet": jet, "cases": cases,
        "cells": cells, "effects": effects, "summary": summary,
    }


def robust_limit(fields: Iterable[np.ndarray], percentile: float = 98.5) -> float:
    values = np.concatenate([np.abs(np.asarray(x)[np.isfinite(x)]) for x in fields])
    if values.size == 0:
        return 1.0
    return max(float(np.nanpercentile(values, percentile)), 1.0e-4)


def plot_effect_grid(results: list[Dict[str, object]], path: Path) -> None:
    nrow = len(results)
    fig, axes = plt.subplots(
        nrow, 4, figsize=(15.5, 3.05 * nrow), sharex=True, sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    keys = ("forcing", "operator", "interaction", "total")
    titles = ("Forcing effect", "Operator effect", "Interaction", "Total JET − CTRL")
    all_fields = [
        result["effects"]["u"][key] for result in results for key in keys
    ]
    vmax = robust_limit(all_fields)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    for row, result in enumerate(results):
        fields = result["effects"]["u"]
        union_changed = result["cases"]["CTRL"]["changed"] | result["cases"]["JET"]["changed"]
        for col, key in enumerate(keys):
            ax = axes[row, col]
            mesh = ax.pcolormesh(
                result["r_km"], result["z_km"], fields[key],
                cmap=NATURE_DIVERGING, norm=norm, shading="auto", rasterized=True,
            )
            mask = result["cases"]["CTRL"]["changed"] if key == "forcing" else union_changed
            ax.contourf(
                result["r_km"], result["z_km"], mask.astype(float),
                levels=[0.5, 1.5], colors="none", hatches=["...."], alpha=0.0,
            )
            ax.axhline(10.0, color="0.35", lw=0.55, ls=":")
            ax.axhline(18.0, color="0.35", lw=0.55, ls=":")
            if row == 0:
                ax.set_title(titles[col], fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{result['hour']:g} h\nHeight (km)")
            if row == nrow - 1:
                ax.set_xlabel("Radius (km)")
    cbar = fig.colorbar(mesh, ax=axes, pad=0.012, fraction=0.022)
    cbar.set_label(f"Radial-wind response (m s$^{{-1}}$); shared limit ±{vmax:.2f}")
    fig.suptitle(
        "Regularized full-domain SE attribution in radial wind\n"
        "stippled: coefficients modified to restore ellipticity",
        fontweight="bold",
    )
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_four_cells(result: Dict[str, object], path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.0), sharex=True, sharey=True, constrained_layout=True)
    cell_keys = ("CC", "CJ", "JC", "JJ")
    cell_titles = (
        "CTRL op / CTRL forcing", "CTRL op / JET forcing",
        "JET op / CTRL forcing", "JET op / JET forcing",
    )
    effect_keys = ("forcing", "operator", "interaction", "total")
    effect_titles = ("Forcing effect", "Operator effect", "Interaction", "Total JET − CTRL")
    cell_fields = [result["cells"][key]["u"] for key in cell_keys]
    effect_fields = [result["effects"]["u"][key] for key in effect_keys]
    limits = (robust_limit(cell_fields), robust_limit(effect_fields))
    for row, (fields, titles, vmax) in enumerate(zip((cell_fields, effect_fields), (cell_titles, effect_titles), limits)):
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        for col, (field, title) in enumerate(zip(fields, titles)):
            ax = axes[row, col]
            mesh = ax.pcolormesh(result["r_km"], result["z_km"], field, cmap=NATURE_DIVERGING, norm=norm, shading="auto", rasterized=True)
            ax.set_title(title)
            ax.set_xlabel("Radius (km)")
            if col == 0:
                ax.set_ylabel("Height (km)")
        cbar = fig.colorbar(mesh, ax=axes[row, :], pad=0.015, fraction=0.025)
        cbar.set_label("SE radial wind (m s$^{-1}$)" if row == 0 else "Radial-wind effect (m s$^{-1}$)")
    fig.suptitle(f"Full-domain four-cell SE radial-wind attribution at {result['hour']:g} h", fontweight="bold")
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_dominance(results: list[Dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(4.0 * len(results), 3.8), sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, result in zip(axes, results):
        f = np.abs(result["effects"]["u"]["forcing"])
        l = np.abs(result["effects"]["u"]["operator"])
        eps = max(float(np.nanpercentile(np.concatenate([f.ravel(), l.ravel()]), 25)) * 0.05, 1.0e-4)
        dominance = np.clip(np.log10((f + eps) / (l + eps)), -2.0, 2.0)
        mesh = ax.pcolormesh(
            result["r_km"], result["z_km"], dominance,
            cmap=DOMINANCE_CMAP, norm=TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0),
            shading="auto", rasterized=True,
        )
        ax.contour(result["r_km"], result["z_km"], dominance, levels=[0.0], colors="0.15", linewidths=0.7)
        outflow = (result["ctrl"]["ur"] >= 2.0) | (result["jet"]["ur"] >= 2.0)
        ax.contour(result["r_km"], result["z_km"], outflow.astype(float), levels=[0.5], colors="#009E73", linewidths=0.8)
        ax.set_title(f"{result['hour']:g} h")
        ax.set_xlabel("Radius (km)")
    axes[0].set_ylabel("Height (km)")
    cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", shrink=0.72, pad=0.12)
    cbar.set_label(r"$\log_{10}[(|\Delta u_F|+\epsilon)/(|\Delta u_L|+\epsilon)]$  ← operator | forcing →")
    fig.suptitle("Where forcing or operator changes dominate the SE radial-wind response\ngreen contour: CTRL/JET upper-outflow union ($u_r\\geq2$ m s$^{-1}$)", fontweight="bold")
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_summary(results: list[Dict[str, object]], path: Path) -> None:
    hours = np.asarray([result["hour"] for result in results], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), constrained_layout=True)
    keys = ("forcing", "operator", "interaction", "total")
    for key in keys:
        values = [
            result["summary"]["radial_wind_attribution"]["full"]
            ["cylindrical_r_weighted_rms_m_s"][key]
            for result in results
        ]
        axes[0].plot(hours, values, marker="o", lw=1.8, color=COLORS[key], label=key.capitalize())
    axes[0].set_xlabel("Time (h)")
    axes[0].set_ylabel("Cylindrical r-weighted RMS (m s$^{-1}$)")
    axes[0].set_title("Full-domain response amplitude")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.22)

    width = 5.5
    offsets = (-width, 0.0, width)
    for key, offset in zip(("forcing", "operator", "interaction"), offsets):
        values = [
            result["summary"]["radial_wind_attribution"]["full"]
            ["projection_share_on_total"][key]
            for result in results
        ]
        axes[1].bar(hours + offset, values, width=width * 0.86, color=COLORS[key], label=key.capitalize())
    axes[1].axhline(0.0, color="0.25", lw=0.7)
    axes[1].axhline(1.0, color="0.55", lw=0.6, ls=":")
    axes[1].set_xlabel("Time (h)")
    axes[1].set_ylabel("Projection on total difference")
    axes[1].set_title("Signed attribution (three terms sum to 1)")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.22)

    for name in ("CTRL", "JET"):
        raw = [result["summary"]["regularization"][name]["raw_nonelliptic_fraction"] for result in results]
        changed = [result["summary"]["regularization"][name]["changed_coefficient_fraction"] for result in results]
        axes[2].plot(hours, raw, marker="o", color=COLORS[name], lw=1.8, label=f"{name} raw non-elliptic")
        axes[2].plot(hours, changed, marker="s", color=COLORS[name], lw=1.2, ls="--", label=f"{name} modified")
    axes[2].set_xlabel("Time (h)")
    axes[2].set_ylabel("Grid-point fraction")
    axes[2].set_ylim(bottom=0.0)
    axes[2].set_title("Dependence on regularization")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(alpha=0.22)

    fig.suptitle("Forcing versus operator control of the full-domain SE radial wind", fontweight="bold")
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_products(result: Dict[str, object], out: Path) -> None:
    hour_tag = f"{result['hour']:06.1f}h".replace(".", "p")
    arrays: Dict[str, np.ndarray] = {
        "r_km": result["r_km"], "z_km": result["z_km"],
        "ur_ctrl_cm1": result["ctrl"]["ur"], "ur_jet_cm1": result["jet"]["ur"],
    }
    for key, cell in result["cells"].items():
        for variable in ("psi", "u", "w"):
            arrays[f"{variable}_{key}"] = cell[variable]
    for variable, effects in result["effects"].items():
        for key, value in effects.items():
            arrays[f"{variable}_{key}_effect"] = value
    for case_name, case in result["cases"].items():
        for key in ("K1_raw", "K2_raw", "K3_raw", "D_raw", "K1_reg", "K2_reg", "K3_reg", "D_reg", "changed", "raw_nonelliptic"):
            arrays[f"{key}_{case_name.lower()}"] = case[key]
    np.savez_compressed(out / f"full_domain_factorial_{hour_tag}.npz", **arrays)
    (out / f"full_domain_factorial_{hour_tag}.json").write_text(
        json.dumps(result["summary"], indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = [solve_hour(hour, args) for hour in args.hours]
    for result in results:
        save_products(result, out)
    plot_effect_grid(results, out / "figure_full_domain_radial_wind_effects.png")
    plot_four_cells(results[0], out / "figure_full_domain_four_cell_first_hour.png")
    plot_dominance(results, out / "figure_forcing_vs_operator_dominance.png")
    plot_summary(results, out / "figure_full_domain_attribution_summary.png")
    combined = {
        "configuration": {
            "nojet": args.nojet, "jet": args.jet, "hours": args.hours,
            "max_r_km": args.max_r_km, "max_z_km": args.max_z_km,
            "dr_km": args.dr_km, "f": args.f,
            "eps_ratio": args.eps_ratio,
            "elliptic_margin": args.elliptic_margin,
            "baroclinic_scale": args.baroclinic_scale,
        },
        "results": [result["summary"] for result in results],
    }
    (out / "full_domain_factorial_summary.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(combined, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
