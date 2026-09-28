#!/usr/bin/env python3
"""Validate whether direct CM1 diabatic heating explains JET-CTRL circulation changes.

The diagnostic uses the CM1 potential-temperature budget terms assembled as
``Q_diabatic`` by the production SE pipeline (normally ``ptb_mp + ptb_rad``).
For each strength-matched pair it holds the CTRL SE operator fixed and solves
responses to (1) the diabatic-heating difference and (2) all remaining forcing
differences.  It also compares these balanced responses with the observed
azimuthal-mean JET-minus-CTRL radial and vertical winds.

The diabatic-heating difference is further split exactly into amplitude and
normalized spatial-structure effects with a symmetric two-factor decomposition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_se_forcing_operator_factorial_full as base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nojet", required=True)
    p.add_argument("--jet", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--pairs", nargs="+", required=True, help="JET:CTRL hours")
    p.add_argument("--eps-ratio", type=float, default=1.0e-4)
    p.add_argument("--max-r-km", type=float, default=300.0)
    p.add_argument("--max-z-km", type=float, default=20.0)
    p.add_argument("--dr-km", type=float, default=12.0)
    p.add_argument("--f", type=float, default=6.2e-5)
    p.add_argument("--elliptic-margin", type=float, default=0.0)
    p.add_argument("--baroclinic-scale", type=float, default=1.0)
    return p.parse_args()


REGIONS = {
    "bl_inflow": ("u", (50.0, 300.0, 0.5, 2.0)),
    "inner_updraft": ("w", (20.0, 150.0, 2.0, 12.0)),
    "upper_outflow": ("u", (100.0, 300.0, 10.0, 17.0)),
}


def region_mask(r: np.ndarray, z: np.ndarray, bounds: tuple[float, ...]) -> np.ndarray:
    r0, r1, z0, z1 = bounds
    rr = np.broadcast_to(r[None, :], (z.size, r.size))
    zz = np.broadcast_to(z[:, None], (z.size, r.size))
    return (rr >= r0) & (rr <= r1) & (zz >= z0) & (zz <= z1)


def weighted_projection(pred: np.ndarray, obs: np.ndarray, r: np.ndarray, mask: np.ndarray) -> float:
    weight = np.broadcast_to(np.maximum(r[None, :], 0.5), pred.shape)
    good = mask & np.isfinite(pred) & np.isfinite(obs)
    den = float(np.sum(weight[good] * obs[good] ** 2))
    if den <= 0.0:
        return float("nan")
    return float(np.sum(weight[good] * pred[good] * obs[good]) / den)


def weighted_correlation(a: np.ndarray, b: np.ndarray, r: np.ndarray, mask: np.ndarray) -> float:
    weight = np.broadcast_to(np.maximum(r[None, :], 0.5), a.shape)
    good = mask & np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(good) < 3:
        return float("nan")
    w = weight[good]
    av = a[good]
    bv = b[good]
    am = float(np.sum(w * av) / np.sum(w))
    bm = float(np.sum(w * bv) / np.sum(w))
    da = av - am
    db = bv - bm
    den = np.sqrt(np.sum(w * da**2) * np.sum(w * db**2))
    return float(np.sum(w * da * db) / den) if den > 0.0 else float("nan")


def weighted_rms(a: np.ndarray, r: np.ndarray, mask: np.ndarray) -> float:
    weight = np.broadcast_to(np.maximum(r[None, :], 0.5), a.shape)
    good = mask & np.isfinite(a)
    return float(np.sqrt(np.sum(weight[good] * a[good] ** 2) / np.sum(weight[good])))


def heating_amplitude(q: np.ndarray, r_m: np.ndarray, z_m: np.ndarray) -> float:
    """Cylindrical integral of positive heating; the omitted 2*pi cancels."""
    dr = float(np.mean(np.diff(r_m)))
    dz = float(np.mean(np.diff(z_m)))
    return float(np.sum(np.maximum(q, 0.0) * r_m[None, :]) * dr * dz)


def heating_geometry(q: np.ndarray, r_km: np.ndarray, z_km: np.ndarray) -> dict[str, float]:
    pos = np.maximum(q, 0.0)
    weight = pos * r_km[None, :]
    total = float(np.sum(weight))
    if total <= 0.0:
        return {"positive_weight": 0.0, "centroid_r_km": np.nan, "centroid_z_km": np.nan,
                "peak_r_km": np.nan, "peak_z_km": np.nan, "peak_K_per_h": np.nan}
    iz, ir = np.unravel_index(np.nanargmax(q), q.shape)
    return {
        "positive_weight": total,
        "centroid_r_km": float(np.sum(weight * r_km[None, :]) / total),
        "centroid_z_km": float(np.sum(weight * z_km[:, None]) / total),
        "peak_r_km": float(r_km[ir]),
        "peak_z_km": float(z_km[iz]),
        "peak_K_per_h": float(q[iz, ir] * 3600.0),
    }


def amplitude_structure_split(
    q_ctrl: np.ndarray, q_jet: np.ndarray, r_m: np.ndarray, z_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Exact symmetric split of q_jet-q_ctrl into amplitude and structure."""
    a_ctrl = heating_amplitude(q_ctrl, r_m, z_m)
    a_jet = heating_amplitude(q_jet, r_m, z_m)
    if min(a_ctrl, a_jet) <= 0.0:
        raise ValueError("positive diabatic-heating integral must be nonzero")
    shape_ctrl = q_ctrl / a_ctrl
    shape_jet = q_jet / a_jet
    amp = 0.5 * (a_jet - a_ctrl) * (shape_ctrl + shape_jet)
    structure = 0.5 * (a_ctrl + a_jet) * (shape_jet - shape_ctrl)
    return amp, structure, a_ctrl, a_jet


def difference(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(a[key]) - np.asarray(b[key]) for key in ("u", "w")}


def solve_pair(jh: float, ch: float, args: argparse.Namespace, out: Path):
    print(f"PAIR J{jh:g}/C{ch:g}", flush=True)
    ctrl = base.azimuthal_average(args.nojet, ch, args)
    jet = base.azimuthal_average(args.jet, jh, args)
    base.check_grids(ctrl, jet)
    r = np.asarray(ctrl["r_km"], float)
    z = np.asarray(ctrl["z_km"], float)
    r_m = r * 1000.0
    z_m = z * 1000.0
    case_ctrl = base.build_case(ctrl, r_m, z_m, args)
    case_jet = base.build_case(jet, r_m, z_m, args)
    zero = np.zeros_like(ctrl["Q"])

    delta_q_diab = jet["Q_diabatic"] - ctrl["Q_diabatic"]
    delta_q_total = jet["Q"] - ctrl["Q"]
    delta_f_total = jet["Fnu"] - ctrl["Fnu"]
    delta_q_other = delta_q_total - delta_q_diab
    q_amp, q_structure, a_ctrl, a_jet = amplitude_structure_split(
        ctrl["Q_diabatic"], jet["Q_diabatic"], r_m, z_m
    )

    diab = base.solve_one(case_ctrl, delta_q_diab, zero, r_m, z_m)
    amp = base.solve_one(case_ctrl, q_amp, zero, r_m, z_m)
    structure = base.solve_one(case_ctrl, q_structure, zero, r_m, z_m)
    non_diab = base.solve_one(case_ctrl, delta_q_other, delta_f_total, r_m, z_m)
    forcing = base.solve_one(case_ctrl, delta_q_total, delta_f_total, r_m, z_m)
    cc = base.solve_one(case_ctrl, ctrl["Q"], ctrl["Fnu"], r_m, z_m)
    cj = base.solve_one(case_ctrl, jet["Q"], jet["Fnu"], r_m, z_m)
    jj = base.solve_one(case_jet, jet["Q"], jet["Fnu"], r_m, z_m)
    operator_remainder = difference(jj, cj)
    balanced = difference(jj, cc)
    observed = {"u": jet["ur"] - ctrl["ur"], "w": jet["w"] - ctrl["w"]}

    effects = {
        "observed": observed,
        "Q_diabatic": diab,
        "Q_amplitude": amp,
        "Q_structure": structure,
        "non_diabatic_forcing": non_diab,
        "all_forcing": forcing,
        "operator_remainder": operator_remainder,
        "full_balanced": balanced,
    }

    q_split_closure = float(
        np.linalg.norm(q_amp + q_structure - delta_q_diab)
        / max(np.linalg.norm(delta_q_diab), 1.0e-30)
    )
    response_split_closure = max(
        float(np.linalg.norm(amp[v] + structure[v] - diab[v]) / max(np.linalg.norm(diab[v]), 1.0e-30))
        for v in ("u", "w")
    )
    forcing_closure = max(
        float(np.linalg.norm(diab[v] + non_diab[v] - forcing[v]) / max(np.linalg.norm(forcing[v]), 1.0e-30))
        for v in ("u", "w")
    )
    absolute_linearity = max(
        float(np.linalg.norm((cj[v] - cc[v]) - forcing[v]) / max(np.linalg.norm(forcing[v]), 1.0e-30))
        for v in ("u", "w")
    )

    rows = []
    for region, (var, bounds) in REGIONS.items():
        mask = region_mask(r, z, bounds)
        obs = observed[var]
        obs_rms = weighted_rms(obs, r, mask)
        for name, effect in effects.items():
            field = effect[var]
            rows.append({
                "jet_hour": jh,
                "ctrl_hour": ch,
                "region": region,
                "variable": var,
                "effect": name,
                "mean_m_s": float(np.nanmean(field[mask])),
                "rms_m_s": weighted_rms(field, r, mask),
                "projection_on_observed": 1.0 if name == "observed" else weighted_projection(field, obs, r, mask),
                "pattern_correlation_with_observed": 1.0 if name == "observed" else weighted_correlation(field, obs, r, mask),
                "normalized_rmse_vs_observed": 0.0 if name == "observed" else weighted_rms(field - obs, r, mask) / max(obs_rms, 1.0e-30),
                "projection_on_all_forcing": np.nan if name == "observed" else weighted_projection(field, forcing[var], r, mask),
                "q_split_closure": q_split_closure,
                "response_split_closure": response_split_closure,
                "forcing_response_closure": forcing_closure,
                "absolute_linearity_closure": absolute_linearity,
            })

    heating_rows = []
    for experiment, q in (("CTRL", ctrl["Q_diabatic"]), ("JET", jet["Q_diabatic"])):
        item = {"jet_hour": jh, "ctrl_hour": ch, "experiment": experiment}
        item.update(heating_geometry(q, r, z))
        item["positive_integral_arbitrary"] = a_ctrl if experiment == "CTRL" else a_jet
        heating_rows.append(item)

    tag = f"J{jh:05.1f}_C{ch:05.1f}".replace(".", "p")
    np.savez_compressed(
        out / f"{tag}_diabatic_validation.npz",
        r_km=r,
        z_km=z,
        q_diab_ctrl=ctrl["Q_diabatic"],
        q_diab_jet=jet["Q_diabatic"],
        q_diab_delta=delta_q_diab,
        q_amplitude=q_amp,
        q_structure=q_structure,
        **{f"{name}_{var}": value[var] for name, value in effects.items() for var in ("u", "w")},
    )
    plot_pair(effects, ctrl["Q_diabatic"], jet["Q_diabatic"], r, z, jh, ch, out / f"{tag}_diabatic_validation.png")
    return rows, heating_rows


def plot_pair(effects, q_ctrl, q_jet, r, z, jh, ch, path):
    names = ["observed", "Q_diabatic", "Q_amplitude", "Q_structure", "non_diabatic_forcing", "all_forcing", "operator_remainder", "full_balanced"]
    fig, axes = plt.subplots(3, len(names), figsize=(2.35 * len(names), 8.2), sharex=True, sharey=True, constrained_layout=True)
    fields_by_row = [
        [q_jet - q_ctrl] + [np.full_like(q_ctrl, np.nan)] * (len(names) - 1),
        [effects[n]["u"] for n in names],
        [effects[n]["w"] for n in names],
    ]
    labels = ["Delta Qdiab (K h-1)", "Delta u (m s-1)", "Delta w (m s-1)"]
    fields_by_row[0][0] = fields_by_row[0][0] * 3600.0
    for row, (fields, label) in enumerate(zip(fields_by_row, labels)):
        valid = [f for f in fields if np.any(np.isfinite(f))]
        vmax = max(float(np.nanpercentile(np.abs(f), 98.5)) for f in valid)
        for col, (name, field) in enumerate(zip(names, fields)):
            ax = axes[row, col]
            if np.any(np.isfinite(field)):
                im = ax.pcolormesh(r, z, field, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
            else:
                ax.axis("off")
            if row == 0:
                ax.set_title(name.replace("_", "\n"), fontsize=8)
            if col == 0:
                ax.set_ylabel(label + "\nHeight (km)")
            if row == 2:
                ax.set_xlabel("Radius (km)")
        fig.colorbar(im, ax=[a for a in axes[row] if a.axison], fraction=0.015, pad=0.01)
    fig.suptitle(f"Direct-CM1 diabatic-heating validation: J{jh:g}/C{ch:g}")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def summary_plot(df: pd.DataFrame, out: Path):
    effects = ["Q_diabatic", "Q_amplitude", "Q_structure", "non_diabatic_forcing", "all_forcing", "operator_remainder", "full_balanced"]
    pairs = df[["jet_hour", "ctrl_hour"]].drop_duplicates()
    labels = [f"J{j:g}/C{c:g}" for j, c in zip(pairs.jet_hour, pairs.ctrl_hour)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    for ax, region in zip(axes, REGIONS):
        a = np.full((len(effects), len(labels)), np.nan)
        for i, effect in enumerate(effects):
            q = df[(df.region == region) & (df.effect == effect)]
            a[i] = q.projection_on_observed.to_numpy()
        lim = max(1.0, float(np.nanmax(np.abs(a))))
        im = ax.imshow(a, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.set_yticks(range(len(effects)), [x.replace("_", " ") for x in effects])
        ax.set_title(region.replace("_", " "))
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle("Projection of balanced responses on observed JET-CTRL circulation difference")
    fig.savefig(out / "diabatic_projection_on_observed_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(df: pd.DataFrame, heating: pd.DataFrame, out: Path):
    focus = df[df.effect.isin(["Q_diabatic", "Q_amplitude", "Q_structure", "all_forcing", "full_balanced"])]
    means = focus.groupby(["region", "effect"])[["projection_on_observed", "pattern_correlation_with_observed", "normalized_rmse_vs_observed"]].mean()
    lines = [
        "# Direct CM1 diabatic-heating validation",
        "",
        "## Method",
        "",
        "`Q_diabatic` is read from the available CM1 potential-temperature budget output (for these files, `ptb_mp + ptb_rad`, K s-1). The CTRL SE operator is fixed while only JET-minus-CTRL forcing fields are changed. Balanced responses are compared directly with observed azimuthal-mean JET-minus-CTRL radial and vertical winds.",
        "",
        "The amplitude/structure split is exact and symmetric. Structure contains radial/vertical displacement, width, tilt, and changes in cooling pattern; it is not a pure translation experiment.",
        "",
        "## Five-pair mean validation metrics",
        "",
        "```text",
        means.to_string(float_format=lambda x: f"{x:.3f}"),
        "```",
        "",
        "## Numerical closure",
        "",
        f"Maximum Q amplitude+structure closure: {df.q_split_closure.max():.3e}",
        f"Maximum response amplitude+structure closure: {df.response_split_closure.max():.3e}",
        f"Maximum diabatic+other forcing-response closure: {df.forcing_response_closure.max():.3e}",
        f"Maximum absolute/difference linearity closure: {df.absolute_linearity_closure.max():.3e}",
        "",
        "## Interpretation boundary",
        "",
        "A large projection on the observed circulation difference supports a balanced dynamical pathway from the direct CM1 heating tendency to the circulation. It does not by itself prove that the jet externally caused the heating change; convection and heating can co-evolve. Lead-lag and heating-budget analysis are required for that upstream causal statement.",
    ]
    (out / "DIABATIC_HEATING_VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    heating_rows = []
    for item in args.pairs:
        jh, ch = map(float, item.split(":"))
        a, b = solve_pair(jh, ch, args, out)
        rows.extend(a)
        heating_rows.extend(b)
    df = pd.DataFrame(rows)
    heating = pd.DataFrame(heating_rows)
    df.to_csv(out / "diabatic_validation_metrics.csv", index=False)
    heating.to_csv(out / "diabatic_heating_geometry.csv", index=False)
    summary_plot(df, out)
    write_report(df, heating, out)
    manifest = {
        "inputs": {"CTRL": args.nojet, "JET": args.jet},
        "pairs": args.pairs,
        "direct_cm1_terms": ["ptb_mp", "ptb_rad"],
        "units": "K s-1",
        "operator": "matched CTRL, regularized balanced projection",
        "eps_ratio": args.eps_ratio,
        "observed_target": "azimuthal-mean JET-minus-CTRL ur and w",
        "scope": "diagnoses heating-to-balanced-circulation pathway; does not alone establish jet-to-heating causality",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(df.groupby(["region", "effect"])[["projection_on_observed", "pattern_correlation_with_observed", "normalized_rmse_vs_observed"]].mean().to_string())


if __name__ == "__main__":
    main()
