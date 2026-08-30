#!/usr/bin/env python3
"""Plot explicit CM1 absolute-angular-momentum budget closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


TERMS = ("mean_r", "mean_z", "eddy_r", "eddy_z", "pgrad", "diffusion", "rdamp")
CASES = ("noJET", "JET")
COLORS = {"noJET": "#0072B2", "JET": "#D55E00"}


def cmap():
    return LinearSegmentedColormap.from_list(
        "nature_balance", ["#214478", "#f7f7f3", "#b2182b"]
    )


def weighted_stats(obs, model, r_km, z_km):
    rr, zz = np.meshgrid(r_km, z_km)
    mask = (
        (rr >= 50.0) & (rr <= 350.0) &
        (zz >= 10.0) & (zz <= 16.0) &
        np.isfinite(obs) & np.isfinite(model)
    )
    w = np.maximum(rr, 0.5 * np.nanmedian(np.diff(r_km)))[mask]
    x, y = obs[mask], model[mask]
    w = w / np.sum(w)
    xm, ym = np.sum(w * x), np.sum(w * y)
    cov = np.sum(w * (x - xm) * (y - ym))
    den = np.sqrt(np.sum(w * (x - xm) ** 2) * np.sum(w * (y - ym) ** 2))
    corr = cov / den if den > 0 else np.nan
    rms_obs = np.sqrt(np.sum(w * x**2))
    rms_res = np.sqrt(np.sum(w * (x - y) ** 2))
    return mask, float(corr), float(rms_res / max(rms_obs, 1e-30))


def explicit_sum(z, case):
    fields = [z[f"{case}_reynolds_M_{name}"] for name in TERMS]
    return np.sum(np.stack(fields), axis=0)


def plot_maps(z, output):
    r = z["r_km"]
    h = z["z_km"]
    all_main, all_res = [], []
    data = {}
    for case in CASES:
        obs = z[f"{case}_reynolds_local_m"]
        model = explicit_sum(z, case)
        residual = obs - model
        mask, corr, ratio = weighted_stats(obs, model, r, h)
        data[case] = (obs, model, residual, mask, corr, ratio)
        all_main.extend([np.abs(obs[mask]), np.abs(model[mask])])
        all_res.append(np.abs(residual[mask]))
    main_lim = float(np.nanpercentile(np.concatenate(all_main), 98))
    res_lim = float(np.nanpercentile(np.concatenate(all_res), 98))

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    im_main = im_res = None
    for row, case in enumerate(CASES):
        obs, model, residual, mask, corr, ratio = data[case]
        for col, (field, title) in enumerate((
            (obs, r"Observed $\Delta M/\Delta t$"),
            (model, "Sum of all CM1 budget terms"),
        )):
            im_main = axes[row, col].pcolormesh(
                r, h, field, shading="auto", cmap=cmap(),
                vmin=-main_lim, vmax=main_lim,
            )
            axes[row, col].set_title(f"{case}: {title}")
        im_res = axes[row, 2].pcolormesh(
            r, h, residual, shading="auto", cmap=cmap(),
            vmin=-res_lim, vmax=res_lim,
        )
        axes[row, 2].set_title(f"{case}: observed − sum")
        x, y = obs[mask], model[mask]
        lim = float(np.nanpercentile(np.abs(np.concatenate([x, y])), 99))
        axes[row, 3].hexbin(x, y, gridsize=45, mincnt=1, cmap="Blues")
        axes[row, 3].plot([-lim, lim], [-lim, lim], "k--", lw=1.2)
        axes[row, 3].set_xlim(-lim, lim)
        axes[row, 3].set_ylim(-lim, lim)
        axes[row, 3].set_aspect("equal", adjustable="box")
        axes[row, 3].set_title(
            f"{case}: gridpoint comparison\n"
            f"weighted r = {corr:.3f}, RMS residual/observed = {ratio:.3f}"
        )
        axes[row, 3].set_xlabel(r"Observed (m$^2$ s$^{-2}$)")
        axes[row, 3].set_ylabel(r"Sum (m$^2$ s$^{-2}$)")
        for col in range(3):
            axes[row, col].set_xlim(50, 350)
            axes[row, col].set_ylim(10, 16)
            axes[row, col].set_xlabel("Radius (km)")
        axes[row, 0].set_ylabel("Height (km)")

    fig.colorbar(
        im_main, ax=axes[:, :2], orientation="horizontal", shrink=0.78, pad=0.08,
        label=r"Absolute-angular-momentum tendency (m$^2$ s$^{-2}$)",
    )
    fig.colorbar(
        im_res, ax=axes[:, 2], orientation="horizontal", shrink=0.86, pad=0.08,
        label=r"Closure residual (m$^2$ s$^{-2}$)",
    )
    fig.suptitle(
        "Explicit closure of the 4-h integrated CM1 angular-momentum budget at 72 h",
        fontsize=16, fontweight="bold",
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {c: {"correlation": data[c][4], "normalized_residual": data[c][5]} for c in CASES}


def plot_timeseries(csv_path, output):
    df = pd.read_csv(csv_path)
    df = df[(df["form"] == "reynolds") & (df["domain"] == "inner_outflow")].copy()
    sum_columns = [f"M_{name}_mean" for name in TERMS]
    df["explicit_sum_mean"] = df[sum_columns].sum(axis=1)
    df["algebra_error"] = (
        df["M_local_mean"] - df["explicit_sum_mean"] - df["M_residual_mean"]
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), sharex=True, constrained_layout=True)
    for col, case in enumerate(CASES):
        q = df[df["case"] == case].sort_values("hour")
        axes[0, col].plot(q["hour"], q["M_local_mean"], "o-", color="black", label="Observed")
        axes[0, col].plot(
            q["hour"], q["explicit_sum_mean"], "s--", color=COLORS[case],
            label="Sum of all terms",
        )
        axes[0, col].axhline(0, color="0.65", lw=0.8)
        axes[0, col].set_title(f"{case}: area-weighted mean tendency")
        axes[0, col].legend(frameon=False)
        axes[0, col].grid(alpha=0.2)
        axes[1, col].plot(
            q["hour"], q["M_closure_corr"], "o-", color=COLORS[case],
            label="Spatial correlation",
        )
        axes[1, col].plot(
            q["hour"], q["M_normalized_residual"], "s--", color="#7A5195",
            label="RMS residual / RMS observed",
        )
        axes[1, col].axhline(0.85, color="0.45", ls=":", lw=1, label="r = 0.85")
        axes[1, col].axhline(0.50, color="0.65", ls="--", lw=1, label="residual ratio = 0.5")
        axes[1, col].set_ylim(0, max(2.5, 1.05 * q["M_normalized_residual"].max()))
        axes[1, col].set_xlabel("Time (h)")
        axes[1, col].set_title(f"{case}: spatial closure quality")
        axes[1, col].grid(alpha=0.2)
        axes[1, col].legend(frameon=False, fontsize=8)
    axes[0, 0].set_ylabel(r"Mean M tendency (m$^2$ s$^{-2}$)")
    axes[1, 0].set_ylabel("Dimensionless metric")
    fig.suptitle(
        "CM1 budget sum versus observed angular-momentum tendency\n"
        "50–350 km, 10–16 km; 4-h integration",
        fontsize=16, fontweight="bold",
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--metrics", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(args.npz)

    reconstruction = {}
    for case in CASES:
        explicit = explicit_sum(z, case)
        saved = z[f"{case}_reynolds_model_m"]
        reconstruction[case] = float(np.nanmax(np.abs(explicit - saved)))

    map_metrics = plot_maps(z, out / "figure7_explicit_budget_closure_72h.png")
    timeseries = plot_timeseries(args.metrics, out / "figure8_budget_sum_timeseries.png")
    timeseries.to_csv(out / "explicit_budget_closure_timeseries.csv", index=False)
    summary = {
        "terms_in_sum": list(TERMS),
        "map_domain": "50-350 km, 10-16 km",
        "integration_window": "4 h",
        "hour_for_maps": 72,
        "explicit_sum_vs_saved_model_max_abs_difference": reconstruction,
        "closure_at_72h": map_metrics,
    }
    with open(out / "explicit_budget_closure_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
