#!/usr/bin/env python3
"""Plot late-time equivalent operator forcing with shared scales."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np


CMAP = LinearSegmentedColormap.from_list(
    "operator_force",
    ["#24486E", "#6F98BE", "#C8D9E7", "#F7F7F4", "#F4C6AF", "#D96B4C", "#8E2C3A"],
    N=256,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", required=True)
    p.add_argument("--hours", type=int, nargs="+", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def wrms(field: np.ndarray, r_km: np.ndarray, z_km: np.ndarray, mask: np.ndarray) -> float:
    r_m = r_km * 1000.0
    z_m = z_km * 1000.0
    weight = (
        np.maximum(r_m, 1.0)[None, :]
        * np.abs(np.gradient(r_m))[None, :]
        * np.abs(np.gradient(z_m))[:, None]
    )
    good = mask & np.isfinite(field)
    return float(np.sqrt(np.sum(weight[good] * field[good] ** 2) / np.sum(weight[good])))


def main() -> None:
    a = parse_args()
    root = Path(a.input_root)
    records = []
    for hour in a.hours:
        with np.load(root / f"t{hour:03d}h" / "operator_perturbation_products.npz") as d:
            records.append(
                {
                    "hour": hour,
                    "r": np.asarray(d["r_km"], float),
                    "z": np.asarray(d["z_km"], float),
                    "s": np.asarray(d["S_total"], float),
                }
            )

    full_values, inner_values = [], []
    for rec in records:
        rr, zz = np.meshgrid(rec["r"], rec["z"])
        full = (rr >= 30) & (rr <= 1200) & (zz >= 0.5) & (zz <= 18)
        inner = (rr >= 20) & (rr <= 350) & (zz >= 0.5) & (zz <= 18)
        full_values.append(np.abs(rec["s"][full]))
        inner_values.append(np.abs(rec["s"][inner]))
    full_lim = max(float(np.nanpercentile(np.concatenate(full_values), 99.0)), 1e-30)
    inner_lim = max(float(np.nanpercentile(np.concatenate(inner_values), 99.0)), 1e-30)
    scale = 1e-13

    fig, axes = plt.subplots(len(records), 2, figsize=(13.2, 3.25 * len(records)),
                             sharey=True, constrained_layout=True)
    if len(records) == 1:
        axes = axes[None, :]
    maps = []
    for i, rec in enumerate(records):
        rr, zz = np.meshgrid(rec["r"], rec["z"])
        domains = [
            ((0, 1200), full_lim, (rr >= 30) & (rr <= 1200) & (zz >= 0.5) & (zz <= 18)),
            ((0, 350), inner_lim, (rr >= 50) & (rr <= 350) & (zz >= 10) & (zz <= 16)),
        ]
        for j, (xlim, lim, metric_mask) in enumerate(domains):
            ax = axes[i, j]
            m = ax.pcolormesh(
                rec["r"], rec["z"], rec["s"] / scale,
                cmap=CMAP,
                norm=TwoSlopeNorm(vmin=-lim / scale, vcenter=0.0, vmax=lim / scale),
                shading="auto", rasterized=True,
            )
            maps.append(m)
            if np.nanmin(rec["s"]) < 0 < np.nanmax(rec["s"]):
                ax.contour(rec["r"], rec["z"], rec["s"], levels=[0], colors="0.25", linewidths=0.45)
            ax.set(xlim=xlim, ylim=(0, 18))
            if j == 0:
                ax.plot(888, 12, "*", ms=8, color="#7A1FA2", mec="white", mew=0.5)
                ax.set_ylabel(f"{rec['hour']} h\nHeight (km)")
            rms = wrms(rec["s"], rec["r"], rec["z"], metric_mask)
            ax.text(
                0.98, 0.95, f"RMS={rms:.2e}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5,
                bbox={"facecolor": "white", "alpha": 0.76, "edgecolor": "none", "pad": 1.3},
            )
            if i == 0:
                ax.set_title("Full jet–TC domain" if j == 0 else "Inner 350 km")
            if i == len(records) - 1:
                ax.set_xlabel("Radius from TC centre (km)")

    for j, lim in enumerate((full_lim, inner_lim)):
        cb = fig.colorbar(maps[j], ax=axes[:, j], orientation="horizontal", shrink=0.78, pad=0.025)
        cb.set_label(
            rf"$S_{{op}}^{{eq}}$ ($10^{{-13}}$ K$^{{-1}}$ s$^{{-3}}$); shared P99 = {lim/scale:.2g}"
        )
    fig.suptitle(
        r"JET15 equivalent operator forcing: $S_{op}^{eq}=-\Delta\mathcal{L}\,\psi_{CTRL}$",
        fontsize=14, fontweight="bold",
    )
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
