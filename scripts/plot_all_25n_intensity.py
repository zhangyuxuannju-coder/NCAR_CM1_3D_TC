#!/usr/bin/env python3
"""Plot minimum surface pressure for every 25N CM1 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
from scipy.ndimage import gaussian_filter


CASES = [
    ("cm1out_25N_nojet.nc", "CTRL", "#222222", 2.6, "-"),
    ("cm1out_25N_5o_jet.nc", "JET 5°", "#7B61A8", 2.0, "-"),
    ("cm1out_25N_8o_jet_15.nc", "JET15 8°", "#2A6FBB", 2.0, "-"),
    ("cm1out_25N_9o_jet_15.nc", "JET15 9°", "#008C95", 2.2, "-"),
    ("cm1out_25N_8o_jet_30.nc", "JET30 8°", "#D1495B", 2.0, "-"),
    ("cm1out_25N_9o_jet_30.nc", "JET30 9°", "#E68A2E", 2.2, "-"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/data/zhangyx/DATA")
    p.add_argument("--output", required=True)
    p.add_argument("--smooth-sigma", type=float, default=2.0)
    return p.parse_args()


def read_intensity(path: Path, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    with Dataset(path) as ds:
        time_h = np.asarray(ds.variables["time"][:], float) / 3600.0
        psfc_var = ds.variables["psfc"]
        pmin = np.empty(time_h.size, dtype=float)
        for it in range(time_h.size):
            field = np.asarray(psfc_var[it], float)
            if sigma > 0:
                field = gaussian_filter(field, sigma=sigma)
            pmin[it] = np.nanmin(field)
    if np.nanmedian(pmin) > 10000.0:
        pmin /= 100.0
    return time_h, pmin


def main() -> None:
    a = parse_args()
    data_dir = Path(a.data_dir)
    fig, ax = plt.subplots(figsize=(12.4, 6.6))
    summaries = []
    for filename, label, color, width, linestyle in CASES:
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        time_h, pmin = read_intensity(path, a.smooth_sigma)
        summaries.append((label, float(time_h[-1]), float(np.nanmin(pmin))))
        ax.plot(time_h, pmin, label=label, color=color, lw=width, ls=linestyle)
        print(f"{label}: nt={len(time_h)}, end={time_h[-1]:.1f} h, minimum={np.nanmin(pmin):.2f} hPa")

    ax.set_title("25°N CM1 Experiments: TC Intensity Evolution", fontsize=16, fontweight="bold")
    ax.set_xlabel("Time (h)", fontsize=13)
    ax.set_ylabel("Minimum Surface Pressure (hPa)", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.32)
    ax.legend(loc="best", frameon=True, fontsize=10.5)
    ax.tick_params(labelsize=11)
    ax.margins(x=0.01)
    fig.tight_layout()
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved: {output}")


if __name__ == "__main__":
    main()
