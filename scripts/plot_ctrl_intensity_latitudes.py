#!/usr/bin/env python3
"""Compare minimum surface pressure among the available CTRL simulations."""

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
    ("cm1out_22N_nojet.nc", "CTRL 22°N", "#2A6FBB"),
    ("cm1out_25N_nojet.nc", "CTRL 25°N", "#222222"),
    ("cm1out_27N_nojet.nc", "CTRL 27°N", "#D1495B"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/zhangyx/DATA")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smooth-sigma", type=float, default=2.0)
    return parser.parse_args()


def read_intensity(path: Path, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    with Dataset(path) as dataset:
        time_h = np.asarray(dataset.variables["time"][:], dtype=float) / 3600.0
        psfc = dataset.variables["psfc"]
        minimum_pressure = np.empty(time_h.size, dtype=float)
        for time_index in range(time_h.size):
            field = np.asarray(psfc[time_index], dtype=float)
            if sigma > 0:
                field = gaussian_filter(field, sigma=sigma)
            minimum_pressure[time_index] = np.nanmin(field)

    if np.nanmedian(minimum_pressure) > 10000.0:
        minimum_pressure /= 100.0
    return time_h, minimum_pressure


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    fig, axis = plt.subplots(figsize=(11.8, 6.5))
    for filename, label, color in CASES:
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        time_h, minimum_pressure = read_intensity(path, args.smooth_sigma)
        axis.plot(time_h, minimum_pressure, color=color, linewidth=2.5, label=label)
        print(
            f"{label}: nt={time_h.size}, end={time_h[-1]:.1f} h, "
            f"minimum={np.nanmin(minimum_pressure):.2f} hPa"
        )

    axis.set_title("CTRL Experiments: TC Intensity Evolution", fontsize=16, fontweight="bold")
    axis.set_xlabel("Time (h)", fontsize=13)
    axis.set_ylabel("Minimum Surface Pressure (hPa)", fontsize=13)
    axis.grid(True, linestyle="--", alpha=0.32)
    axis.legend(loc="best", frameon=True, fontsize=11)
    axis.tick_params(labelsize=11)
    axis.margins(x=0.01)
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved: {output}")


if __name__ == "__main__":
    main()
