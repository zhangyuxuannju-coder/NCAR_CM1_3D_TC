#!/usr/bin/env python3
"""Plot grouped CM1 potential-temperature budget terms without closure testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Existing theta_budget_fields.npz")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window", default="77–83 h")
    return parser.parse_args()


def symmetric_limit(values: list[np.ndarray], percentile: float = 99.0) -> float:
    finite = np.concatenate([np.abs(value[np.isfinite(value)]) for value in values])
    return max(float(np.nanpercentile(finite, percentile)) * 1.05, 1.0e-12)


def grouped_fields(z: np.lib.npyio.NpzFile) -> dict[str, dict[str, np.ndarray]]:
    fields: dict[str, dict[str, np.ndarray]] = {}
    for case in ("CTRL", "JET30"):
        prefix = f"{case}_CM1_"
        fields[case] = {
            "ADV": z[prefix + "HADV"] + z[prefix + "VADV"],
            "TURB_DIFF": (z[prefix + "HTURB"] + z[prefix + "VTURB"] +
                          z[prefix + "HIDIFF"] + z[prefix + "VIDIFF"]),
            "MOIST": z[prefix + "MP"] + z[prefix + "DIV"],
            "RADIATION": z[prefix + "RAD"],
            "RDAMP": z[prefix + "RDAMP"],
        }
        fields[case]["CM1_SUM"] = sum(fields[case].values(), np.zeros_like(fields[case]["ADV"]))
    fields["DELTA"] = {
        key: fields["JET30"][key] - fields["CTRL"][key]
        for key in fields["CTRL"]
    }
    return fields


def plot(fields: dict[str, dict[str, np.ndarray]], radius_km: np.ndarray, height_km: np.ndarray,
         window: str, output: Path) -> None:
    rows = [
        ("ADV", "ADV = CM1 HADV + VADV"),
        ("TURB_DIFF", "TURB + DIFF"),
        ("MOIST", "MOIST = MP + moist divergence"),
        ("RADIATION", "RADIATION"),
        ("CM1_SUM", "Sum of all available CM1 terms"),
    ]
    cases = ("CTRL", "JET30", "DELTA")
    fig, axes = plt.subplots(len(rows), 3, figsize=(14, 20), sharex=True, sharey=True,
                             constrained_layout=True)
    for row, (key, label) in enumerate(rows):
        main_lim = symmetric_limit([fields["CTRL"][key], fields["JET30"][key]]) * 1.0e3
        delta_lim = symmetric_limit([fields["DELTA"][key]]) * 1.0e3
        for col, case in enumerate(cases):
            value = fields[case][key] * 1.0e3
            lim = main_lim if case != "DELTA" else delta_lim
            image = axes[row, col].pcolormesh(
                radius_km, height_km, value, cmap="RdBu_r",
                norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim), shading="auto"
            )
            try:
                axes[row, col].contour(radius_km, height_km, value, levels=[0.0],
                                       colors="k", linewidths=0.35)
            except ValueError:
                pass
            axes[row, col].set(xlim=(0, 300), ylim=(0, 18))
            if row == 0:
                axes[row, col].set_title(("CTRL", "JET30", "JET30 − CTRL")[col])
            if col == 0:
                axes[row, col].set_ylabel(label + "\nHeight (km)")
            if row == len(rows) - 1:
                axes[row, col].set_xlabel("Radial distance (km)")
            fig.colorbar(image, ax=axes[row, col], fraction=0.046, pad=0.02,
                         label=r"$10^{-3}$ K s$^{-1}$")
    fig.suptitle(f"CM1 potential-temperature budget terms, {window} mean", fontsize=16)
    fig.savefig(output, dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(input_path) as z:
        radius_km = np.asarray(z["radius_km"], dtype=float)
        height_km = np.asarray(z["height_km"], dtype=float)
        fields = grouped_fields(z)
        np.savez_compressed(
            output / "theta_budget_cm1_terms.npz",
            radius_km=radius_km,
            height_km=height_km,
            **{f"{case}_{key}": value
               for case in ("CTRL", "JET30", "DELTA")
               for key, value in fields[case].items()},
        )
    plot(fields, radius_km, height_km, args.window, output / "theta_budget_cm1_terms_radial_height.png")
    summary = {
        "status": "completed_cm1_terms_only",
        "source_npz": str(input_path),
        "window": args.window,
        "terms": {
            "ADV": ["CM1_HADV", "CM1_VADV"],
            "TURB_DIFF": ["CM1_HTURB", "CM1_VTURB", "CM1_HIDIFF", "CM1_VIDIFF"],
            "MOIST": ["CM1_MP", "CM1_DIV"],
            "RADIATION": ["CM1_RAD"],
            "RDAMP": ["CM1_RDAMP"],
            "CM1_SUM": "ADV + TURB_DIFF + MOIST + RADIATION + RDAMP",
        },
        "units": "10^-3 K s^-1 in figures; source fields are K s^-1",
        "closure_check": False,
        "interpretation": "CM1 reported budget-term structure only; CM1_SUM is not independently validated total theta tendency.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    readme = (
        f"# CM1 potential-temperature budget terms, {args.window}\n\n"
        "This product uses the already averaged CM1 `ptb_*` output fields. "
        "It does not calculate an independent total tendency and does not perform a closure test.\n\n"
        "Grouped terms are `ADV = CM1_HADV + CM1_VADV`, `TURB_DIFF = CM1_HTURB + CM1_VTURB + "
        "CM1_HIDIFF + CM1_VIDIFF`, `MOIST = CM1_MP + CM1_DIV`, and `RADIATION = CM1_RAD`. "
        "`CM1_SUM` also includes `CM1_RDAMP`; RDAMP is exactly zero in this dataset and is not relabelled as DISS.\n\n"
        "`CM1_SUM` should be interpreted as a sum of reported output terms, not as an independently "
        "validated total potential-temperature tendency.\n\n"
        "The figure uses the existing 3-km radial grid, native heights through 18 km, 360-azimuth average, "
        "and zero-centred diverging colour scales.\n"
    )
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
