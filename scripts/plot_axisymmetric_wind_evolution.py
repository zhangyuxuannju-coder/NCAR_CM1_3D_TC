#!/usr/bin/env python3
"""Plot time-evolving storm-centred axisymmetric CM1 wind structure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src._se_pipeline_single import PipelineConfig, azimuthal_average_from_3d


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nojet", required=True)
    p.add_argument("--jet", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--data-npz", default=None)
    p.add_argument("--summary-json", default=None)
    p.add_argument("--times-hours", type=float, nargs="+", default=[25.0, 55.0, 80.0])
    p.add_argument("--max-r-km", type=float, default=600.0)
    p.add_argument("--dr-km", type=float, default=12.0)
    p.add_argument("--max-z-km", type=float, default=20.0)
    p.add_argument("--f", type=float, default=5.464e-5)
    p.add_argument("--tangential-percentile", type=float, default=99.0)
    return p.parse_args()


def nice_up(value, step=5.0):
    return step * np.ceil(max(value, step) / step)


def main():
    a = parse_args()
    case_inputs = [("noJET", a.nojet), ("JET", a.jet)]
    panels = []
    for hour in a.times_hours:
        for case, input_file in case_inputs:
            cfg = PipelineConfig(
                input_file=input_file,
                output_dir=str(Path(a.output).parent),
                target_time_hours=hour,
                max_r_km=a.max_r_km,
                dr_km=a.dr_km,
                max_z_km=a.max_z_km,
                coriolis_f=a.f,
                include_model_budget_terms=False,
                write_netcdf=False,
                write_ieee=False,
                plot_solution=False,
            )
            avg = azimuthal_average_from_3d(cfg)
            panels.append({
                "case": case,
                "requested_hour": hour,
                "selected_hour": float(avg["time_seconds_used"][0] / 3600.0),
                "r_km": np.asarray(avg["r_km"], dtype=float),
                "z_km": np.asarray(avg["z_km"], dtype=float),
                "ut": np.asarray(avg["ut"], dtype=float),
                "ur": np.asarray(avg["ur"], dtype=float),
                "w": np.asarray(avg["w"], dtype=float),
                "rho": np.asarray(avg["rho"], dtype=float),
                "center_x_km": float(avg["center_x_km"][0]),
                "center_y_km": float(avg["center_y_km"][0]),
            })

    all_ut = np.concatenate([p["ut"].ravel() for p in panels])
    all_ut = all_ut[np.isfinite(all_ut)]
    negative = all_ut[all_ut < 0.0]
    positive = all_ut[all_ut > 0.0]
    vmin = -nice_up(abs(float(np.nanpercentile(negative, 1.0)))) if negative.size else -5.0
    vmax = nice_up(float(np.nanpercentile(positive, a.tangential_percentile))) if positive.size else 5.0
    vmin = max(vmin, -30.0)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list(
        "nature_tangential",
        ["#3C5488", "#8EA6C8", "#F7F7F4", "#F2C4A7", "#E64B35", "#8F1D20"],
        N=256,
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.linewidth": 0.75,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(len(a.times_hours), 2, figsize=(11.2, 3.65 * len(a.times_hours)),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_2d(axes)
    letters = "abcdefghijklmnopqrstuvwxyz"
    mappable = None
    radial_levels_negative = [-10.0, -5.0, -2.0]
    radial_levels_positive = [2.0, 5.0, 10.0]

    for i, (ax, p) in enumerate(zip(axes.flat, panels)):
        r, z = p["r_km"], p["z_km"]
        mappable = ax.pcolormesh(r, z, p["ut"], cmap=cmap, norm=norm,
                                 shading="auto", rasterized=True)
        ur_min, ur_max = float(np.nanmin(p["ur"])), float(np.nanmax(p["ur"]))
        neg = [x for x in radial_levels_negative if ur_min <= x <= ur_max]
        pos = [x for x in radial_levels_positive if ur_min <= x <= ur_max]
        if neg:
            csn = ax.contour(r, z, p["ur"], levels=neg, colors="#3C5488",
                             linestyles="--", linewidths=0.85)
            ax.clabel(csn, fmt="%g", fontsize=7, inline_spacing=2)
        if pos:
            csp = ax.contour(r, z, p["ur"], levels=pos, colors="#00A087",
                             linestyles="-", linewidths=0.9)
            ax.clabel(csp, fmt="%g", fontsize=7, inline_spacing=2)
        if np.nanmin(p["ut"]) <= 0.0 <= np.nanmax(p["ut"]):
            ax.contour(r, z, p["ut"], levels=[0.0], colors="#252525", linewidths=0.6)
        ax.text(-0.075, 1.02, f"({letters[i]})", transform=ax.transAxes,
                fontsize=11.5, fontweight="bold", va="bottom")
        ax.set_title(f"{p['case']}  ·  {p['selected_hour']:.0f} h",
                     fontsize=11, fontweight="bold", pad=6)
        ax.set_xlim(0.0, a.max_r_km)
        ax.set_ylim(0.0, a.max_z_km)
        ax.set_xticks(np.arange(0, a.max_r_km + 1, 100.0))
        ax.set_yticks(np.arange(0, a.max_z_km + 0.1, 2.5))
        for spine in ax.spines.values():
            spine.set_color("#4A4A4A")

    for ax in axes[-1, :]:
        ax.set_xlabel("Radius from TC center (km)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Height (km)")
    cbar = fig.colorbar(mappable, ax=axes, orientation="horizontal", shrink=0.68,
                        aspect=42, pad=0.035, extend="both")
    cbar.set_label(r"Azimuthal-mean tangential wind $\overline{v_\lambda}$ (m s$^{-1}$)")
    legend = [
        Line2D([0], [0], color="#00A087", lw=1.2, label="radial outflow (+), m s$^{-1}$"),
        Line2D([0], [0], color="#3C5488", lw=1.2, ls="--", label="radial inflow (−), m s$^{-1}$"),
    ]
    axes[0, 0].legend(handles=legend, loc="upper right", frameon=False,
                      fontsize=7.5, handlelength=2.2)
    fig.suptitle("Evolution of storm-centred axisymmetric wind structure",
                 fontsize=14, fontweight="bold")

    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if a.data_npz:
        arrays = {"r_km": panels[0]["r_km"], "z_km": panels[0]["z_km"]}
        for p in panels:
            tag = f"{p['case'].lower()}_{p['selected_hour']:.0f}h"
            arrays[f"ut_{tag}"] = p["ut"]
            arrays[f"ur_{tag}"] = p["ur"]
            arrays[f"w_{tag}"] = p["w"]
            arrays[f"rho_{tag}"] = p["rho"]
        np.savez_compressed(a.data_npz, **arrays)

    summary = {
        "tangential_color_limits_m_s": [float(vmin), float(vmax)],
        "radial_contours_m_s": radial_levels_negative + radial_levels_positive,
        "panels": [],
    }
    for p in panels:
        outflow_mask = (p["z_km"][:, None] >= 10.0) & (p["z_km"][:, None] <= 17.0)
        summary["panels"].append({
            "case": p["case"], "hour": p["selected_hour"],
            "center_x_km": p["center_x_km"], "center_y_km": p["center_y_km"],
            "max_tangential_wind_m_s": float(np.nanmax(p["ut"])),
            "max_upper_outflow_m_s": float(np.nanmax(np.where(outflow_mask, p["ur"], np.nan))),
            "min_upper_radial_wind_m_s": float(np.nanmin(np.where(outflow_mask, p["ur"], np.nan))),
        })
    summary_path = Path(a.summary_json) if a.summary_json else output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
