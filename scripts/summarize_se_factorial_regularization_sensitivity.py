#!/usr/bin/env python3
"""Summarize full-domain SE attribution sensitivity to ellipticity floors."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"1e-3": "#3C5488", "1e-4": "#009E73", "1e-5": "#B9363E"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs=3, required=True, metavar=("EPS1E3", "EPS1E4", "EPS1E5"))
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def load(path):
    return json.loads((Path(path) / "full_domain_factorial_summary.json").read_text(encoding="utf-8"))


def main():
    a = parse_args(); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    labels = ("1e-3", "1e-4", "1e-5")
    runs = {label: load(path) for label, path in zip(labels, a.runs)}
    records = []
    for label, run in runs.items():
        for result in run["results"]:
            for region, block in result["radial_wind_attribution"].items():
                rms = block["cylindrical_r_weighted_rms_m_s"]
                shares = block["projection_share_on_total"]
                records.append({
                    "eps": label, "hour": result["hour"], "region": region,
                    "forcing_rms": rms["forcing"], "operator_rms": rms["operator"],
                    "interaction_rms": rms["interaction"], "total_rms": rms["total"],
                    "forcing_operator_ratio": block["forcing_to_operator_rms_ratio"],
                    "forcing_projection": shares["forcing"],
                    "operator_projection": shares["operator"],
                    "interaction_projection": shares["interaction"],
                    "ctrl_modified_fraction": result["regularization"]["CTRL"]["changed_coefficient_fraction"],
                    "jet_modified_fraction": result["regularization"]["JET"]["changed_coefficient_fraction"],
                })
    (out / "regularization_sensitivity_records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.3), constrained_layout=True)
    for label in labels:
        color = COLORS[label]
        for ax, region, title in [
            (axes[0,0], "full", "Full domain"),
            (axes[0,1], "upper_outflow_union", "Upper-outflow union"),
        ]:
            selected = [x for x in records if x["eps"] == label and x["region"] == region]
            hours = [x["hour"] for x in selected]
            ratios = [x["forcing_operator_ratio"] for x in selected]
            ax.plot(hours, ratios, marker="o", lw=1.7, color=color, label=rf"$\epsilon={label}$")
            ax.axhline(1.0, color="0.35", lw=0.8, ls="--")
            ax.set_yscale("log")
            ax.set_xlabel("Time (h)")
            ax.set_ylabel("RMS forcing / RMS operator")
            ax.set_title(title)
            ax.grid(alpha=0.2)
    axes[0,0].legend(frameon=False)

    for label in labels:
        selected = [x for x in records if x["eps"] == label and x["region"] == "full"]
        hours = [x["hour"] for x in selected]
        values = [x["interaction_projection"] for x in selected]
        axes[1,0].plot(hours, values, marker="o", lw=1.7, color=COLORS[label], label=rf"$\epsilon={label}$")
    axes[1,0].axhline(0.0, color="0.35", lw=0.8)
    axes[1,0].axhline(1.0, color="0.55", lw=0.7, ls=":")
    axes[1,0].set_xlabel("Time (h)")
    axes[1,0].set_ylabel("Interaction projection on total")
    axes[1,0].set_title("Operator–forcing interaction")
    axes[1,0].grid(alpha=0.2)

    for label in labels:
        selected = [x for x in records if x["eps"] == label and x["region"] == "full"]
        hours = [x["hour"] for x in selected]
        ctrl = [x["ctrl_modified_fraction"] for x in selected]
        jet = [x["jet_modified_fraction"] for x in selected]
        axes[1,1].plot(hours, ctrl, marker="o", lw=1.5, color=COLORS[label], label=rf"CTRL $\epsilon={label}$")
        axes[1,1].plot(hours, jet, marker="s", lw=1.0, ls="--", color=COLORS[label], label=rf"JET $\epsilon={label}$")
    axes[1,1].set_xlabel("Time (h)")
    axes[1,1].set_ylabel("Modified grid-point fraction")
    axes[1,1].set_title("Regularization footprint")
    axes[1,1].grid(alpha=0.2)
    axes[1,1].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Sensitivity of full-domain SE attribution to the ellipticity floor", fontweight="bold")
    fig.savefig(out / "figure_regularization_sensitivity.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    robust = {}
    for region in ("full", "upper_10_18km", "upper_outflow_union"):
        robust[region] = {}
        hours = sorted({x["hour"] for x in records if x["region"] == region})
        for hour in hours:
            vals = [x["forcing_operator_ratio"] for x in records if x["region"] == region and x["hour"] == hour]
            if min(vals) > 1.2:
                decision = "forcing robustly larger"
            elif max(vals) < 1.0/1.2:
                decision = "operator robustly larger"
            elif min(vals) >= 1.0/1.2 and max(vals) <= 1.2:
                decision = "comparable"
            else:
                decision = "regularization-sensitive"
            robust[region][str(hour)] = {"ratio_min": min(vals), "ratio_max": max(vals), "decision": decision}
    (out / "robust_attribution_decisions.json").write_text(json.dumps(robust, indent=2), encoding="utf-8")
    print(json.dumps(robust, indent=2))

if __name__ == "__main__":
    main()
