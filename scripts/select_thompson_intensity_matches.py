#!/usr/bin/env python3
"""Select auditable pre-150 h JET--CTRL matches in intensity and storm size."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", default="output/thompson_240h_se_attribution/baseline/hourly_intensity_baseline.csv")
    p.add_argument("--output-dir", default="output/thompson_240h_se_attribution/matching")
    p.add_argument("--jet-hours", type=float, nargs="+", default=[72, 84, 96, 108])
    p.add_argument("--v-tolerance", type=float, default=2.0)
    p.add_argument("--rmw-tolerance", type=float, default=6.0)
    p.add_argument("--pressure-tolerance", type=float, default=5.0)
    p.add_argument("--growth-tolerance", type=float, default=0.5)
    return p.parse_args()


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    a = args(); out = ROOT / a.output_dir; out.mkdir(parents=True, exist_ok=True)
    with (ROOT / a.baseline).open(encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    ctrl = [x for x in rows if x["case"] == "CTRL" and 38 <= f(x, "time_h") <= 150]
    jet = {round(f(x, "time_h"), 6): x for x in rows if x["case"] == "JET"}
    records = []
    for jh in a.jet_hours:
        j = jet.get(round(jh, 6))
        if j is None:
            raise KeyError(f"JET {jh:g} h is absent from the hourly baseline")
        candidates = []
        for c in ctrl:
            dv = abs(f(j, "vt0_axisymmetric_max_m_s") - f(c, "vt0_axisymmetric_max_m_s"))
            dr = abs(f(j, "rmw0_km") - f(c, "rmw0_km"))
            dp = abs(f(j, "psfc_center_hpa") - f(c, "psfc_center_hpa"))
            dg = abs(f(j, "vt0_axisymmetric_max_m_s_tendency_6h") - f(c, "vt0_axisymmetric_max_m_s_tendency_6h"))
            accepted = (dv <= a.v_tolerance and dr <= a.rmw_tolerance and dp <= a.pressure_tolerance
                        and f(j, "vt0_axisymmetric_max_m_s_tendency_6h") > 0
                        and f(c, "vt0_axisymmetric_max_m_s_tendency_6h") > 0 and dg <= a.growth_tolerance)
            if accepted:
                score = dv/a.v_tolerance + dr/a.rmw_tolerance + dp/a.pressure_tolerance + dg/a.growth_tolerance
                candidates.append((score, c, dv, dr, dp, dg))
        if not candidates:
            records.append({"jet_hour": jh, "accepted": False, "reason": "no CTRL state meets all predeclared thresholds"})
            continue
        score, c, dv, dr, dp, dg = min(candidates, key=lambda x: x[0])
        records.append({
            "jet_hour": jh, "ctrl_hour": f(c, "time_h"), "accepted": True, "score": score,
            "delta_vt_ms": f(j, "vt0_axisymmetric_max_m_s") - f(c, "vt0_axisymmetric_max_m_s"),
            "abs_delta_vt_ms": dv, "delta_rmw_km": f(j, "rmw0_km") - f(c, "rmw0_km"), "abs_delta_rmw_km": dr,
            "delta_pressure_hpa": f(j, "psfc_center_hpa") - f(c, "psfc_center_hpa"), "abs_delta_pressure_hpa": dp,
            "delta_growth_6h_ms_per_h": f(j, "vt0_axisymmetric_max_m_s_tendency_6h") - f(c, "vt0_axisymmetric_max_m_s_tendency_6h"),
            "jet_vt_ms": f(j, "vt0_axisymmetric_max_m_s"), "ctrl_vt_ms": f(c, "vt0_axisymmetric_max_m_s"),
            "jet_rmw_km": f(j, "rmw0_km"), "ctrl_rmw_km": f(c, "rmw0_km"),
            "jet_pressure_hpa": f(j, "psfc_center_hpa"), "ctrl_pressure_hpa": f(c, "psfc_center_hpa"),
            "jet_growth_6h_ms_per_h": f(j, "vt0_axisymmetric_max_m_s_tendency_6h"), "ctrl_growth_6h_ms_per_h": f(c, "vt0_axisymmetric_max_m_s_tendency_6h"),
        })
    accepted = [x for x in records if x["accepted"]]
    with (out / "strength_matched_pairs.json").open("w", encoding="utf-8") as h:
        json.dump({"criteria": vars(a), "records": records}, h, ensure_ascii=False, indent=2)
    if accepted:
        with (out / "strength_matched_pairs.csv").open("w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=list(accepted[0])); w.writeheader(); w.writerows(accepted)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for case, color in (("CTRL", "#1f77b4"), ("JET", "#b62f4a")):
        q = [x for x in rows if x["case"] == case and 38 <= f(x, "time_h") <= 150]
        ax[0].plot([f(x,"time_h") for x in q], [f(x,"vt0_axisymmetric_max_m_s") for x in q], color=color, label=case)
    for row in accepted:
        ax[0].plot([row["ctrl_hour"], row["jet_hour"]], [row["ctrl_vt_ms"], row["jet_vt_ms"]], color="0.35", lw=1)
        ax[0].scatter([row["ctrl_hour"], row["jet_hour"]], [row["ctrl_vt_ms"], row["jet_vt_ms"]], c=["#1f77b4", "#b62f4a"], zorder=3)
    ax[0].set(xlabel="Time (h)", ylabel="lowest-level axisymmetric Vt max (m s$^{-1}$)", title="Pre-150 h strength matches")
    ax[0].legend(frameon=False); ax[0].grid(alpha=.2)
    labels = [f"J{r['jet_hour']:g}/C{r['ctrl_hour']:g}" for r in accepted]
    x = np.arange(len(accepted)); width = .25
    for offset, key, name, color in ((-width, "abs_delta_vt_ms", "|ΔVt| (m s$^{-1}$)", "#1f77b4"), (0, "abs_delta_rmw_km", "|ΔRMW| (km)", "#ff7f0e"), (width, "abs_delta_pressure_hpa", "|Δp| (hPa)", "#2ca02c")):
        ax[1].bar(x+offset, [r[key] for r in accepted], width, label=name, color=color)
    ax[1].set(xticks=x, xticklabels=labels, ylabel="residual mismatch", title="Match residuals")
    ax[1].legend(frameon=False, fontsize=8); ax[1].grid(axis="y", alpha=.2)
    fig.savefig(out / "strength_matched_pairs.png", dpi=200)
    plt.close(fig)
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
