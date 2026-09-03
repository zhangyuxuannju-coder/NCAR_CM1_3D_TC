#!/usr/bin/env python3
"""Summarize regularization sensitivity of matched operator-only SE solutions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def corr(a, b, mask, weight):
    good = mask & np.isfinite(a) & np.isfinite(b)
    w = weight[good]
    x, y = a[good], b[good]
    w = w / np.sum(w)
    xm, ym = np.sum(w*x), np.sum(w*y)
    den = np.sqrt(np.sum(w*(x-xm)**2)*np.sum(w*(y-ym)**2))
    return float(np.sum(w*(x-xm)*(y-ym))/den) if den > 0 else np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    root = Path(a.root)
    entries = [
        (1e-3, root/"matched_operator_SE_25N_JET30_9deg_J75_CTRL70_outer100_balanced"),
        (1e-4, root/"matched_operator_SE_25N_JET30_9deg_J75_CTRL70_outer100_eps1e-4"),
        (1e-5, root/"matched_operator_SE_25N_JET30_9deg_J75_CTRL70_outer100_eps1e-5"),
        (1e-6, root/"matched_operator_SE_25N_JET30_9deg_J75_CTRL70_outer100_eps1e-6"),
    ]
    rows = []
    products = {}
    for eps, path in entries:
        s = json.loads((path/"summary.json").read_text(encoding="utf-8"))
        n = np.load(path/"matched_operator_outer100_products.npz")
        products[eps] = {k: np.asarray(n[k]) for k in ("r_km", "z_km", "u_total", "w_total")}
        rows.append({
            "eps": eps,
            "changed": s["regularization"]["changed_coefficient_fraction"],
            "u_inner": s["comparison_with_actual_matched_cm1_difference"]["u"]["inner_0_100km"]["pred_rms"],
            "u_outflow": s["comparison_with_actual_matched_cm1_difference"]["u"]["inner_outflow"]["pred_rms"],
            "w_inner": s["comparison_with_actual_matched_cm1_difference"]["w"]["inner_0_100km"]["pred_rms"],
            "w_outflow": s["comparison_with_actual_matched_cm1_difference"]["w"]["inner_outflow"]["pred_rms"],
            "u_corr": s["comparison_with_actual_matched_cm1_difference"]["u"]["inner_outflow"]["correlation"],
            "w_corr": s["comparison_with_actual_matched_cm1_difference"]["w"]["inner_outflow"]["correlation"],
        })
    ref = products[1e-4]
    r, z = ref["r_km"], ref["z_km"]
    mask = ((r[None, :] <= 400) & (z[:, None] >= 8) & (z[:, None] <= 16))
    weight = np.broadcast_to(np.maximum(r[None, :], 0.5), ref["u_total"].shape)
    for row in rows:
        q = products[row["eps"]]
        row["u_pattern_corr"] = corr(q["u_total"], ref["u_total"], mask, weight)
        row["w_pattern_corr"] = corr(q["w_total"], ref["w_total"], mask, weight)

    eps = np.array([x["eps"] for x in rows])
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.5), constrained_layout=True)
    ax[0,0].semilogx(eps, [x["changed"]*100 for x in rows], "o-", color="#6F4E7C")
    ax[0,0].set_ylabel("Coefficient cells changed (%)")
    ax[0,0].set_title("Regularization footprint")
    ax[0,0].invert_xaxis()

    ax[0,1].loglog(eps, [x["u_inner"] for x in rows], "o-", label="u, r<100 km")
    ax[0,1].loglog(eps, [x["u_outflow"] for x in rows], "s-", label="u, r<300 km / 10-18 km")
    ax[0,1].set_ylabel("RMS radial-wind response (m s$^{-1}$)")
    ax[0,1].set_title("Response amplitude is not robust")
    ax[0,1].invert_xaxis(); ax[0,1].legend(frameon=False, fontsize=8)

    ax[1,0].loglog(eps, [x["w_inner"] for x in rows], "o-", label="w, r<100 km")
    ax[1,0].loglog(eps, [x["w_outflow"] for x in rows], "s-", label="w, r<300 km / 10-18 km")
    ax[1,0].set_ylabel("RMS vertical-velocity response (m s$^{-1}$)")
    ax[1,0].set_title("Vertical response sensitivity")
    ax[1,0].invert_xaxis(); ax[1,0].legend(frameon=False, fontsize=8)

    ax[1,1].semilogx(eps, [x["u_pattern_corr"] for x in rows], "o-", label="u pattern vs $10^{-4}$")
    ax[1,1].semilogx(eps, [x["w_pattern_corr"] for x in rows], "s-", label="w pattern vs $10^{-4}$")
    ax[1,1].axhline(0, color="0.4", lw=0.8)
    ax[1,1].set_ylim(-1.05, 1.05)
    ax[1,1].set_ylabel("Spatial correlation, r<400 km / 8-16 km")
    ax[1,1].set_title("Pattern robustness")
    ax[1,1].invert_xaxis(); ax[1,1].legend(frameon=False, fontsize=8)
    for q in ax.flat:
        q.set_xlabel(r"Ellipticity floor ratio $\epsilon$")
        q.grid(alpha=0.25, ls="--")
    fig.suptitle("Matched operator-only SE response: regularization sensitivity",
                 fontsize=14, fontweight="bold")
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
