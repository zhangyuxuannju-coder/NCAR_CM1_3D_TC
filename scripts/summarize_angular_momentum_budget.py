#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COL = {"mean": "#0072B2", "eddy": "#D55E00", "pgrad": "#CC79A7",
       "diffusion": "#009E73", "residual": "#6F6F6F", "observed": "#111111"}


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def classify(corr, residual):
    if corr >= 0.85 and residual <= 0.50:
        return "PASS"
    if corr >= 0.60 and residual <= 1.00:
        return "CAUTION"
    return "FAIL"


def delta_series(df, key, hours):
    out = []
    for h in hours:
        j = df[(df.case == "JET") & (df.hour == h)].iloc[0]
        c = df[(df.case == "noJET") & (df.hour == h)].iloc[0]
        out.append(float(j[key] - c[key]))
    return np.asarray(out)


def integrate(df, key, hours):
    y = delta_series(df, key, hours)
    return float(np.trapezoid(y, x=np.asarray(hours) * 3600.0))


def main():
    a = args(); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(a.input)
    d = raw[(raw.domain == "inner_outflow") & (raw.form == "reynolds")].copy()
    d["hour"] = d.hour.astype(float)
    hours = sorted(d.hour.unique())
    d["quality"] = [classify(c, r) for c, r in zip(d.M_closure_corr, d.M_normalized_residual)]

    fig, ax = plt.subplots(2, 2, figsize=(13.5, 8.5), sharex=True)
    for case, color in (("noJET", "#0072B2"), ("JET", "#D55E00")):
        q = d[d.case == case].sort_values("hour")
        ax[0, 0].plot(q.hour, q.M_closure_corr, "o-", color=color, label=case)
        ax[0, 1].plot(q.hour, q.M_normalized_residual, "o-", color=color, label=case)
        ax[1, 0].plot(q.hour, q.eddy_direct_r_corr, "o-", color=color, label=f"{case} radial")
        ax[1, 0].plot(q.hour, q.eddy_direct_z_corr, "s--", color=color, label=f"{case} vertical")
    fav = raw[(raw.domain == "inner_outflow") & (raw.form == "favre")].copy()
    fav["hour"] = fav.hour.astype(float)
    for case, color in (("noJET", "#0072B2"), ("JET", "#D55E00")):
        r = d[d.case == case].sort_values("hour")
        f = fav[fav.case == case].sort_values("hour")
        ax[1, 1].plot(r.hour, f.M_normalized_residual.values-r.M_normalized_residual.values,
                      "o-", color=color, label=case)
    ax[0, 0].axhline(.85, color="0.5", ls="--"); ax[0, 0].set_ylabel("Spatial correlation")
    ax[0, 0].set_title("CM1 M-budget pattern closure")
    ax[0, 1].axhline(.5, color="0.5", ls="--"); ax[0, 1].axhline(1, color="0.7", ls=":")
    ax[0, 1].set_ylabel("RMS residual / RMS observed"); ax[0, 1].set_title("Closure-amplitude error")
    ax[1, 0].axhline(0, color="0.5", lw=.8); ax[1, 0].set_ylabel("Spatial correlation")
    ax[1, 0].set_title("CM1 eddy residual vs direct 3-D flux")
    ax[1, 1].axhline(0, color="0.5", lw=.8); ax[1, 1].set_ylabel("Favre − Reynolds residual ratio")
    ax[1, 1].set_title("Density-weighting sensitivity")
    for x in ax.flat:
        x.set_xlabel("Time (h)"); x.grid(alpha=.2); x.legend(frameon=False, fontsize=8)
    fig.suptitle("Angular-momentum budget validation in 50–350 km, 10–16 km",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(out/"figure5_budget_quality.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    eval_hours = [45., 50., 55., 60., 65., 70., 75.]
    mean = delta_series(d, "M_mean_r_mean", eval_hours) + delta_series(d, "M_mean_z_mean", eval_hours)
    eddy = delta_series(d, "M_eddy_r_mean", eval_hours) + delta_series(d, "M_eddy_z_mean", eval_hours)
    pgrad = delta_series(d, "M_pgrad_mean", eval_hours)
    diffusion = delta_series(d, "M_diffusion_mean", eval_hours)
    residual = delta_series(d, "M_residual_mean", eval_hours)
    observed = delta_series(d, "M_local_mean", eval_hours)
    phases = {"45–55": [45., 50., 55.], "55–65": [55., 60., 65.],
              "65–75": [65., 70., 75.], "45–70 robust": [45., 50., 55., 60., 65., 70.],
              "45–75": eval_hours}
    ikeys = {"mean": ("M_mean_r_mean", "M_mean_z_mean"),
             "eddy": ("M_eddy_r_mean", "M_eddy_z_mean"),
             "pgrad": ("M_pgrad_mean",), "diffusion": ("M_diffusion_mean",),
             "residual": ("M_residual_mean",), "observed": ("M_local_mean",)}
    integ = {}
    for phase, hs in phases.items():
        integ[phase] = {name: sum(integrate(d, k, hs) for k in keys)
                        for name, keys in ikeys.items()}

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    for name, vals in (("observed", observed), ("mean", mean), ("eddy", eddy),
                       ("pgrad", pgrad), ("residual", residual)):
        ax[0].plot(eval_hours, vals, "o-", color=COL[name], label=name)
    ax[0].axhline(0, color="0.5", lw=.8); ax[0].set_xlabel("Time (h)")
    ax[0].set_ylabel(r"JET−noJET M tendency (m$^2$ s$^{-2}$)")
    ax[0].set_title("Time evolution"); ax[0].legend(frameon=False)
    phase_names = list(phases)
    x = np.arange(len(phase_names)); width=.16
    for i, name in enumerate(("mean", "eddy", "pgrad", "residual")):
        ax[1].bar(x+(i-1.5)*width, [integ[p][name]/1e5 for p in phase_names], width,
                  color=COL[name], label=name)
    ax[1].plot(x, [integ[p]["observed"]/1e5 for p in phase_names], "kD", label="observed")
    ax[1].axhline(0, color="0.5", lw=.8); ax[1].set_xticks(x, phase_names, rotation=18)
    ax[1].set_ylabel(r"Cumulative ΔM ($10^5$ m$^2$ s$^{-1}$)")
    ax[1].set_title("Phase-integrated pathways"); ax[1].legend(frameon=False, fontsize=8)
    for i, phase in enumerate(phase_names[:3]):
        radial = integrate(d, "M_eddy_r_mean", phases[phase]) / 1e5
        vertical = integrate(d, "M_eddy_z_mean", phases[phase]) / 1e5
        ax[2].bar(i-.18, radial, .36, color="#D55E00", label="radial" if i == 0 else None)
        ax[2].bar(i+.18, vertical, .36, color="#E69F00", label="vertical" if i == 0 else None)
    ax[2].axhline(0, color="0.5", lw=.8); ax[2].set_xticks(range(3), phase_names[:3])
    ax[2].set_ylabel(r"Cumulative eddy ΔM ($10^5$ m$^2$ s$^{-1}$)")
    ax[2].set_title("Radial versus vertical eddy transport"); ax[2].legend(frameon=False)
    for xax in ax: xax.grid(axis="y", alpha=.2)
    fig.suptitle("Stage-dependent origin of the JET−noJET angular-momentum difference",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(out/"figure6_stage_integrated_attribution.png", dpi=220,
                                    bbox_inches="tight"); plt.close(fig)

    quality = [{"case": r.case, "hour": float(r.hour), "correlation": float(r.M_closure_corr),
                "normalized_residual": float(r.M_normalized_residual), "status": r.quality}
               for _, r in d.sort_values(["hour", "case"]).iterrows()]
    summary = {
        "verification_status": "PARTIALLY_VERIFIED",
        "primary_validated_quantity": "4-h integrated axisymmetric absolute-angular-momentum budget",
        "not_validated": "instantaneous/gridpoint classic inertial-stability tendency budget",
        "quality": quality,
        "phase_integrals_m2_s-1": integ,
        "robust_45_70_fraction_of_observed": {
            "mean": integ["45–70 robust"]["mean"] / integ["45–70 robust"]["observed"],
            "eddy": integ["45–70 robust"]["eddy"] / integ["45–70 robust"]["observed"],
            "pgrad": integ["45–70 robust"]["pgrad"] / integ["45–70 robust"]["observed"],
            "residual": integ["45–70 robust"]["residual"] / integ["45–70 robust"]["observed"],
        },
    }
    (out/"budget_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    d.to_csv(out/"inner_outflow_reynolds_quality.csv", index=False)


if __name__ == "__main__":
    main()
