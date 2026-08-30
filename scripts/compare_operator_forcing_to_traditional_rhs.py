#!/usr/bin/env python3
"""Compare equivalent operator-perturbation forcing with Bui Eq. (14) RHS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src._se_pipeline_single import PipelineConfig, azimuthal_average_from_3d
from src.se_bui import build_basic_state, build_forcing


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--nojet", required=True)
    p.add_argument("--jet", required=True)
    p.add_argument("--operator-products", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--time-hours", type=float, default=5.0)
    p.add_argument("--max-r-km", type=float, default=1200.0)
    p.add_argument("--dr-km", type=float, default=12.0)
    p.add_argument("--max-z-km", type=float, default=20.0)
    p.add_argument("--f", type=float, default=6.2e-5)
    return p.parse_args()


def read_case(path, a):
    cfg = PipelineConfig(
        input_file=path, output_dir=a.output_dir,
        target_time_hours=a.time_hours,
        max_r_km=a.max_r_km, dr_km=a.dr_km, max_z_km=a.max_z_km,
        coriolis_f=a.f,
        include_model_budget_terms=True,
        write_netcdf=False, write_ieee=False, plot_solution=False,
    )
    return azimuthal_average_from_3d(cfg)


def weights(r_m, z_m):
    return (
        np.maximum(r_m, 1.0)[None, :]
        * np.abs(np.gradient(r_m))[None, :]
        * np.abs(np.gradient(z_m))[:, None]
    )


def mask(r, z, b):
    r0, r1, z0, z1 = b
    return ((r[None, :] >= r0) & (r[None, :] <= r1)
            & (z[:, None] >= z0) & (z[:, None] <= z1))


def wrms(x, w, m):
    good = m & np.isfinite(x) & np.isfinite(w)
    den = np.sum(w[good])
    return float(np.sqrt(np.sum(w[good] * x[good] ** 2) / den)) if den > 0 else np.nan


def wcorr(x, y, w, m):
    good = m & np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    if np.count_nonzero(good) < 4:
        return np.nan
    ww = w[good] / np.sum(w[good]); xx = x[good]; yy = y[good]
    xm = np.sum(ww * xx); ym = np.sum(ww * yy)
    num = np.sum(ww * (xx-xm) * (yy-ym))
    den = np.sqrt(np.sum(ww*(xx-xm)**2) * np.sum(ww*(yy-ym)**2))
    return float(num/den) if den > 0 else np.nan


def safe_ratio(x, y):
    return float(x/y) if np.isfinite(y) and y > 1e-30 else np.nan


def forcing(avg, r_m, z_m, f):
    basic = build_basic_state(avg["ut"], avg["theta"], avg["rho"], r_m, z_m, f)
    return build_forcing(basic, avg["Q"], avg["Fnu"], r_m, z_m)


def cmap():
    return LinearSegmentedColormap.from_list(
        "rhs", ["#24486E", "#6F98BE", "#C8D9E7", "#F7F7F4", "#F4C6AF", "#D96B4C", "#8E2C3A"], N=256
    )


def scale(fs, m):
    x = np.concatenate([np.abs(f[m & np.isfinite(f)]) for f in fs])
    vmax = max(float(np.percentile(x, 99.0)), 1e-30)
    s = 10.0**np.floor(np.log10(vmax))
    return s, vmax/s


def draw(r, z, fields, output):
    shown = (r[None, :] >= 30) & (z[:, None] <= 18)
    diff_fields = fields[:4]
    abs_fields = fields[4:]
    sd, ld = scale(diff_fields, shown)
    sa, la = scale(abs_fields, shown)
    titles = [
        r"Equivalent operator forcing $S_{jet}^{stab}$",
        r"JET-CTRL traditional thermal RHS",
        r"JET-CTRL traditional momentum RHS",
        r"JET-CTRL traditional total RHS",
        r"CTRL traditional total RHS",
        r"JET traditional total RHS",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.2), sharex=True, sharey=True, constrained_layout=True)
    maps = []
    for i, (ax, f, title) in enumerate(zip(axes.flat, fields, titles)):
        s, lim = (sd, ld) if i < 4 else (sa, la)
        p = ax.pcolormesh(r, z, f/s, cmap=cmap(), norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), shading="auto", rasterized=True)
        finite = f[np.isfinite(f)]
        if finite.size and np.min(finite) <= 0 <= np.max(finite):
            ax.contour(r, z, f, levels=[0], colors="0.25", linewidths=0.45)
        ax.plot(888, 12, "*", ms=9, color="#7A1FA2", mec="white", mew=.5)
        ax.set_title(title, fontsize=9.7)
        ax.text(-.1, 1.03, f"({'abcdef'[i]})", transform=ax.transAxes, fontweight="bold")
        ax.set_xlim(0, r[-1]); ax.set_ylim(0, 18)
        maps.append(p)
    for ax in axes[-1]: ax.set_xlabel("Radius from TC centre (km)")
    for ax in axes[:, 0]: ax.set_ylabel("Height (km)")
    cb1 = fig.colorbar(maps[0], ax=axes[0], orientation="horizontal", shrink=.72, pad=.03)
    cb1.set_label(rf"Difference/equivalent forcing ($\times10^{{{int(np.log10(sd))}}}$ K$^{{-1}}$ s$^{{-3}}$)")
    cb2 = fig.colorbar(maps[-1], ax=axes[1], orientation="horizontal", shrink=.72, pad=.03)
    cb2.set_label(rf"Absolute traditional RHS ($\times10^{{{int(np.log10(sa))}}}$ K$^{{-1}}$ s$^{{-3}}$)")
    fig.suptitle("Operator-equivalent forcing versus the traditional Bui SE RHS · 5 h", fontsize=14, fontweight="bold")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    a = args(); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    c = read_case(a.nojet, a); j = read_case(a.jet, a)
    r = np.asarray(c["r_km"], float); z = np.asarray(c["z_km"], float)
    rm, zm = r*1000, z*1000
    fc = forcing(c, rm, zm, a.f); fj = forcing(j, rm, zm, a.f)
    prod = np.load(a.operator_products)
    if not np.allclose(r, prod["r_km"]) or not np.allclose(z, prod["z_km"]):
        raise ValueError("Operator products and RHS use different grids")
    s = np.asarray(prod["S_total"], float)
    dth = fj["forcing_thermal"] - fc["forcing_thermal"]
    dmo = fj["forcing_momentum"] - fc["forcing_momentum"]
    dto = fj["forcing_total"] - fc["forcing_total"]
    w = weights(rm, zm)
    domains = {
        "full": (30, 1200, .5, 18),
        "inner_outflow": (50, 350, 10, 16),
        "jet_annulus": (650, 1150, 10, 16),
    }
    report = {
        "selected_time_hours": float(c["time_seconds_used"][0]/3600),
        "traditional_rhs_definition": "Bui Eq.14 full available Q_total and F_lambda_total from direct eddies plus available CM1 non-advective/model budget terms",
        "thermal_terms_ctrl": [str(x) for x in c["thermal_budget_terms_used"]],
        "thermal_terms_jet": [str(x) for x in j["thermal_budget_terms_used"]],
        "momentum_pairs_ctrl": [str(x) for x in c["momentum_budget_pairs_used"]],
        "momentum_pairs_jet": [str(x) for x in j["momentum_budget_pairs_used"]],
        "domains": {},
    }
    for name, b in domains.items():
        m = mask(r, z, b)
        sr = wrms(s,w,m); dr = wrms(dto,w,m)
        cr = wrms(fc["forcing_total"],w,m); jr = wrms(fj["forcing_total"],w,m)
        report["domains"][name] = {
            "bounds_r0_r1_z0_z1_km": list(b),
            "operator_equivalent_rms": sr,
            "ctrl_traditional_total_rms": cr,
            "jet_traditional_total_rms": jr,
            "traditional_delta_thermal_rms": wrms(dth,w,m),
            "traditional_delta_momentum_rms": wrms(dmo,w,m),
            "traditional_delta_total_rms": dr,
            "operator_over_ctrl_total": safe_ratio(sr,cr),
            "operator_over_jet_total": safe_ratio(sr,jr),
            "operator_over_traditional_delta_total": safe_ratio(sr,dr),
            "operator_vs_traditional_delta_correlation": wcorr(s,dto,w,m),
            "traditional_delta_cancellation_factor": safe_ratio(wrms(dth,w,m)+wrms(dmo,w,m),dr),
        }
    draw(r,z,[s,dth,dmo,dto,fc["forcing_total"],fj["forcing_total"]],out/"operator_vs_traditional_rhs.png")
    np.savez_compressed(out/"operator_vs_traditional_rhs.npz",r_km=r,z_km=z,S_operator=s,
                        rhs_ctrl_total=fc["forcing_total"],rhs_jet_total=fj["forcing_total"],
                        delta_rhs_thermal=dth,delta_rhs_momentum=dmo,delta_rhs_total=dto)
    (out/"operator_vs_traditional_rhs.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
