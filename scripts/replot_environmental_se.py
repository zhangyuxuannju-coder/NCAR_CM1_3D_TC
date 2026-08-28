#!/usr/bin/env python3
"""Replot environmental-eddy SE products while excluding inner radii."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-npz", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-r-km", type=float, default=50.0)
    p.add_argument("--max-r-km", type=float, default=None)
    p.add_argument("--percentile", type=float, default=99.0)
    p.add_argument("--jet-axis-r-km", type=float, default=888.0)
    p.add_argument("--jet-effective-r-km", type=float, default=None)
    return p.parse_args()


def limit(fields, percentile):
    values = np.concatenate([np.abs(x[np.isfinite(x)]) for x in fields])
    return max(float(np.nanpercentile(values, percentile)), 1.0e-20)


def velocity_zr(field, nr, nz):
    field = np.asarray(field, dtype=float)
    if field.shape == (nr, nz + 2):
        return field[:, 1:-1].T
    if field.shape == (nr, nz):
        return field.T
    if field.shape == (nz, nr):
        return field
    raise ValueError(f"Unsupported velocity shape: {field.shape}")


def markers(ax, configured, effective):
    ax.axvline(configured, color="#7b3294", lw=1.2, ls="--", label="configured jet axis")
    if effective is not None:
        ax.axvline(effective, color="#008837", lw=1.2, ls=":", label="72-h effective separation")


def extrema(field, r, z, mask):
    valid = np.asarray(mask, dtype=bool) & np.isfinite(field)
    if not np.any(valid):
        return {
            "max": None,
            "min": None,
            "rms": None,
            "status": "requested region is outside the displayed domain",
        }
    work = np.where(valid, field, np.nan)
    imax = np.unravel_index(np.nanargmax(work), work.shape)
    imin = np.unravel_index(np.nanargmin(work), work.shape)
    return {
        "max": {"value": float(work[imax]), "r_km": float(r[imax[1]]), "z_km": float(z[imax[0]])},
        "min": {"value": float(work[imin]), "r_km": float(r[imin[1]]), "z_km": float(z[imin[0]])},
        "rms": float(np.sqrt(np.nanmean(work ** 2))),
    }


def main():
    a = parse_args()
    d = np.load(a.input_npz)
    r0 = np.asarray(d["r_km"], dtype=float)
    z = np.asarray(d["z_km"], dtype=float)
    rmax = float(r0[-1]) if a.max_r_km is None else a.max_r_km
    keep = (r0 >= a.min_r_km) & (r0 <= rmax)
    r = r0[keep]
    nr, nz = r0.size, z.size
    total = np.asarray(d["F_lambda_env"])[:, keep]
    radial = np.asarray(d["F_lambda_env_radial"])[:, keep]
    vertical = np.asarray(d["F_lambda_env_vertical"])[:, keep]
    rhs = np.asarray(d["forcing_env"])[:, keep]
    uenv = velocity_zr(d["U_env"], nr, nz)[:, keep]
    wenv = velocity_zr(d["W_env"], nr, nz)[:, keep]
    uctrl = velocity_zr(d["U_ctrl"], nr, nz)[:, keep]
    usum = velocity_zr(d["U_ctrl_plus_env"], nr, nz)[:, keep]
    rr, zz = np.meshgrid(r, z)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    flimit = limit([total, radial, vertical], a.percentile)
    levels = np.linspace(-flimit, flimit, 25)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), sharey=True, constrained_layout=True)
    for ax, field, title in zip(axes, [total, radial, vertical],
            [r"$F_{\lambda,env}$", r"$F_{\lambda,env}^{(r)}$", r"$F_{\lambda,env}^{(z)}$"]):
        im = ax.contourf(rr, zz, field, levels=levels, cmap="RdBu_r", extend="both")
        markers(ax, a.jet_axis_r_km, a.jet_effective_r_km)
        ax.set_title(title + r" (m s$^{-2}$)")
        ax.set_xlabel("Radius from TC center (km)")
        ax.grid(alpha=.14, ls="--")
        fig.colorbar(im, ax=ax, pad=.015)
    axes[0].set_ylabel("Height (km)")
    axes[-1].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(
        f"Environmental eddy-momentum forcing, r = {r[0]:.0f}–{r[-1]:.0f} km\n"
        f"Shared color scale: {a.percentile:g}th percentile after excluding r < {a.min_r_km:g} km",
        fontsize=13, fontweight="bold")
    fig.savefig(out / "environmental_eddy_forcing_r50_jet_region.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fields = [total, rhs, uenv, wenv, uctrl, usum]
    titles = [r"$F_{\lambda,env}$ (m s$^{-2}$)",
              r"$-\partial_z(\chi\xi F_{\lambda,env})$",
              r"$U_{env}$ (m s$^{-1}$)", r"$W_{env}$ (m s$^{-1}$)",
              r"$U_{CTRL}$ (m s$^{-1}$)", r"$U_{CTRL+env}$ (m s$^{-1}$)"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.4), sharex=True, sharey=True,
                             constrained_layout=True)
    color_limits = {}
    for ax, field, title in zip(axes.flat, fields, titles):
        vmax = limit([field], a.percentile)
        color_limits[title] = vmax
        im = ax.contourf(rr, zz, field, levels=np.linspace(-vmax, vmax, 25),
                         cmap="RdBu_r", extend="both")
        markers(ax, a.jet_axis_r_km, a.jet_effective_r_km)
        ax.set_title(title)
        ax.set_xlabel("Radius from TC center (km)")
        ax.set_ylabel("Height (km)")
        ax.grid(alpha=.14, ls="--")
        fig.colorbar(im, ax=ax, pad=.015)
    axes[0, -1].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(
        f"Fixed-CTRL SE diagnosis including the environmental-jet region\n"
        f"Displayed and color-scaled with r = {r[0]:.0f}–{r[-1]:.0f} km only",
        fontsize=13, fontweight="bold")
    fig.savefig(out / "se_environmental_eddy_response_r50_jet_region.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    jetmask = (rr >= 750) & (rr <= 1100) & (zz >= 10) & (zz <= 17)
    summary = {
        "displayed_radius_km": [float(r[0]), float(r[-1])],
        "excluded_inner_radius_km": a.min_r_km,
        "color_percentile": a.percentile,
        "configured_jet_axis_r_km": a.jet_axis_r_km,
        "effective_jet_separation_r_km": a.jet_effective_r_km,
        "jet_box_r_km": [750, 1100], "jet_box_z_km": [10, 17],
        "jet_box": {
            "F_lambda_env_m_s2": extrema(total, r, z, jetmask),
            "U_env_m_s": extrema(uenv, r, z, jetmask),
            "W_env_m_s": extrema(wenv, r, z, jetmask),
        },
        "color_limits": color_limits,
        "warning": "U_env/W_env are fixed-CTRL regularized balanced responses, not the full JET-minus-CTRL circulation."
    }
    (out / "outer_jet_region_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
