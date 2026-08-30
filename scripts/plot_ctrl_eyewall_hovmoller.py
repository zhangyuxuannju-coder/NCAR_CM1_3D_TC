#!/usr/bin/env python3
"""Time-radius eyewall diagnostics for the three 25N/22N/27N CTRL runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter, gaussian_filter1d


CASES = (
    ("22N CTRL", "cm1out_22N_nojet.nc"),
    ("25N CTRL", "cm1out_25N_nojet.nc"),
    ("27N CTRL", "cm1out_27N_nojet.nc"),
)


def radial_mean(field, bins, valid, nr):
    values = np.asarray(field, dtype=np.float64).ravel()[valid]
    ibin = bins[valid]
    finite = np.isfinite(values)
    sums = np.bincount(ibin[finite], weights=values[finite], minlength=nr)
    counts = np.bincount(ibin[finite], minlength=nr)
    out = np.full(nr, np.nan)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def diagnose(path: Path, start_h: float, end_h: float, max_r: float, dr: float):
    with Dataset(path) as ds:
        hours_all = np.asarray(ds.variables["time"][:], dtype=float) / 3600.0
        tidx = np.where((hours_all >= start_h - 1e-5) & (hours_all <= end_h + 1e-5))[0]
        if tidx.size == 0:
            raise ValueError(f"No output between {start_h:g} and {end_h:g} h in {path}")

        xh = np.asarray(ds.variables["xh"][:], dtype=float)
        yh = np.asarray(ds.variables["yh"][:], dtype=float)
        zh = np.asarray(ds.variables["zh"][:], dtype=float)
        iz_wind = int(np.argmin(np.abs(zh - 1.2)))
        iz_dbz = int(np.argmin(np.abs(zh - 2.0)))

        edges = np.arange(0.0, max_r + dr + 1e-8, dr)
        radii = 0.5 * (edges[:-1] + edges[1:])
        nr = radii.size
        vt = np.full((tidx.size, nr), np.nan)
        dbz = np.full_like(vt, np.nan)
        centers = np.full((tidx.size, 2), np.nan)

        xx, yy = np.meshgrid(xh, yh)
        for j, it in enumerate(tidx):
            psfc = np.asarray(ds.variables["psfc"][it, :, :], dtype=float)
            smooth = gaussian_filter(psfc, sigma=2.0, mode="nearest")
            iy, ix = np.unravel_index(np.nanargmin(smooth), smooth.shape)
            xc, yc = xh[ix], yh[iy]
            centers[j] = (xc, yc)

            dx = xx - xc
            dy = yy - yc
            rr = np.hypot(dx, dy)
            theta = np.arctan2(dy, dx)
            ibin2d = np.floor(rr / dr).astype(np.int32)
            valid = ((rr >= 0.0) & (rr < max_r) & (ibin2d >= 0) & (ibin2d < nr)).ravel()
            bins = ibin2d.ravel()

            u_stag = np.asarray(ds.variables["u"][it, iz_wind, :, :], dtype=float)
            v_stag = np.asarray(ds.variables["v"][it, iz_wind, :, :], dtype=float)
            u = 0.5 * (u_stag[:, :-1] + u_stag[:, 1:])
            v = 0.5 * (v_stag[:-1, :] + v_stag[1:, :])
            vtheta = -u * np.sin(theta) + v * np.cos(theta)
            vt[j] = radial_mean(vtheta, bins, valid, nr)

            dbz2d = np.asarray(ds.variables["dbz"][it, iz_dbz, :, :], dtype=float)
            linear_z = np.where(np.isfinite(dbz2d), 10.0 ** (dbz2d / 10.0), np.nan)
            zmean = radial_mean(linear_z, bins, valid, nr)
            dbz[j] = 10.0 * np.log10(np.maximum(zmean, 1.0e-4))

            if j % 10 == 0 or j == tidx.size - 1:
                print(f"  {path.name}: {hours_all[it]:.0f} h ({j + 1}/{tidx.size})", flush=True)

    vt_smooth = gaussian_filter1d(vt, sigma=0.8, axis=1, mode="nearest")
    inner = (radii >= 6.0) & (radii <= min(max_r, 120.0))
    rmw = radii[inner][np.nanargmax(vt_smooth[:, inner], axis=1)]
    return {
        "hours": hours_all[tidx],
        "radii": radii,
        "vt": vt,
        "dbz": dbz,
        "rmw": rmw,
        "wind_z": zh[iz_wind],
        "dbz_z": zh[iz_dbz],
        "centers": centers,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/data/zhangyx/DATA")
    p.add_argument("--output", required=True)
    p.add_argument("--start", type=float, default=50.0)
    p.add_argument("--end", type=float, default=100.0)
    p.add_argument("--max-radius", type=float, default=180.0)
    p.add_argument("--dr", type=float, default=3.0)
    args = p.parse_args()

    results = []
    for label, filename in CASES:
        print(f"Diagnosing {label}: {filename}", flush=True)
        results.append((label, diagnose(Path(args.data_dir) / filename, args.start, args.end,
                                        args.max_radius, args.dr)))

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(3, 2, figsize=(13.2, 12.5), sharex=True, sharey=True,
                             constrained_layout=False)

    vt_levels = np.arange(0, 66, 5)
    dbz_levels = np.arange(0, 56, 5)
    vt_cmap = plt.get_cmap("turbo", len(vt_levels) - 1)
    dbz_cmap = plt.get_cmap("YlGnBu", len(dbz_levels) - 1)
    vt_norm = BoundaryNorm(vt_levels, vt_cmap.N, clip=False)
    dbz_norm = BoundaryNorm(dbz_levels, dbz_cmap.N, clip=False)
    m_vt = m_dbz = None

    for row, (label, result) in enumerate(results):
        t = result["hours"]
        r = result["radii"]
        ax = axes[row, 0]
        m_vt = ax.contourf(t, r, result["vt"].T, levels=vt_levels, cmap=vt_cmap,
                           norm=vt_norm, extend="max")
        cs = ax.contour(t, r, result["vt"].T, levels=[15, 25, 35, 45, 55],
                        colors="k", linewidths=0.45, alpha=0.7)
        ax.clabel(cs, fmt="%d", fontsize=7, inline=True)
        ax.plot(t, result["rmw"], color="white", lw=2.3, label="RMW")
        ax.plot(t, result["rmw"], color="black", lw=0.7)
        ax.axvline(72.0, color="crimson", ls="--", lw=1.6)
        ax.set_title(f"{label}: azimuthal-mean tangential wind\n"
                     f"z = {result['wind_z']:.2f} km")

        ax = axes[row, 1]
        m_dbz = ax.contourf(t, r, result["dbz"].T, levels=dbz_levels, cmap=dbz_cmap,
                            norm=dbz_norm, extend="both")
        cs = ax.contour(t, r, result["dbz"].T, levels=[20, 30, 40, 50],
                        colors="k", linewidths=0.5, alpha=0.7)
        ax.clabel(cs, fmt="%d", fontsize=7, inline=True)
        ax.axvline(72.0, color="crimson", ls="--", lw=1.6)
        ax.set_title(f"{label}: azimuthal-mean reflectivity\n"
                     f"z = {result['dbz_z']:.2f} km (linear-Z mean)")

        for ax in axes[row]:
            ax.set_ylim(0, args.max_radius)
            ax.grid(color="0.7", lw=0.35, alpha=0.45)
            ax.set_ylabel("Radius from pressure centre (km)")

    for ax in axes[-1]:
        ax.set_xlabel("Simulation time (h)")

    fig.suptitle("CTRL eyewall-structure evolution around the 72-h intensity change",
                 fontsize=16, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.105, top=0.92,
                        hspace=0.34, wspace=0.12)
    cax1 = fig.add_axes([0.09, 0.045, 0.39, 0.018])
    cb1 = fig.colorbar(m_vt, cax=cax1, orientation="horizontal")
    cb1.set_label(r"Tangential wind (m s$^{-1}$); white/black line = RMW")
    cax2 = fig.add_axes([0.54, 0.045, 0.39, 0.018])
    cb2 = fig.colorbar(m_dbz, cax=cax2, orientation="horizontal")
    cb2.set_label("Reflectivity (dBZ; azimuthal mean in linear Z)")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
