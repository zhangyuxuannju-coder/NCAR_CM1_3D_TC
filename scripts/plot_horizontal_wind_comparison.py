#!/usr/bin/env python3
"""Compare storm-centred upper-level horizontal wind inside a fixed radius."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import netCDF4
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nojet", required=True)
    p.add_argument("--jet", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-json", default=None)
    p.add_argument("--data-npz", default=None)
    p.add_argument("--times-hours", type=float, nargs="+", default=[25.0, 55.0, 80.0])
    p.add_argument("--height-km", type=float, default=14.0)
    p.add_argument("--radius-km", type=float, default=300.0)
    p.add_argument("--center-smooth-sigma", type=float, default=2.0)
    p.add_argument("--stream-density", type=float, default=1.05)
    p.add_argument("--stream-grid-size", type=int, default=181)
    p.add_argument("--speed-max", type=float, default=None)
    return p.parse_args()


def center_velocity(u_stag, v_stag):
    u = 0.5 * (u_stag[:, :-1] + u_stag[:, 1:])
    v = 0.5 * (v_stag[:-1, :] + v_stag[1:, :])
    return u, v


def diagnose(path, target_hour, target_height_km, radius_km, center_sigma, ngrid):
    with netCDF4.Dataset(path) as ds:
        time = np.asarray(ds.variables["time"][:], dtype=float)
        x = np.asarray(ds.variables["xh"][:], dtype=float)
        y = np.asarray(ds.variables["yh"][:], dtype=float)
        z = np.asarray(ds.variables["zh"][:], dtype=float)
        it = int(np.nanargmin(np.abs(time - target_hour * 3600.0)))
        iz = int(np.nanargmin(np.abs(z - target_height_km)))
        psfc = np.asarray(ds.variables["psfc"][it], dtype=float)
        u_stag = np.asarray(ds.variables["u"][it, iz], dtype=float)
        v_stag = np.asarray(ds.variables["v"][it, iz], dtype=float)

    smooth = gaussian_filter(psfc, sigma=center_sigma, mode="nearest")
    iyc, ixc = np.unravel_index(np.nanargmin(smooth), smooth.shape)
    xc, yc = float(x[ixc]), float(y[iyc])
    u, v = center_velocity(u_stag, v_stag)

    xr, yr = x - xc, y - yc
    ix = np.where(np.abs(xr) <= radius_km)[0]
    iy = np.where(np.abs(yr) <= radius_km)[0]
    u_native = u[np.ix_(iy, ix)]
    v_native = v[np.ix_(iy, ix)]

    # CM1 may use a stretched mesh. Interpolate to a regular storm-centred grid
    # for streamlines and mask the corners outside the requested radius.
    grid = np.linspace(-radius_km, radius_km, ngrid)
    xx, yy = np.meshgrid(grid, grid)
    query = np.column_stack((yy.ravel(), xx.ravel()))
    u_grid = RegularGridInterpolator(
        (yr[iy], xr[ix]), u_native, bounds_error=False, fill_value=np.nan
    )(query).reshape(xx.shape)
    v_grid = RegularGridInterpolator(
        (yr[iy], xr[ix]), v_native, bounds_error=False, fill_value=np.nan
    )(query).reshape(xx.shape)
    outside = np.hypot(xx, yy) > radius_km
    u_grid[outside] = np.nan
    v_grid[outside] = np.nan
    speed = np.hypot(u_grid, v_grid)

    return {
        "x_km": grid, "y_km": grid, "u_m_s": u_grid, "v_m_s": v_grid,
        "speed_m_s": speed, "selected_hour": float(time[it] / 3600.0),
        "selected_height_km": float(z[iz]), "center_x_km": xc,
        "center_y_km": yc, "min_psfc_hpa": float(smooth[iyc, ixc] / 100.0),
    }


def main():
    a = parse_args()
    cases = [("noJET", a.nojet), ("JET", a.jet)]
    panels = []
    # Landscape ordering: cases occupy rows and times occupy columns.
    for case, path in cases:
        for hour in a.times_hours:
            panel = diagnose(path, hour, a.height_km, a.radius_km,
                             a.center_smooth_sigma, a.stream_grid_size)
            panel["case"] = case
            panel["requested_hour"] = hour
            panels.append(panel)

    finite = np.concatenate([p["speed_m_s"][np.isfinite(p["speed_m_s"])] for p in panels])
    vmax = a.speed_max
    if vmax is None:
        vmax = max(10.0, float(np.ceil(np.nanpercentile(finite, 99.5) / 5.0) * 5.0))
    levels = np.linspace(0.0, vmax, 31)
    cmap = LinearSegmentedColormap.from_list(
        "nature_wind", ["#F7F7F2", "#DCE9EB", "#A8CDD2", "#6AABB6",
                        "#377D91", "#20546A", "#123747"], N=256
    )
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.linewidth": 0.75, "xtick.major.width": 0.75,
        "ytick.major.width": 0.75, "xtick.direction": "out",
        "ytick.direction": "out", "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    nrows = len(cases)
    ncols = len(a.times_hours)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.15 * ncols, 8.7),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_2d(axes)
    letters = "abcdefghijklmnopqrstuvwxyz"
    mappable = None
    for i, (ax, p) in enumerate(zip(axes.flat, panels)):
        x, y = p["x_km"], p["y_km"]
        mappable = ax.contourf(x, y, p["speed_m_s"], levels=levels,
                               cmap=cmap, extend="max")
        ax.streamplot(x, y, p["u_m_s"], p["v_m_s"], density=a.stream_density,
                      color="#252B35", linewidth=0.58, arrowsize=0.72,
                      minlength=0.15, integration_direction="both")
        ring_radii = (
            (100.0, 200.0, 300.0)
            if a.radius_km <= 300.0
            else tuple(np.linspace(a.radius_km / 4.0, a.radius_km, 4))
        )
        for radius in ring_radii:
            ax.add_patch(plt.Circle((0.0, 0.0), radius, fill=False,
                                    color="#60656C", lw=0.45, ls="--", alpha=0.55))
        ax.scatter(0.0, 0.0, s=24, marker="x", c="#9D2933", linewidths=1.2,
                   zorder=8)
        ax.text(-0.13, 1.03, f"({letters[i]})", transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom")
        ax.set_title(f"{p['case']}  ·  {p['selected_hour']:.0f} h",
                     fontsize=11, fontweight="bold", pad=7)
        ax.set_xlim(-a.radius_km, a.radius_km)
        ax.set_ylim(-a.radius_km, a.radius_km)
        ax.set_aspect("equal", adjustable="box")
        tick_values = np.linspace(-a.radius_km, a.radius_km, 5)
        ax.set_xticks(tick_values)
        ax.set_yticks(tick_values)
        for spine in ax.spines.values():
            spine.set_color("#4A4A4A")
    for ax in axes[-1, :]:
        ax.set_xlabel("Storm-relative x (km)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Storm-relative y (km)")

    cbar = fig.colorbar(mappable, ax=axes, orientation="horizontal", shrink=0.70,
                        pad=0.025, aspect=40, extend="max")
    color_ticks = np.linspace(0.0, vmax, 5)
    cbar.set_ticks(color_ticks)
    cbar.set_ticklabels([f"{value:g}" for value in color_ticks])
    cbar.set_label(r"Horizontal wind speed (m s$^{-1}$)")
    fig.suptitle(
        f"Storm-centred horizontal flow within {a.radius_km:.0f} km at "
        f"z = {panels[0]['selected_height_km']:.2f} km",
        fontsize=14, fontweight="bold",
    )

    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "height_requested_km": a.height_km,
        "selected_height_km": panels[0]["selected_height_km"],
        "radius_km": a.radius_km,
        "wind_speed_color_limit_m_s": [0.0, vmax],
        "wind_frame": "Earth-relative CM1 horizontal wind, displayed in storm-centred coordinates",
        "panels": [],
    }
    archive = {"x_km": panels[0]["x_km"], "y_km": panels[0]["y_km"]}
    for p in panels:
        key = f"{p['case'].lower()}_{p['selected_hour']:.0f}h"
        archive[f"{key}_u_m_s"] = p["u_m_s"]
        archive[f"{key}_v_m_s"] = p["v_m_s"]
        archive[f"{key}_speed_m_s"] = p["speed_m_s"]
        values = p["speed_m_s"][np.isfinite(p["speed_m_s"])]
        summary["panels"].append({
            "case": p["case"], "requested_hour": p["requested_hour"],
            "selected_hour": p["selected_hour"],
            "center_x_km": p["center_x_km"], "center_y_km": p["center_y_km"],
            "min_psfc_hpa": p["min_psfc_hpa"],
            "wind_speed_max_m_s": float(np.nanmax(values)),
            "wind_speed_mean_m_s": float(np.nanmean(values)),
            "wind_speed_p95_m_s": float(np.nanpercentile(values, 95.0)),
        })
    summary_path = Path(a.summary_json) if a.summary_json else output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if a.data_npz:
        data_path = Path(a.data_npz)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(data_path, **archive)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
