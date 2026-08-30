#!/usr/bin/env python3
"""Plot storm-centred Ertel PV and horizontal-wind streamlines for CM1 cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import netCDF4
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nojet", required=True)
    p.add_argument("--jet", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-json", default=None)
    p.add_argument("--times-hours", type=float, nargs="+", default=[25.0, 55.0, 80.0, 110.0])
    p.add_argument("--height-km", type=float, default=13.0)
    p.add_argument("--half-width-km", type=float, default=300.0)
    p.add_argument("--pv-limit", type=float, default=5.0)
    p.add_argument("--pv-smooth-sigma", type=float, default=0.65)
    p.add_argument("--center-smooth-sigma", type=float, default=2.0)
    p.add_argument("--stream-density", type=float, default=0.85)
    p.add_argument("--stream-stride", type=int, default=2)
    return p.parse_args()


def center_velocity(u_stag, v_stag):
    return 0.5 * (u_stag[..., :-1] + u_stag[..., 1:]), 0.5 * (
        v_stag[..., :-1, :] + v_stag[..., 1:, :]
    )


def diagnose_panel(path, target_hour, target_height_km, half_width_km,
                   center_sigma, pv_sigma):
    with netCDF4.Dataset(path) as ds:
        time = np.asarray(ds.variables["time"][:], dtype=float)
        it = int(np.nanargmin(np.abs(time - target_hour * 3600.0)))
        x = np.asarray(ds.variables["xh"][:], dtype=float)
        y = np.asarray(ds.variables["yh"][:], dtype=float)
        z = np.asarray(ds.variables["zh"][:], dtype=float)
        f = float(np.asarray(ds.variables["f_cor"][:]).ravel()[0])
        k = int(np.nanargmin(np.abs(z - target_height_km)))
        if k < 1 or k > z.size - 2:
            raise ValueError("PV calculation requires one scalar level above and below target")
        ks = slice(k - 1, k + 2)

        psfc = np.asarray(ds.variables["psfc"][it], dtype=float)
        psfc_smooth = gaussian_filter(psfc, sigma=center_sigma, mode="nearest")
        iyc, ixc = np.unravel_index(np.nanargmin(psfc_smooth), psfc_smooth.shape)
        xc, yc = float(x[ixc]), float(y[iyc])

        theta = np.asarray(ds.variables["th"][it, ks], dtype=float)
        rho = np.asarray(ds.variables["rho"][it, ks], dtype=float)
        u_stag = np.asarray(ds.variables["u"][it, ks], dtype=float)
        v_stag = np.asarray(ds.variables["v"][it, ks], dtype=float)
        u, v = center_velocity(u_stag, v_stag)
        # w is staggered vertically. Average adjacent full levels to the three
        # selected scalar levels before taking derivatives.
        w_full = np.asarray(ds.variables["w"][it, k - 1:k + 3], dtype=float)
        w = 0.5 * (w_full[:-1] + w_full[1:])

    xm = x * 1000.0
    ym = y * 1000.0
    zm = z[k - 1:k + 2] * 1000.0
    dth_dz, dth_dy, dth_dx = np.gradient(theta, zm, ym, xm, edge_order=2)
    du_dz, du_dy, du_dx = np.gradient(u, zm, ym, xm, edge_order=2)
    dv_dz, dv_dy, dv_dx = np.gradient(v, zm, ym, xm, edge_order=2)
    dw_dz, dw_dy, dw_dx = np.gradient(w, zm, ym, xm, edge_order=2)

    omega_x = dw_dy - dv_dz
    omega_y = du_dz - dw_dx
    omega_z_abs = dv_dx - du_dy + f
    pv = (
        omega_x * dth_dx + omega_y * dth_dy + omega_z_abs * dth_dz
    ) / np.maximum(rho, 1.0e-8)
    pv_pvu = pv[1] * 1.0e6
    if pv_sigma > 0.0:
        pv_pvu = gaussian_filter(pv_pvu, sigma=pv_sigma, mode="nearest")

    xr = x - xc
    yr = y - yc
    ix = np.where(np.abs(xr) <= half_width_km)[0]
    iy = np.where(np.abs(yr) <= half_width_km)[0]
    selection = np.ix_(iy, ix)
    x_plot = xr[ix]
    y_plot = yr[iy]
    xx_plot, yy_plot = np.meshgrid(x_plot, y_plot)
    outside = np.hypot(xx_plot, yy_plot) > half_width_km
    pv_plot = pv_pvu[selection]
    u_plot = u[1][selection]
    v_plot = v[1][selection]
    pv_plot[outside] = np.nan
    u_plot[outside] = np.nan
    v_plot[outside] = np.nan
    return {
        "x_km": x_plot,
        "y_km": y_plot,
        "pv_pvu": pv_plot,
        "u_m_s": u_plot,
        "v_m_s": v_plot,
        "selected_hour": float(time[it] / 3600.0),
        "selected_height_km": float(z[k]),
        "center_x_km": xc,
        "center_y_km": yc,
        "min_psfc_hpa": float(psfc_smooth[iyc, ixc] / 100.0),
        "f_s_1": f,
    }


def main():
    a = parse_args()
    cases = [("noJET", a.nojet), ("JET", a.jet)]
    panels = []
    for hour in a.times_hours:
        for name, path in cases:
            result = diagnose_panel(
                path, hour, a.height_km, a.half_width_km,
                a.center_smooth_sigma, a.pv_smooth_sigma,
            )
            result["case"] = name
            result["requested_hour"] = hour
            panels.append(result)

    # Nature-inspired restrained blue-neutral-red palette: negative and
    # positive PV remain equally prominent and zero stays nearly white.
    cmap = LinearSegmentedColormap.from_list(
        "nature_pv",
        ["#2F4B7C", "#6785B5", "#C5D4E8", "#F7F7F4",
         "#F3C6B3", "#E56B4E", "#9D2933"],
        N=256,
    )
    norm = TwoSlopeNorm(vmin=-a.pv_limit, vcenter=0.0, vmax=a.pv_limit)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    nrows = len(a.times_hours)
    fig, axes = plt.subplots(nrows, 2, figsize=(9.2, 4.15 * nrows),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_2d(axes)
    letters = "abcdefghijklmnopqrstuvwxyz"
    mappable = None
    for i, (ax, p) in enumerate(zip(axes.flat, panels)):
        # CM1 uses a stretched outer grid.  Keep PV on that native grid, but
        # interpolate the vector field to a uniform grid required by streamplot.
        x_raw, y_raw = p["x_km"], p["y_km"]
        x_stream = np.linspace(float(x_raw[0]), float(x_raw[-1]), x_raw.size)
        y_stream = np.linspace(float(y_raw[0]), float(y_raw[-1]), y_raw.size)
        xx_stream, yy_stream = np.meshgrid(x_stream, y_stream)
        query = np.column_stack((yy_stream.ravel(), xx_stream.ravel()))
        u_stream = RegularGridInterpolator(
            (y_raw, x_raw), p["u_m_s"], bounds_error=False, fill_value=np.nan
        )(query).reshape(yy_stream.shape)
        v_stream = RegularGridInterpolator(
            (y_raw, x_raw), p["v_m_s"], bounds_error=False, fill_value=np.nan
        )(query).reshape(yy_stream.shape)
        outside_stream = np.hypot(xx_stream, yy_stream) > a.half_width_km
        u_stream[outside_stream] = np.nan
        v_stream[outside_stream] = np.nan
        mappable = ax.pcolormesh(x_raw, y_raw, p["pv_pvu"], shading="auto", cmap=cmap,
                                 norm=norm, rasterized=True)
        stride = max(1, a.stream_stride)
        ax.streamplot(
            x_stream[::stride], y_stream[::stride],
            u_stream[::stride, ::stride], v_stream[::stride, ::stride],
            density=a.stream_density, color="#232936", linewidth=0.72,
            arrowsize=0.85, arrowstyle="-|>", minlength=0.15,
            integration_direction="both",
        )
        for radius in (100.0, 200.0, 300.0):
            if radius <= a.half_width_km:
                ax.add_patch(plt.Circle(
                    (0.0, 0.0), radius, fill=False, color="#60656C",
                    lw=0.45, ls="--", alpha=0.55,
                ))
        ax.scatter(0.0, 0.0, s=24, marker="x", c="#9D2933",
                   linewidths=1.2, zorder=8)
        ax.text(-0.12, 1.04, f"({letters[i]})", transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom")
        ax.set_title(f"{p['case']}  ·  {p['selected_hour']:.0f} h",
                     fontsize=11, fontweight="bold", pad=7)
        ax.set_xlim(-a.half_width_km, a.half_width_km)
        ax.set_ylim(-a.half_width_km, a.half_width_km)
        ax.set_aspect("equal", adjustable="box")
        ticks = np.linspace(-a.half_width_km, a.half_width_km, 5)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        for spine in ax.spines.values():
            spine.set_color("#4A4A4A")
    for ax in axes[-1, :]:
        ax.set_xlabel("Storm-relative x (km)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Storm-relative y (km)")

    cbar = fig.colorbar(mappable, ax=axes, orientation="horizontal", shrink=0.68,
                        pad=0.025, aspect=40, extend="both")
    cbar.set_label(r"Ertel potential vorticity (PVU; $10^{-6}$ K m$^2$ kg$^{-1}$ s$^{-1}$)")
    cbar.set_ticks(np.linspace(-a.pv_limit, a.pv_limit, 5))
    fig.suptitle(
        f"PV and horizontal flow within {a.half_width_km:.0f} km at "
        f"z = {panels[0]['selected_height_km']:.2f} km",
        fontsize=14, fontweight="bold",
    )

    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "definition": "dry Ertel PV = rho^-1 (absolute vorticity dot grad(theta))",
        "pv_units": "PVU = 1e-6 K m2 kg-1 s-1",
        "pv_color_limit": [-a.pv_limit, a.pv_limit],
        "height_requested_km": a.height_km,
        "panels": [],
    }
    for p in panels:
        finite = p["pv_pvu"][np.isfinite(p["pv_pvu"])]
        summary["panels"].append({
            "case": p["case"], "requested_hour": p["requested_hour"],
            "selected_hour": p["selected_hour"],
            "selected_height_km": p["selected_height_km"],
            "center_x_km": p["center_x_km"], "center_y_km": p["center_y_km"],
            "min_psfc_hpa": p["min_psfc_hpa"],
            "pv_pvu_percentiles_1_50_99": [float(x) for x in np.nanpercentile(finite, [1, 50, 99])],
        })
    summary_path = Path(a.summary_json) if a.summary_json else output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
