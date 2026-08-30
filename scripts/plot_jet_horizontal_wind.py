#!/usr/bin/env python3
"""Plot a TC-centred upper-level horizontal wind field and diagnose a jet axis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter


def _destagger_u(u: np.ndarray) -> np.ndarray:
    return 0.5 * (u[:, :-1] + u[:, 1:])


def _destagger_v(v: np.ndarray) -> np.ndarray:
    return 0.5 * (v[:-1, :] + v[1:, :])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot upper-level horizontal wind and diagnose the environmental jet axis."
    )
    parser.add_argument("--input", required=True, help="CM1 NetCDF file")
    parser.add_argument("--time-hours", type=float, default=0.0)
    parser.add_argument("--z-km", type=float, default=11.0)
    parser.add_argument("--jet-offset-km", type=float, default=555.0)
    parser.add_argument("--x-half-width-km", type=float, default=1500.0)
    parser.add_argument("--y-south-km", type=float, default=1000.0)
    parser.add_argument("--y-north-km", type=float, default=1600.0)
    parser.add_argument("--center-smooth-sigma", type=float, default=2.0)
    parser.add_argument("--quiver-stride", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(args.input, decode_times=False) as ds:
        time_s = np.asarray(ds["time"].values, dtype=float)
        zh = np.asarray(ds["zh"].values, dtype=float)
        xh = np.asarray(ds["xh"].values, dtype=float)
        yh = np.asarray(ds["yh"].values, dtype=float)

        time_index = int(np.argmin(np.abs(time_s - args.time_hours * 3600.0)))
        z_index = int(np.argmin(np.abs(zh - args.z_km)))
        time_hours = float(time_s[time_index] / 3600.0)
        z_km = float(zh[z_index])

        u_stag = np.asarray(ds["u"].isel(time=time_index, zh=z_index).values, dtype=float)
        v_stag = np.asarray(ds["v"].isel(time=time_index, zh=z_index).values, dtype=float)
        psfc = np.asarray(ds["psfc"].isel(time=time_index).values, dtype=float)

    u = _destagger_u(u_stag)
    v = _destagger_v(v_stag)
    wind_speed = np.hypot(u, v)

    psfc_smooth = gaussian_filter(psfc, sigma=args.center_smooth_sigma)
    iyc, ixc = np.unravel_index(np.nanargmin(psfc_smooth), psfc_smooth.shape)
    xc = float(xh[ixc])
    yc = float(yh[iyc])
    x_rel = xh - xc
    y_rel = yh - yc

    x_mask = np.abs(x_rel) <= args.x_half_width_km
    y_mask = (y_rel >= -args.y_south_km) & (y_rel <= args.y_north_km)
    x_plot = x_rel[x_mask]
    y_plot = y_rel[y_mask]
    u_plot = u[np.ix_(y_mask, x_mask)]
    v_plot = v[np.ix_(y_mask, x_mask)]
    speed_plot = wind_speed[np.ix_(y_mask, x_mask)]

    # The TC occupies only a small fraction of this wide x range. The x mean
    # therefore isolates the imposed, zonally elongated westerly jet.
    u_xmean = np.nanmean(u_plot, axis=1)
    diagnosed_index = int(np.nanargmax(u_xmean))
    diagnosed_axis_km = float(y_plot[diagnosed_index])
    diagnosed_umax = float(u_xmean[diagnosed_index])

    vmax = max(5.0, float(np.ceil(np.nanpercentile(speed_plot, 99.5) / 5.0) * 5.0))
    levels = np.linspace(0.0, vmax, 31)

    fig = plt.figure(figsize=(14, 8.5), dpi=160)
    gs = fig.add_gridspec(1, 2, width_ratios=(4.8, 1.35), wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    ax_profile = fig.add_subplot(gs[0, 1], sharey=ax)

    cf = ax.contourf(x_plot, y_plot, speed_plot, levels=levels,
                     cmap="turbo", extend="max")
    cbar = fig.colorbar(cf, ax=ax, pad=0.015, fraction=0.04)
    cbar.set_label("Horizontal wind speed (m s$^{-1}$)")

    stride = max(1, args.quiver_stride)
    xx, yy = np.meshgrid(x_plot, y_plot)
    q = ax.quiver(
        xx[::stride, ::stride], yy[::stride, ::stride],
        u_plot[::stride, ::stride], v_plot[::stride, ::stride],
        color="black", alpha=0.65, scale=500, width=0.0022,
    )
    ax.quiverkey(q, 0.89, 0.04, 20.0, "20 m s$^{-1}$", coordinates="axes")
    ax.scatter(0.0, 0.0, marker="o", s=70, c="white", edgecolors="black",
               linewidths=1.5, zorder=6, label="TC centre")
    ax.axhline(args.jet_offset_km, color="magenta", linestyle="--", linewidth=2.0,
               label=f"Expected jet axis (+{args.jet_offset_km:.0f} km)")
    ax.axhline(diagnosed_axis_km, color="cyan", linestyle="-.", linewidth=2.0,
               label=f"Diagnosed x-mean u max ({diagnosed_axis_km:.0f} km)")
    ax.set_xlabel("Zonal distance from TC centre (km)")
    ax.set_ylabel("Meridional distance from TC centre (km)")
    ax.set_title(
        f"Upper-level horizontal wind: t={time_hours:.0f} h, z={z_km:.2f} km\n"
        f"{Path(args.input).name}",
        fontweight="bold",
    )
    ax.grid(alpha=0.18, linestyle="--")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92)

    ax_profile.plot(u_xmean, y_plot, color="navy", linewidth=2.2)
    ax_profile.axhline(args.jet_offset_km, color="magenta", linestyle="--", linewidth=1.8)
    ax_profile.axhline(diagnosed_axis_km, color="cyan", linestyle="-.", linewidth=1.8)
    ax_profile.axvline(0.0, color="0.4", linewidth=0.8)
    ax_profile.scatter([diagnosed_umax], [diagnosed_axis_km], color="cyan",
                       edgecolor="black", zorder=5)
    ax_profile.set_xlabel("x-mean u (m s$^{-1}$)")
    ax_profile.set_title("Zonal-mean\nwesterly profile")
    ax_profile.grid(alpha=0.25, linestyle="--")
    ax_profile.tick_params(labelleft=False)

    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "input": args.input,
        "time_index": time_index,
        "time_hours": time_hours,
        "z_index": z_index,
        "z_km": z_km,
        "tc_center_km": [xc, yc],
        "expected_jet_axis_relative_y_km": args.jet_offset_km,
        "diagnosed_xmean_u_axis_relative_y_km": diagnosed_axis_km,
        "diagnosed_xmean_u_max_m_s": diagnosed_umax,
        "plot_max_wind_speed_m_s": float(np.nanmax(speed_plot)),
        "output": str(output),
    }
    summary_path = Path(args.summary_json) if args.summary_json else output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
