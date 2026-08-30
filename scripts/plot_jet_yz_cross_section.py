#!/usr/bin/env python3
"""Plot the imposed CM1 westerly jet in a meridional-height section."""
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


def read_section(path: str, target_hour: float, sigma: float) -> dict:
    with xr.open_dataset(path, decode_times=False) as ds:
        time_s = np.asarray(ds["time"].values, dtype=float)
        it = int(np.argmin(np.abs(time_s - target_hour * 3600.0)))
        psfc = np.asarray(ds["psfc"].isel(time=it).values, dtype=float)
        iy, ix = np.unravel_index(
            np.nanargmin(gaussian_filter(psfc, sigma=sigma)), psfc.shape
        )
        # Destagger u using the two x faces surrounding the storm-centre column.
        u_faces = np.asarray(
            ds["u"].isel(time=it, xf=slice(ix, ix + 2)).values, dtype=float
        )
        return {
            "time_hours": float(time_s[it] / 3600.0),
            "u_yz": np.nanmean(u_faces, axis=-1),
            "yh": np.asarray(ds["yh"].values, dtype=float),
            "zh": np.asarray(ds["zh"].values, dtype=float),
            "xc": float(ds["xh"].values[ix]),
            "yc": float(ds["yh"].values[iy]),
        }


def nice_limit(value: float) -> float:
    return max(5.0, float(np.ceil(value / 5.0) * 5.0))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jet", required=True)
    p.add_argument("--nojet", default=None)
    p.add_argument("--time-hours", type=float, default=0.0)
    p.add_argument("--max-z-km", type=float, default=20.0)
    p.add_argument("--y-half-width-km", type=float, default=2500.0)
    p.add_argument("--jet-offset-km", type=float, default=888.0)
    p.add_argument("--jet-z-km", type=float, default=12.0)
    p.add_argument("--center-smooth-sigma", type=float, default=2.0)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-json", default=None)
    a = p.parse_args()

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    jet = read_section(a.jet, a.time_hours, a.center_smooth_sigma)
    nojet = read_section(a.nojet, a.time_hours, a.center_smooth_sigma) if a.nojet else None

    yrel = jet["yh"] - jet["yc"]
    keep_y = np.abs(yrel) <= a.y_half_width_km
    keep_z = jet["zh"] <= a.max_z_km
    y = yrel[keep_y]
    z = jet["zh"][keep_z]
    ujet = jet["u_yz"][np.ix_(keep_z, keep_y)]
    fields = [(ujet, "JET total zonal wind")]
    if nojet is not None:
        if not (np.allclose(nojet["yh"], jet["yh"]) and np.allclose(nojet["zh"], jet["zh"])):
            raise ValueError("JET and noJET grids do not match")
        unojet = nojet["u_yz"][np.ix_(keep_z, keep_y)]
        fields.append((ujet - unojet, "JET - noJET (imposed-jet signal)"))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(
        1, len(fields), figsize=(7.2 * len(fields), 5.5), sharex=True, sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    maxima = []
    for ip, (ax, (field, title)) in enumerate(zip(axes, fields)):
        limit = nice_limit(float(np.nanpercentile(np.abs(field), 99.8)))
        levels = np.linspace(-limit, limit, 33)
        cf = ax.contourf(y, z, field, levels=levels, cmap="RdBu_r", extend="both")
        pos = np.arange(10.0, limit + 0.1, 10.0)
        if pos.size:
            ax.contour(y, z, field, levels=pos, colors="#8b1a1a", linewidths=0.7)
            ax.contour(y, z, field, levels=-pos[::-1], colors="#174a7e", linewidths=0.7)
        ax.axvline(0.0, color="#222222", linestyle="--", linewidth=1.0, label="TC centre")
        ax.axvline(a.jet_offset_km, color="#7b3294", linestyle="--", linewidth=1.4, label="Jet axis")
        ax.scatter([a.jet_offset_km], [a.jet_z_km], marker="*", s=105,
                   facecolor="#f2c14e", edgecolor="#222222", linewidth=0.8, zorder=5)
        ax.set_title(f"({chr(97 + ip)}) {title}", loc="left", fontweight="bold")
        ax.set_xlabel("Meridional distance from TC centre (km)")
        ax.set_xlim(-a.y_half_width_km, a.y_half_width_km)
        ax.set_ylim(0.0, a.max_z_km)
        ax.grid(color="0.6", linewidth=0.35, alpha=0.22)
        cb = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.11, fraction=0.08)
        cb.set_label("Zonal wind, u (m s$^{-1}$)")
        iz, iy = np.unravel_index(np.nanargmax(field), field.shape)
        maxima.append({"panel": title, "maximum_m_s": float(field[iz, iy]),
                       "maximum_y_relative_km": float(y[iy]), "maximum_z_km": float(z[iz]),
                       "minimum_m_s": float(np.nanmin(field))})
    axes[0].set_ylabel("Height (km)")
    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle(
        f"Meridional-height structure of the imposed westerly jet at {jet['time_hours']:.0f} h\n"
        "section through the pressure-defined TC centre, looking west",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = {"jet_file": a.jet, "nojet_file": a.nojet,
               "selected_time_hours": jet["time_hours"], "section_x_km": jet["xc"],
               "jet_tc_center_y_km": jet["yc"],
               "expected_jet_axis_relative_y_km": a.jet_offset_km,
               "expected_jet_axis_z_km": a.jet_z_km,
               "plotted_y_relative_km": [float(y[0]), float(y[-1])],
               "plotted_z_km": [float(z[0]), float(z[-1])],
               "maxima": maxima, "output": str(out)}
    summary_path = Path(a.summary_json) if a.summary_json else out.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
