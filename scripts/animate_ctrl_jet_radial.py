#!/usr/bin/env python3
from pathlib import Path
import json
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

sys.path.insert(0, ".")
from src._se_pipeline_single import PipelineConfig, azimuthal_average_from_3d

CASES = {
    "CTRL": "/data/zhangyx/DATA/cm1out_25N_nojet.nc",
    "JET": "/data/zhangyx/DATA/cm1out_25N_9o_jet_30_15km.nc",
}
HOURS = np.arange(0.0, 210.01, 2.0)
OUT = Path("output/ctrl_jet_radial_wind_animation_25N_9deg_30_15km")
CACHE = OUT / "cache"
FRAMES = OUT / "frames"


def extract(case, input_file, hour):
    path = CACHE / f"{case}_{int(hour):03d}h.npz"
    if path.exists():
        return np.load(path)
    cfg = PipelineConfig(
        input_file=input_file, output_dir=str(OUT), target_time_hours=float(hour),
        max_r_km=600.0, dr_km=12.0, max_z_km=20.0,
        coriolis_f=6.1636e-5, include_model_budget_terms=False,
        write_netcdf=False, write_ieee=False, plot_solution=False,
    )
    avg = azimuthal_average_from_3d(cfg)
    np.savez_compressed(
        path, r_km=np.asarray(avg["r_km"], float),
        z_km=np.asarray(avg["z_km"], float),
        ur=np.asarray(avg["ur"], np.float32),
        hour=float(avg["time_seconds_used"][0] / 3600.0),
        center_x_km=float(avg["center_x_km"][0]),
        center_y_km=float(avg["center_y_km"][0]),
    )
    return np.load(path)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    FRAMES.mkdir(parents=True, exist_ok=True)
    records = []
    values = []
    for case, input_file in CASES.items():
        for i, hour in enumerate(HOURS):
            data = extract(case, input_file, hour)
            values.append(np.abs(data["ur"]).ravel())
            print(f"extract {case} {i + 1}/{len(HOURS)} h={hour:g}", flush=True)
    limit = float(np.nanpercentile(np.concatenate(values), 99.2))
    limit = max(2.0, min(limit, 15.0))
    cmap = LinearSegmentedColormap.from_list(
        "radial", ["#274C77", "#6096BA", "#C9DCE8", "#F7F7F4",
                   "#F2C4A7", "#E76F51", "#9D2933"], N=256,
    )
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    for i, hour in enumerate(HOURS):
        fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), sharex=True,
                                 sharey=True, constrained_layout=True)
        panel_record = {"frame": i, "requested_hour": float(hour)}
        image = None
        for ax, (case, _) in zip(axes, CASES.items()):
            data = np.load(CACHE / f"{case}_{int(hour):03d}h.npz")
            r, z, ur = data["r_km"], data["z_km"], data["ur"]
            image = ax.pcolormesh(r, z, ur, cmap=cmap, norm=norm,
                                  shading="auto", rasterized=True)
            ax.contour(r, z, ur, levels=[0.0], colors="0.2", linewidths=0.55)
            ax.axhspan(10.0, 17.0, color="#F2C14E", alpha=0.08)
            ax.axhline(10.0, color="#A66F00", ls=":", lw=0.7)
            ax.axhline(17.0, color="#A66F00", ls=":", lw=0.7)
            selected_hour = float(data["hour"])
            ax.set_title(f"{case}  |  {selected_hour:.0f} h", fontweight="bold")
            ax.set_xlim(0.0, 600.0)
            ax.set_ylim(0.0, 20.0)
            ax.set_xlabel("Radius from TC center (km)")
            ax.grid(alpha=0.15, ls="--")
            panel_record[case] = {
                "hour": selected_hour,
                "center_x_km": float(data["center_x_km"]),
                "center_y_km": float(data["center_y_km"]),
            }
        axes[0].set_ylabel("Height (km)")
        colorbar = fig.colorbar(image, ax=axes, pad=0.02, extend="both")
        colorbar.set_label("Azimuthal-mean radial wind (m s$^{-1}$; outward +)")
        fig.suptitle(
            "CTRL versus JET: storm-centered axisymmetric radial circulation\n"
            "Yellow band: 10–17 km outflow layer",
            fontweight="bold",
        )
        fig.savefig(FRAMES / f"frame_{i:04d}.png", dpi=140,
                    bbox_inches="tight")
        plt.close(fig)
        records.append(panel_record)
        print(f"render {i + 1}/{len(HOURS)}", flush=True)
    video = OUT / "ctrl_vs_jet_axisymmetric_radial_wind_0_210h.mp4"
    ffmpeg = "/data1/home/zhangyx/miniconda3/envs/cm1_tc/bin/ffmpeg"
    subprocess.run([
        ffmpeg, "-y", "-framerate", "5", "-i", str(FRAMES / "frame_%04d.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-crf", "20", str(video),
    ], check=True)
    for index in (0, len(HOURS) // 2, len(HOURS) - 1):
        destination = OUT / f"sample_{int(HOURS[index]):03d}h.png"
        destination.write_bytes((FRAMES / f"frame_{index:04d}.png").read_bytes())
    metadata = {
        "inputs": CASES,
        "field": "storm-centered azimuthal-mean radial wind",
        "time_range_h": [0.0, 210.0], "interval_h": 2.0,
        "frames": len(HOURS), "fps": 5,
        "domain": {"radius_km": [0.0, 600.0], "height_km": [0.0, 20.0]},
        "fixed_color_limit_m_s": [-limit, limit],
        "outflow_layer_km": [10.0, 17.0], "records": records,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"COMPLETE {video} limit={limit:g}", flush=True)


if __name__ == "__main__":
    main()
