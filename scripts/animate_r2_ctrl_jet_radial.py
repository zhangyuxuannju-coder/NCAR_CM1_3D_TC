#!/usr/bin/env python3
"""Same-time CTRL/JET R2 Thompson radial-wind animation (0--240 h)."""
from pathlib import Path
import json
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

sys.path.insert(0, ".")
from src._se_pipeline_single import PipelineConfig, azimuthal_average_from_3d

CASES = {
    "CTRL": "/data/zhangyx/DATA/cm1out_25N_CTRL_R2_Thompson_OML100_240h.nc",
    "JET": "/data/zhangyx/DATA/cm1out_25N_JET_R2_Thompson_OML100_240h.nc",
}
HOURS = np.arange(0.0, 240.01, 2.0)
OUT = Path("output/r2_thompson_oml100_ctrl_jet_radial_wind_animation")
CACHE, FRAMES = OUT / "cache", OUT / "frames"


def extract(case, input_file, hour):
    path = CACHE / f"{case}_{int(hour):03d}h.npz"
    if path.exists():
        return np.load(path)
    cfg = PipelineConfig(input_file=input_file, output_dir=str(OUT), target_time_hours=float(hour),
                         max_r_km=600, dr_km=12, max_z_km=20, coriolis_f=6.1636e-5,
                         include_model_budget_terms=False, write_netcdf=False, write_ieee=False,
                         plot_solution=False)
    avg = azimuthal_average_from_3d(cfg)
    np.savez_compressed(path, r_km=np.asarray(avg["r_km"], float), z_km=np.asarray(avg["z_km"], float),
                        ur=np.asarray(avg["ur"], np.float32),
                        hour=float(avg["time_seconds_used"][0] / 3600),
                        center_x_km=float(avg["center_x_km"][0]), center_y_km=float(avg["center_y_km"][0]))
    return np.load(path)


def main():
    CACHE.mkdir(parents=True, exist_ok=True); FRAMES.mkdir(parents=True, exist_ok=True)
    all_abs, records = [], []
    for case, source in CASES.items():
        for n, hour in enumerate(HOURS, 1):
            data = extract(case, source, hour)
            all_abs.append(np.abs(data["ur"]).ravel())
            print(f"extract {case} {n}/{len(HOURS)} h={hour:g}", flush=True)
    limit = max(2.0, min(float(np.nanpercentile(np.concatenate(all_abs), 99.2)), 15.0))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    for n, hour in enumerate(HOURS):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharex=True, sharey=True, constrained_layout=True)
        record = {"frame": n, "requested_hour": float(hour)}
        image = None
        for ax, (case, _) in zip(axes, CASES.items()):
            data = np.load(CACHE / f"{case}_{int(hour):03d}h.npz")
            image = ax.pcolormesh(data["r_km"], data["z_km"], data["ur"], cmap="RdBu_r", norm=norm, shading="auto")
            ax.contour(data["r_km"], data["z_km"], data["ur"], levels=[0], colors="0.2", linewidths=.55)
            ax.axhspan(10, 17, color="#F2C14E", alpha=.10); ax.axhline(10, color="#A66F00", ls=":", lw=.7); ax.axhline(17, color="#A66F00", ls=":", lw=.7)
            ax.set_title(f"{case} | {data['hour']:.0f} h", fontweight="bold"); ax.set_xlim(0,600); ax.set_ylim(0,20); ax.set_xlabel("Radius from TC center (km)")
            record[case] = {"hour":float(data["hour"]), "center_x_km":float(data["center_x_km"]), "center_y_km":float(data["center_y_km"])}
        axes[0].set_ylabel("Height (km)")
        bar=fig.colorbar(image,ax=axes,pad=.02,extend="both"); bar.set_label("Azimuthal-mean radial wind (m s$^{-1}$; outward +)")
        fig.suptitle("Same-time CTRL and JET radial circulation | yellow: 10–17 km outflow layer", fontweight="bold")
        fig.savefig(FRAMES / f"frame_{n:04d}.png", dpi=140, bbox_inches="tight"); plt.close(fig)
        records.append(record); print(f"render {n+1}/{len(HOURS)}",flush=True)
    video = OUT / "ctrl_vs_jet_axisymmetric_radial_wind_0_240h.mp4"
    subprocess.run(["/data1/home/zhangyx/miniconda3/envs/cm1_tc/bin/ffmpeg", "-y", "-framerate", "5", "-i", str(FRAMES / "frame_%04d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(video)], check=True)
    for n in (0, len(HOURS)//2, len(HOURS)-1): (OUT / f"sample_{int(HOURS[n]):03d}h.png").write_bytes((FRAMES / f"frame_{n:04d}.png").read_bytes())
    (OUT / "metadata.json").write_text(json.dumps({"inputs":CASES,"field":"storm-centered azimuthal-mean radial wind","time_range_h":[0,240],"interval_h":2,"frames":len(HOURS),"fps":5,"fixed_color_limit_m_s":[-limit,limit],"outflow_layer_km":[10,17],"records":records},indent=2))
    print(f"COMPLETE {video} limit={limit:g}")


if __name__ == "__main__": main()
