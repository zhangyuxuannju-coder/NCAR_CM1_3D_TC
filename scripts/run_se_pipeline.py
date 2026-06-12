#!/usr/bin/env python
"""
Sawyer-Eliassen (SE) 方程诊断流水线 — 统一入口。

refactor/ 完全自包含，不依赖上级目录。

用法:
  python scripts/run_se_pipeline.py --mode single \\
      --input dataset/cm1out.nc --target-time-hours 72 --output-dir output/se_pipeline/72h

  python scripts/run_se_pipeline.py --mode evap \\
      --input dataset/cm1out.nc --target-time-hours 72 --output-dir output/se_pipeline/evap_72h

  python scripts/run_se_pipeline.py --mode timeavg \\
      --input dataset/cm1out.nc --time-avg-start-hours 64 --time-avg-end-hours 72
"""

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SE 方程诊断流水线 (single/evap/timeavg)")

    p.add_argument("--mode", choices=["single","evap","timeavg"], default="single")
    p.add_argument("--input", default="dataset/cm1out.nc")
    p.add_argument("--output-dir", default="output/se_pipeline")

    p.add_argument("--time-index", type=int, default=0)
    p.add_argument("--target-time-hours", type=float, default=None)
    p.add_argument("--target-time-seconds", type=float, default=None)
    p.add_argument("--time-avg-start-hours", type=float, default=None)
    p.add_argument("--time-avg-end-hours", type=float, default=None)

    p.add_argument("--max-r-km", type=float, default=300.0)
    p.add_argument("--dr-km", type=float, default=2.0)
    p.add_argument("--max-z-km", type=float, default=20.0)

    p.add_argument("--center-window", type=int, default=21)
    p.add_argument("--center-method", choices=["min","centroid","streamfunction"], default="min")

    p.add_argument("--sor-max-iter", type=int, default=60000)
    p.add_argument("--sor-omega", type=float, default=1.8)
    p.add_argument("--sor-tol", type=float, default=1.0e-14)
    p.add_argument("--baroclinic-scale", type=float, default=0.4)

    p.add_argument("--no-netcdf", action="store_true")
    p.add_argument("--no-ieee", action="store_true")
    p.add_argument("--no-plot", action="store_true")

    p.add_argument("--evap-q0", type=float, default=-2.0e-4)
    p.add_argument("--evap-r-center", type=float, default=145.0)
    p.add_argument("--evap-z-center", type=float, default=15.0)
    p.add_argument("--evap-r-half", type=float, default=105.0)
    p.add_argument("--evap-z-half", type=float, default=2.5)
    p.add_argument("--evap-dipole", action="store_true")

    return p


def main() -> None:
    args = build_parser().parse_args()
    print(f"[INFO] SE Pipeline mode={args.mode}, input={args.input}")

    if args.mode == "single":
        from src._se_pipeline_single import PipelineConfig, run_pipeline
        cfg = PipelineConfig(
            input_file=args.input, output_dir=args.output_dir,
            target_time_hours=args.target_time_hours,
            target_time_seconds=args.target_time_seconds, time_index=args.time_index,
            max_r_km=args.max_r_km, dr_km=args.dr_km, max_z_km=args.max_z_km,
            center_window=args.center_window, center_method=args.center_method,
            sor_max_iter=args.sor_max_iter, sor_omega=args.sor_omega,
            sor_tol=args.sor_tol, baroclinic_scale=args.baroclinic_scale,
            write_netcdf=not args.no_netcdf, write_ieee=not args.no_ieee,
            plot_solution=not args.no_plot,
        )
        run_pipeline(cfg)

    elif args.mode == "evap":
        from src._se_pipeline_evap import PipelineConfig, run_pipeline
        cfg = PipelineConfig(
            input_file=args.input, output_dir=args.output_dir,
            target_time_hours=args.target_time_hours,
            target_time_seconds=args.target_time_seconds, time_index=args.time_index,
            max_r_km=args.max_r_km, dr_km=args.dr_km, max_z_km=args.max_z_km,
            center_window=args.center_window, center_method=args.center_method,
            sor_max_iter=args.sor_max_iter, sor_omega=args.sor_omega,
            sor_tol=args.sor_tol, baroclinic_scale=args.baroclinic_scale,
            write_netcdf=not args.no_netcdf, write_ieee=not args.no_ieee,
            plot_solution=not args.no_plot,
            evap_cooling_q0=args.evap_q0, evap_r_center=args.evap_r_center,
            evap_z_center=args.evap_z_center, evap_r_half=args.evap_r_half,
            evap_z_half=args.evap_z_half, evap_dipole=args.evap_dipole,
        )
        run_pipeline(cfg)

    elif args.mode == "timeavg":
        from src._se_pipeline_timeavg import PipelineConfig, run_pipeline
        cfg = PipelineConfig(
            input_file=args.input, output_dir=args.output_dir,
            target_time_hours=args.target_time_hours,
            target_time_seconds=args.target_time_seconds, time_index=args.time_index,
            time_avg_start_hours=args.time_avg_start_hours,
            time_avg_end_hours=args.time_avg_end_hours,
            max_r_km=args.max_r_km, dr_km=args.dr_km, max_z_km=args.max_z_km,
            center_window=args.center_window, center_method=args.center_method,
            sor_max_iter=args.sor_max_iter, sor_omega=args.sor_omega,
            sor_tol=args.sor_tol, baroclinic_scale=args.baroclinic_scale,
            write_netcdf=not args.no_netcdf, write_ieee=not args.no_ieee,
            plot_solution=not args.no_plot,
        )
        run_pipeline(cfg)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print("Pipeline completed.")


if __name__ == "__main__":
    main()
