#!/usr/bin/env python
"""
台风中心 3D 轨迹图。

从 prs_centers_profile.py 提取。

用法:
  python scripts/profile_centers.py --input dataset/cm1out.nc --output figure/centers_3d.png
"""

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
from src.center_finder import find_smoothed_min_prs
from src.plotting import plot_centers_3d


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="台风中心 3D 轨迹图")
    parser.add_argument("--input", default="dataset/cm1out.nc")
    parser.add_argument("--time", type=float, default=400)
    parser.add_argument("--z-min", type=float, default=0.5)
    parser.add_argument("--z-max", type=float, default=20.0)
    parser.add_argument("--z-step", type=float, default=0.5)
    parser.add_argument("--var", default="prs")
    parser.add_argument("--center-method", default="min")
    parser.add_argument("--output", default="figure/centers_3d.png")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    z_levels = np.arange(args.z_min, args.z_max + args.z_step, args.z_step)
    centers = []

    for z in z_levels:
        try:
            c = find_smoothed_min_prs(
                nc_file=args.input, time_key=args.time,
                target_zh=float(z), var_name=args.var,
                center_method=args.center_method, verbose=False,
            )
            centers.append(c)
        except Exception as e:
            if not args.quiet:
                print(f"[WARN] z={z:.1f} km 定位失败: {e}")

    plot_centers_3d(centers, out_png=Path(args.output))
    print(f"[INFO] 3D 轨迹图已保存至: {args.output}")


if __name__ == "__main__":
    main()
