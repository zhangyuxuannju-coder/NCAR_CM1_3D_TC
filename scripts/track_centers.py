#!/usr/bin/env python
"""
台风中心批量追踪。

从 identification_loc.py 提取，支持多时次中心追踪 + 输出 CSV + 绘图。

用法:
  python scripts/track_centers.py --input dataset/cm1out.nc --output centers.csv --plot
"""

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
from src.center_finder import find_smoothed_min_point


def track_all_times(
    nc_file: str, var_name: str = "psfc", center_method: str = "min",
    window: int = 21, verbose: bool = True,
) -> list:
    """追踪所有时次的台风中心，返回列表。"""
    from netCDF4 import Dataset

    results = []
    with Dataset(nc_file, "r") as nc:
        time_arr = nc.variables["time"][:]
        n_times = len(time_arr)

        for t_idx in range(n_times):
            center = find_smoothed_min_point(
                nc_file, time_key=t_idx, var_name=var_name,
                window=window, center_method=center_method, verbose=False,
            )
            results.append({
                "time_index": t_idx,
                "time_value": float(time_arr[t_idx]),
                "x": float(center["x"]),
                "y": float(center["y"]),
                "psfc_min": float(center["smoothed_value"]),
            })
            if verbose and (t_idx % 10 == 0 or t_idx == n_times - 1):
                print(f"[{t_idx:04d}/{n_times:04d}] "
                      f"center=({center['x']:.2f}, {center['y']:.2f})")

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="台风中心批量追踪")
    parser.add_argument("--input", default="dataset/cm1out.nc", help="输入 NC 文件")
    parser.add_argument("--output", default=None, help="输出 CSV 文件")
    parser.add_argument("--var", default="psfc", help="用于定位的变量名")
    parser.add_argument("--center-method", choices=["min", "centroid", "streamfunction"],
                        default="min")
    parser.add_argument("--window", type=int, default=21)
    parser.add_argument("--plot", action="store_true", help="绘制轨迹图")
    parser.add_argument("--plot-output", default="figure/center_tracks.png")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    results = track_all_times(
        nc_file=args.input, var_name=args.var,
        center_method=args.center_method, window=args.window,
        verbose=not args.quiet,
    )

    if args.output:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"[INFO] 中心轨迹已保存至: {args.output}")

    if args.plot:
        import pandas as pd
        from src.plotting import plot_center_tracks

        df = pd.DataFrame(results)
        plot_center_tracks(df, Path(args.plot_output))
        print(f"[INFO] 轨迹图已保存至: {args.plot_output}")


if __name__ == "__main__":
    main()
