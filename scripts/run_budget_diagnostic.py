#!/usr/bin/env python
"""
柱坐标方位角平均与径向/切向动量收支诊断。

合并了 budget_full 和 budget_full_grouped_residual 的功能。
通过 --grouped-residual 开关控制是否启用进阶功能。

用法:
  # 基础版
  python scripts/run_budget_diagnostic.py --input dataset/cm1out.nc --output dataset/budget.nc

  # 进阶版（分组残差分配 + 内核约束）
  python scripts/run_budget_diagnostic.py --input dataset/cm1out.nc --grouped-residual \\
      --enable-core-stabilization --core-radius-km 6.0 --subtract-translation-speed
"""

import argparse
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，支持直接运行
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.azimuthal_avg import run_budget_diagnostic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CM1 柱坐标方位角平均与 mean/eddy 动量诊断"
    )
    parser.add_argument("--input", default="dataset/cm1out.nc", help="输入 NC 文件")
    parser.add_argument(
        "--output", default="dataset/typhoon_azimuthal_avg_budget.nc",
        help="输出 NC 文件"
    )
    parser.add_argument("--max-r-km", type=float, default=300.0, help="最大半径 (km)")
    parser.add_argument("--dr-km", type=float, default=2.0, help="径向分箱间隔 (km)")
    parser.add_argument("--max-z-km", type=float, default=20.0, help="最大高度 (km)")
    parser.add_argument("--center-window", type=int, default=21, help="中心定位平滑窗口")
    parser.add_argument(
        "--center-method", choices=["min", "centroid", "streamfunction"],
        default="min", help="台风中心定位方法"
    )
    parser.add_argument("--max-times", type=int, default=None, help="仅处理 N 个时次")
    parser.add_argument("--start-time", type=int, default=0, help="起始时间索引（默认 0）")

    # 进阶功能
    parser.add_argument(
        "--grouped-residual", action="store_true",
        help="启用分组残差分配（进阶版）"
    )
    parser.add_argument(
        "--enable-core-stabilization", action="store_true",
        help="启用 r=0 轴对称约束"
    )
    parser.add_argument(
        "--core-radius-km", type=float, default=6.0,
        help="内核约束半径 (km)"
    )
    parser.add_argument(
        "--center-time-smooth-window", type=int, default=11,
        help="中心轨迹时间平滑窗口"
    )
    parser.add_argument(
        "--subtract-translation-speed", action="store_true",
        help="减去台风移动速度"
    )
    parser.add_argument("--quiet", action="store_true", help="减少日志输出")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    run_budget_diagnostic(
        input_file=args.input,
        output_file=args.output,
        max_r_km=args.max_r_km,
        dr_km=args.dr_km,
        max_z_km=args.max_z_km,
        center_window=args.center_window,
        center_method=args.center_method,
        enable_core_stabilization=args.enable_core_stabilization,
        core_radius_km=args.core_radius_km,
        center_time_smooth_window=args.center_time_smooth_window,
        subtract_translation_speed=args.subtract_translation_speed,
        max_times=args.max_times,
        start_time_idx=args.start_time,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
