#!/usr/bin/env python
"""
水平场绘图工具 — 统一入口。

合并了 cm1_out_nc_plot.py 和 plot_single_frame.py 的功能。

用法:
  # 单帧填色图
  python scripts/plot_horizontal_field.py --input dataset/cm1out.nc --var prs \\
      --zh 1000 --time 400 --xy-limit 200

  # 生成时间序列视频
  python scripts/plot_horizontal_field.py --input dataset/cm1out.nc --var u \\
      --zh 2000 --save-video --fps 5
"""

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.plotting import plot_horizontal_slice, make_time_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CM1 水平场绘图工具（单帧/视频）"
    )
    parser.add_argument("--input", default="dataset/cm1out.nc", help="输入 NC 文件")
    parser.add_argument("--var", default="prs", help="变量名")
    parser.add_argument("--zh", type=float, default=1000.0, help="目标高度")
    parser.add_argument("--time", type=float, default=0, help="目标时间")

    parser.add_argument("--zh-dim", default="zh")
    parser.add_argument("--time-dim", default="time")
    parser.add_argument("--x-dim", default="xh")
    parser.add_argument("--y-dim", default="yh")

    parser.add_argument("--cmap", default=None, help="colormap 名称")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--xy-limit", type=float, default=None)

    parser.add_argument("--save-video", action="store_true", help="生成视频")
    parser.add_argument("--fps", type=int, default=5, help="视频帧率")
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--end-time", type=float, default=None)

    parser.add_argument("--figure-dir", default="figure")
    parser.add_argument("--figure-name", default=None)
    parser.add_argument("--video-dir", default="video")
    parser.add_argument("--out-name", default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--quiet", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    # 解析 cmap
    cmap = args.cmap
    if cmap is not None:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(cmap)

    if args.save_video:
        make_time_video(
            nc_file=args.input,
            var_name=args.var,
            target_zh=args.zh,
            zh_dim=args.zh_dim,
            time_dim=args.time_dim,
            x_dim=args.x_dim,
            y_dim=args.y_dim,
            cmap=cmap,
            vmin=args.vmin,
            vmax=args.vmax,
            xy_limit=args.xy_limit,
            start_time=args.start_time,
            end_time=args.end_time,
            fps=args.fps,
            video_dir=args.video_dir,
            out_name=args.out_name,
            dpi=args.dpi,
            verbose=not args.quiet,
        )
    else:
        plot_horizontal_slice(
            nc_file=args.input,
            var_name=args.var,
            target_zh=args.zh,
            target_time=args.time,
            zh_dim=args.zh_dim,
            time_dim=args.time_dim,
            x_dim=args.x_dim,
            y_dim=args.y_dim,
            cmap=cmap,
            vmin=args.vmin,
            vmax=args.vmax,
            xy_limit=args.xy_limit,
            figure_dir=args.figure_dir,
            figure_name=args.figure_name,
            dpi=args.dpi,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
