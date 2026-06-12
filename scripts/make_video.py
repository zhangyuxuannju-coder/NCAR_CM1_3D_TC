#!/usr/bin/env python
"""
PNG 图片序列合成 MP4 视频。

用法:
  python scripts/make_video.py --input-dir figure_az_avg --pattern "az_avg_*.png" \\
      --output az_avg_animation.mp4 --fps 15
"""

import argparse
import glob
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PNG → MP4 视频合成")
    parser.add_argument("--input-dir", default="figure_az_avg", help="图片目录")
    parser.add_argument("--pattern", default="*.png", help="图片 glob 模式")
    parser.add_argument("--output", default="animation.mp4", help="输出 MP4 文件")
    parser.add_argument("--fps", type=int, default=15, help="帧率")
    parser.add_argument("--sort", action="store_true", default=True,
                        help="按文件名排序（默认开启）")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    import cv2

    images = glob.glob(os.path.join(args.input_dir, args.pattern))
    if args.sort:
        images.sort()

    if not images:
        print(f"[ERROR] 在 {args.input_dir} 中未找到匹配 '{args.pattern}' 的图片")
        sys.exit(1)

    frame = cv2.imread(images[0])
    if frame is None:
        print(f"[ERROR] 无法读取图片: {images[0]}")
        sys.exit(1)

    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(args.output, fourcc, args.fps, (width, height))

    print(f"开始生成视频: {len(images)} 帧, {args.fps} fps...")
    for img_path in images:
        video.write(cv2.imread(img_path))

    video.release()
    print(f"[INFO] 视频已保存: {args.output}")


if __name__ == "__main__":
    main()
