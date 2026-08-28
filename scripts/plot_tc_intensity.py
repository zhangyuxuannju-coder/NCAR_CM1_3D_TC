#!/usr/bin/env python
"""
绘制台风强度（中心最低气压）随时间变化图。

同时绘制 jet2 和 nojet 两个实验的中心最低气压演变，便于对比。

用法:
  python scripts/plot_tc_intensity.py
  python scripts/plot_tc_intensity.py --output output/figures/tc_intensity.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter


def extract_pmin_timeseries(nc_file: str, smooth_sigma: float = 2.0) -> tuple:
    """
    从 CM1 输出文件中提取中心最低气压时间序列。

    Parameters
    ----------
    nc_file : str
        CM1 输出 netCDF 文件路径。
    smooth_sigma : float
        高斯平滑 sigma（网格点数），用于减少噪声。默认 2.0。

    Returns
    -------
    time_h : np.ndarray
        时间（小时），shape (nt,)
    pmin_hpa : np.ndarray
        中心最低气压（hPa），shape (nt,)
    """
    with Dataset(nc_file, "r") as nc:
        time_s = np.asarray(nc.variables["time"][:], dtype=float)
        time_h = time_s / 3600.0
        psfc = np.asarray(nc.variables["psfc"][:], dtype=float)  # (nt, ny, nx)

    # psfc 单位通常是 Pa，转换为 hPa
    # 检查量级：典型海平面气压 ~100000 Pa → 1000 hPa
    pmin_per_step = []
    for t_idx in range(psfc.shape[0]):
        field = psfc[t_idx]
        # 高斯平滑后取最小值
        if smooth_sigma > 0:
            field_sm = gaussian_filter(field, sigma=smooth_sigma)
        else:
            field_sm = field
        pmin = np.nanmin(field_sm)
        pmin_per_step.append(pmin)

    pmin_arr = np.array(pmin_per_step)

    # 自动判断单位：若数值 > 10000 则为 Pa，转换为 hPa
    if np.nanmedian(pmin_arr) > 10000:
        pmin_arr = pmin_arr / 100.0

    return time_h, pmin_arr


def plot_intensity_comparison(
    jet2_file: str,
    nojet_file: str,
    output_png: str = "output/figures/tc_intensity.png",
    smooth_sigma: float = 2.0,
    jet_label: str = "JET",
    nojet_label: str = "CTRL",
) -> None:
    """绘制两个实验的中心最低气压对比图。"""
    print(f"读取 jet2: {jet2_file}")
    time_jet2, pmin_jet2 = extract_pmin_timeseries(jet2_file, smooth_sigma=smooth_sigma)

    print(f"读取 nojet: {nojet_file}")
    time_nojet, pmin_nojet = extract_pmin_timeseries(nojet_file, smooth_sigma=smooth_sigma)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(time_jet2, pmin_jet2, linewidth=1.8, color="#E74C3C", label=jet_label)
    ax.plot(time_nojet, pmin_nojet, linewidth=1.8, color="#3498DB", label=nojet_label)

    ax.set_xlabel("Time (h)", fontsize=13)
    ax.set_ylabel("Minimum Surface Pressure (hPa)", fontsize=13)
    ax.set_title("TC Intensity: Central Minimum Pressure", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12, frameon=True, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)

    # 标注最终强度
    for label, t_arr, p_arr, color, y_offset in [
        (jet_label, time_jet2, pmin_jet2, "#E74C3C", -15),
        (nojet_label, time_nojet, pmin_nojet, "#3498DB", 15),
    ]:
        ax.annotate(
            f"{label}: {p_arr[-1]:.1f} hPa",
            xy=(t_arr[-1], p_arr[-1]),
            xytext=(15, y_offset),
            textcoords="offset points",
            fontsize=10,
            color=color,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        )

    # 标注最大差值
    # 在共同时间范围内计算差值
    common_t = np.intersect1d(
        np.round(time_jet2, 6), np.round(time_nojet, 6)
    )
    if len(common_t) > 0:
        diff = np.abs(
            np.interp(common_t, time_jet2, pmin_jet2)
            - np.interp(common_t, time_nojet, pmin_nojet)
        )
        idx_max = np.argmax(diff)
        ax.annotate(
            f"Max Δ = {diff[idx_max]:.1f} hPa\nat t = {common_t[idx_max]:.1f} h",
            xy=(common_t[idx_max],
                (np.interp(common_t[idx_max], time_jet2, pmin_jet2)
                 + np.interp(common_t[idx_max], time_nojet, pmin_nojet)) / 2),
            fontsize=9,
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

    fig.tight_layout()

    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    print(f"[INFO] 图像已保存: {output_png}")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="绘制 TC 中心最低气压时间序列对比图"
    )
    parser.add_argument(
        "--jet2", default="/data/zhangyx/DATA/cm1out_22N_8o_jet.nc",
        help="jet2 实验 CM1 输出文件"
    )
    parser.add_argument(
        "--nojet", default="/data/zhangyx/DATA/cm1out_22N_nojet.nc",
        help="nojet 实验 CM1 输出文件"
    )
    parser.add_argument(
        "--output", default="output/figures/tc_intensity.png",
        help="输出图像路径"
    )
    parser.add_argument(
        "--smooth-sigma", type=float, default=2.0,
        help="高斯平滑 sigma（网格点）"
    )

    parser.add_argument("--jet-label", default="JET", help="有急流试验图例")
    parser.add_argument("--nojet-label", default="CTRL", help="对照试验图例")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not Path(args.jet2).exists():
        raise FileNotFoundError(f"jet2 文件不存在: {args.jet2}")
    if not Path(args.nojet).exists():
        raise FileNotFoundError(f"nojet 文件不存在: {args.nojet}")

    plot_intensity_comparison(
        jet2_file=args.jet2,
        nojet_file=args.nojet,
        output_png=args.output,
        smooth_sigma=args.smooth_sigma,
        jet_label=args.jet_label,
        nojet_label=args.nojet_label,
    )


if __name__ == "__main__":
    main()
