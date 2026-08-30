#!/usr/bin/env python3
"""
绘制 200 hPa (z≈12 km) 高空急流/出流水平场图 + 动画。

数据源: data/cm1out_jet2.nc
高度: z ≈ 12,000 m (约 200 hPa)
X 范围: centerx ± 800 km
Y 范围: centery - 400 km ~ centery + 1400 km

填色: 径向风 ur (入流=蓝/出流=红)
标注: TC 中心 ●、急流轴线 ---

用法:
  python scripts/plot_jet_upper_outflow.py
"""

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import sys

# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════
INPUT_FILE = "/data/zhangyx/DATA/cm1out_22N_8o_jet.nc"
OUTPUT_DIR = Path("output/figures/jet_upper_outflow")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_Z_KM = 12.0          # 目标高度 (约 200 hPa)
X_WINDOW_KM = 800.0         # X 范围半宽
Y_WINDOW_LOW_KM = 400.0     # Y 范围下界
Y_WINDOW_HIGH_KM = 1400.0   # Y 范围上界
JET_OFFSET_KM = 888.0       # 急流轴线距 TC 中心距离

T_START_H = 0.0             # 起始时间
T_END_H = 120.0             # 结束时间

DPI = 120                   # 批量输出降低 dpi 加速


def destagger_u_to_scalar(u_stag: np.ndarray) -> np.ndarray:
    return 0.5 * (u_stag[..., :-1] + u_stag[..., 1:])


def destagger_v_to_scalar(v_stag: np.ndarray) -> np.ndarray:
    return 0.5 * (v_stag[..., :-1, :] + v_stag[..., 1:, :])


def compute_radial_wind(ua, va, xh, yh, xc, yc):
    X, Y = np.meshgrid(xh, yh)
    dx, dy = X - xc, Y - yc
    r = np.sqrt(dx**2 + dy**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ur = np.where(r > 0, (ua * dx + va * dy) / r, 0.0)
    return ur


def find_tc_center(psfc, xh, yh):
    iy_min, ix_min = np.unravel_index(np.nanargmin(psfc), psfc.shape)
    return float(xh[ix_min]), float(yh[iy_min]), ix_min, iy_min


def plot_single_frame(ds, time_idx, time_hours, xh, yh, zh, z_idx, z_actual,
                      global_ur_max, output_path):
    """绘制单帧：径向风填色 + TC中心 + 急流轴线（无矢量箭头）。"""

    # 读取交错风场并去交错
    u_stag = np.asarray(ds["u"].isel(time=time_idx, zh=z_idx), dtype=np.float64)
    v_stag = np.asarray(ds["v"].isel(time=time_idx, zh=z_idx), dtype=np.float64)
    ua = destagger_u_to_scalar(u_stag)
    va = destagger_v_to_scalar(v_stag)

    # TC 中心
    psfc = np.asarray(ds["psfc"].isel(time=time_idx), dtype=np.float64)
    xc, yc, ix_c, iy_c = find_tc_center(psfc, xh, yh)
    psfc_hpa = psfc[iy_c, ix_c] / 100.0

    # 裁剪域
    x_mask = (xh >= xc - X_WINDOW_KM) & (xh <= xc + X_WINDOW_KM)
    y_mask = (yh >= yc - Y_WINDOW_LOW_KM) & (yh <= yc + Y_WINDOW_HIGH_KM)
    xh_crop = xh[x_mask]
    yh_crop = yh[y_mask]
    ua_crop = ua[np.ix_(y_mask, x_mask)]
    va_crop = va[np.ix_(y_mask, x_mask)]

    # 径向风
    ur = compute_radial_wind(ua_crop, va_crop, xh_crop, yh_crop, xc, yc)

    # ── 绘图 ──
    fig, ax = plt.subplots(figsize=(12, 12), dpi=DPI)

    vmax = global_ur_max
    im = ax.contourf(xh_crop, yh_crop, ur, levels=41,
                     cmap="RdBu_r", vmin=-vmax, vmax=vmax, extend="both")
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Radial Wind (m/s)  Outflow → | ← Inflow", fontsize=12)

    # TC 中心
    ax.plot(xc, yc, marker='o', color='gold', markersize=12,
            markeredgecolor='black', markeredgewidth=2,
            zorder=5, label=f'TC ({xc:.0f}, {yc:.0f}) | psfc={psfc_hpa:.1f} hPa')

    # 急流轴线
    jet_y = yc + JET_OFFSET_KM
    if yh_crop.min() <= jet_y <= yh_crop.max():
        ax.axhline(y=jet_y, color='magenta', linewidth=2.5, linestyle='--',
                   alpha=0.9, label=f'Jet Axis (y={jet_y:.0f} km)')

    ax.set_xlabel("X (km)", fontsize=12)
    ax.set_ylabel("Y (km)", fontsize=12)
    ax.set_title(
        f"200 hPa Radial Wind  —  t = {time_hours:.0f} h  |  "
        f"z ≈ {z_actual:.1f} km",
        fontsize=14, fontweight="bold",
    )
    ax.set_aspect("equal")
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.15, linestyle="--")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    print(f"加载数据: {INPUT_FILE}")
    ds = xr.open_dataset(INPUT_FILE)

    xh = np.asarray(ds["xh"], dtype=np.float64)
    yh = np.asarray(ds["yh"], dtype=np.float64)
    zh = np.asarray(ds["zh"], dtype=np.float64)
    time_vals = np.asarray(ds["time"], dtype=np.float64)
    time_hours_arr = time_vals / 3600.0

    # 选高度层
    z_idx = int(np.argmin(np.abs(zh - TARGET_Z_KM)))
    z_actual = float(zh[z_idx])
    print(f"高度层: z_idx={z_idx}, z={z_actual:.2f} km")

    # 筛选时间索引
    time_indices = [i for i, th in enumerate(time_hours_arr)
                    if T_START_H <= th <= T_END_H]
    n_frames = len(time_indices)
    print(f"时间范围: {T_START_H:.0f}~{T_END_H:.0f} h, 共 {n_frames} 帧")

    # ── 第一遍扫描：确定全局色标范围（采样每6h避免IO过多）──
    print("\n扫描色标范围...")
    ur_samples = []
    sample_indices = time_indices[::6]  # 每6h采样
    for ti in sample_indices:
        u_stag = np.asarray(ds["u"].isel(time=ti, zh=z_idx), dtype=np.float64)
        v_stag = np.asarray(ds["v"].isel(time=ti, zh=z_idx), dtype=np.float64)
        ua = destagger_u_to_scalar(u_stag)
        va = destagger_v_to_scalar(v_stag)
        psfc = np.asarray(ds["psfc"].isel(time=ti), dtype=np.float64)
        xc, yc, _, _ = find_tc_center(psfc, xh, yh)
        x_mask = (xh >= xc - X_WINDOW_KM) & (xh <= xc + X_WINDOW_KM)
        y_mask = (yh >= yc - Y_WINDOW_LOW_KM) & (yh <= yc + Y_WINDOW_HIGH_KM)
        ur = compute_radial_wind(
            ua[np.ix_(y_mask, x_mask)], va[np.ix_(y_mask, x_mask)],
            xh[x_mask], yh[y_mask], xc, yc
        )
        ur_samples.append(np.nanpercentile(np.abs(ur), 98))

    global_ur_max = float(np.median(ur_samples))
    # 向上取整到 5 的倍数
    global_ur_max = np.ceil(global_ur_max / 5.0) * 5.0
    global_ur_max = max(global_ur_max, 10.0)
    print(f"全局色标范围: ±{global_ur_max:.0f} m/s (径向风)")

    # ── 逐帧输出 ──
    print(f"\n开始逐帧输出 ({n_frames} 帧)...")
    for idx, ti in enumerate(time_indices):
        th = time_hours_arr[ti]
        fname = f"frame_{idx:04d}_t{th:05.1f}h.png"
        out_path = OUTPUT_DIR / fname
        print(f"  [{idx+1:3d}/{n_frames}] t={th:5.1f}h", end="")
        plot_single_frame(ds, ti, th, xh, yh, zh, z_idx, z_actual,
                          global_ur_max, out_path)
        print(" ✓")

    ds.close()
    print(f"\n帧输出完成: {n_frames} 张 → {OUTPUT_DIR}")

    # ── 合成动画 ──
    print("\n合成 MP4 动画...")
    video_path = OUTPUT_DIR / "jet2_upper_outflow_0_120h.mp4"
    # ffmpeg: 10 fps, 高质量编码
    cmd = [
        "ffmpeg", "-y",
        "-framerate", "10",
        "-i", str(OUTPUT_DIR / "frame_%04d_t*.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # 确保宽高为偶数
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"动画已保存: {video_path}")
    else:
        # 尝试用通配符替代方案
        print(f"ffmpeg 通配符失败，尝试 concat 方式...")
        print(result.stderr[-500:])
        # 生成文件列表
        flist = sorted(OUTPUT_DIR.glob("frame_*.png"))
        list_path = OUTPUT_DIR / "frame_list.txt"
        with open(list_path, "w") as f:
            for fp in flist:
                f.write(f"file '{fp.name}'\n")
        cmd2 = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-framerate", "10",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "20",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            str(video_path),
        ]
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        if result2.returncode == 0:
            print(f"动画已保存: {video_path}")
        else:
            print(f"ffmpeg 失败:\n{result2.stderr[-800:]}")
            print(f"请手动合成: ffmpeg -framerate 10 -i {OUTPUT_DIR}/frame_%04d_t*.png ...")

    print("\n完成。")


if __name__ == "__main__":
    main()
