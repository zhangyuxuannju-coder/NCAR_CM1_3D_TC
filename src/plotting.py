"""
统一绘图模块 (Unified Plotting)

汇总所有可视化函数:
  - 水平场剖面 (horizontal_slice, make_time_video)
  - SE 解场 (plot_se_solution)
  - SE 强迫项 (plot_se_forcing)
  - 台风中心轨迹 (plot_center_tracks)
  - 3D 中心轨迹 (plot_centers_3d)
  - 径向动量收支分组图 (plot_budget_grouped_panels)
  - 单个诊断项 R-Z 图 (plot_single_diagnostic_rz)

统一配色方案和绘图风格。
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


# ==============================================================================
# 配色方案
# ==============================================================================

# 气压场配色 (深蓝→浅蓝→白→橙→深红)
PRESSURE_CMAP = LinearSegmentedColormap.from_list("pressure_cmap", [
    '#00008B', '#0000CD', '#4682B4', '#87CEFA', '#FFFFFF',
    '#FFD700', '#FFA500', '#FF6347', '#8B0000'
])

# SE 解场配色 (红蓝 diverging)
SE_CMAP = "RdBu_r"

# 风速配色
WIND_CMAP = "RdBu_r"


# ==============================================================================
# 辅助函数
# ==============================================================================

def find_index(arr: np.ndarray, target: float) -> int:
    """在一维数组中寻找最接近 target 的索引。"""
    return int(np.argmin(np.abs(arr - target)))


def time_to_index(time_arr: np.ndarray, key, default_idx: int) -> int:
    """将 key（索引或时间值或 None）转换为索引。"""
    if key is None:
        return default_idx
    if isinstance(key, int):
        if 0 <= key < len(time_arr):
            return int(key)
    return find_index(time_arr, key)


# ==============================================================================
# 水平场剖面图
# ==============================================================================

def plot_horizontal_slice(
    nc_file: str,
    var_name: str = "prs",
    target_zh: float = 1000.0,
    target_time: int = 0,
    zh_dim: str = "zh",
    time_dim: str = "time",
    x_dim: str = "xh",
    y_dim: str = "yh",
    cmap=None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    xy_limit: Optional[float] = None,
    figure_dir: str = "figure",
    figure_name: Optional[str] = None,
    dpi: int = 150,
    verbose: bool = True,
) -> None:
    """
    绘制单个时间步、单个高度层的水平剖面填色图。

    Parameters
    ----------
    nc_file : str
        netCDF 文件路径。
    var_name : str
        变量名，默认 'prs'。
    target_zh : float
        目标高度值。
    target_time : int or float
        目标时间索引或值。
    xy_limit : float or None
        绘图域 [-xy_limit, xy_limit]，None 为全域。
    """
    from netCDF4 import Dataset
    os = __import__("os")

    if cmap is None:
        cmap = PRESSURE_CMAP

    with Dataset(nc_file, "r") as nc:
        if var_name not in nc.variables:
            raise KeyError(f"变量 '{var_name}' 不在文件中。")

        var = nc.variables[var_name]
        zh_arr = nc.variables[zh_dim][:]
        time_arr = nc.variables[time_dim][:]
        xh = nc.variables[x_dim][:]
        yh = nc.variables[y_dim][:]

        zh_idx = find_index(zh_arr, target_zh)
        t_idx = time_to_index(time_arr, target_time, 0)

        if verbose:
            print(f"绘制: var={var_name}, t_idx={t_idx}, "
                  f"zh={zh_arr[zh_idx]:.2f}, time={time_arr[t_idx]:.2f}")

        # 读取切片
        sl = np.asarray(var[t_idx, zh_idx, :, :], dtype=float)

    if xy_limit is not None:
        x_mask = (xh >= -xy_limit) & (xh <= xy_limit)
        y_mask = (yh >= -xy_limit) & (yh <= xy_limit)
        xh = xh[x_mask]
        yh = yh[y_mask]
        sl = sl[np.ix_(y_mask, x_mask)]

    # 确定色阶
    if vmin is None:
        vmin = float(np.nanpercentile(sl, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(sl, 98))

    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
    im = ax.pcolormesh(xh, yh, sl, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading='auto')
    plt.colorbar(im, ax=ax, label=var_name, fraction=0.046)
    ax.set_xlabel(f"{x_dim} (km)")
    ax.set_ylabel(f"{y_dim} (km)")
    ax.set_title(f"{var_name} at z≈{zh_arr[zh_idx]:.1f} km, "
                 f"t={time_arr[t_idx]:.0f} s")
    ax.set_aspect("equal")

    os.makedirs(figure_dir, exist_ok=True)
    if figure_name is None:
        figure_name = f"{var_name}_t{t_idx:04d}_z{zh_idx:04d}.png"

    out_path = os.path.join(figure_dir, figure_name)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] 图片已保存: {out_path}")


def make_time_video(
    nc_file: str,
    var_name: str = "prs",
    target_zh: float = 1000.0,
    zh_dim: str = "zh",
    time_dim: str = "time",
    x_dim: str = "xh",
    y_dim: str = "yh",
    cmap=None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    xy_limit: Optional[float] = None,
    start_time=None,
    end_time=None,
    fps: int = 5,
    video_dir: str = "video",
    out_name: Optional[str] = None,
    dpi: int = 150,
    verbose: bool = True,
) -> None:
    """
    在固定高度层生成时间序列动画视频。

    需要系统安装 ffmpeg。
    """
    from netCDF4 import Dataset
    os = __import__("os")

    if cmap is None:
        cmap = PRESSURE_CMAP
    os.makedirs(video_dir, exist_ok=True)

    with Dataset(nc_file, "r") as nc:
        var = nc.variables[var_name]
        zh_arr = nc.variables[zh_dim][:]
        time_arr = nc.variables[time_dim][:]
        xh = nc.variables[x_dim][:]
        yh = nc.variables[y_dim][:]

        zh_idx = find_index(zh_arr, target_zh)
        t_start = time_to_index(time_arr, start_time, 0)
        t_end = time_to_index(time_arr, end_time, len(time_arr) - 1)
        t_indices = list(range(t_start, t_end + 1))

        if xy_limit is not None:
            x_mask = (xh >= -xy_limit) & (xh <= xy_limit)
            y_mask = (yh >= -xy_limit) & (yh <= xy_limit)
            xh_plot = xh[x_mask]
            yh_plot = yh[y_mask]
        else:
            x_mask = slice(None)
            y_mask = slice(None)
            xh_plot = xh
            yh_plot = yh

        # 确定色阶
        all_data = []
        for t_idx in t_indices[::max(1, len(t_indices)//20)]:
            sl = np.asarray(var[t_idx, zh_idx, :, :], dtype=float)
            all_data.append(sl[np.ix_(y_mask, x_mask)].ravel())
        all_flat = np.concatenate(all_data)
        if vmin is None:
            vmin = float(np.nanpercentile(all_flat, 2))
        if vmax is None:
            vmax = float(np.nanpercentile(all_flat, 98))

    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)

    def animate(frame_idx):
        ax.clear()
        with Dataset(nc_file, "r") as nc:
            sl = np.asarray(
                nc.variables[var_name][t_indices[frame_idx], zh_idx, :, :],
                dtype=float
            )
        sl = sl[np.ix_(y_mask, x_mask)]
        im = ax.pcolormesh(xh_plot, yh_plot, sl, cmap=cmap,
                           vmin=vmin, vmax=vmax, shading='auto')
        ax.set_title(f"{var_name} t={time_arr[t_indices[frame_idx]]:.0f}s")
        ax.set_aspect("equal")
        return [im]

    ani = animation.FuncAnimation(
        fig, animate, frames=len(t_indices), interval=1000//fps, blit=False
    )

    if out_name is None:
        out_name = f"{var_name}_z{target_zh:.0f}.mp4"
    out_path = os.path.join(video_dir, out_name)
    ani.save(out_path, fps=fps, writer="ffmpeg", dpi=dpi)
    plt.close(fig)
    print(f"[INFO] 视频已保存: {out_path}")


# ==============================================================================
# SE 解场图
# ==============================================================================

def plot_se_solution(
    r_km: np.ndarray,
    z_km: np.ndarray,
    psi_rz: np.ndarray,
    u_rz: np.ndarray,
    w_rz: np.ndarray,
    ut_rz: np.ndarray,
    out_png: Path,
    dpi: int = 150,
) -> None:
    """
    SE 解场三列图: ψ (流函数), U_se (径向次级环流), W_se (垂直次级环流)。

    叠加切向风 ut 等值线。
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), constrained_layout=True)

    # ψ
    ax = axes[0]
    vmax_psi = max(abs(np.nanmin(psi_rz)), abs(np.nanmax(psi_rz)), 1e-10)
    levels_psi = np.linspace(-vmax_psi, vmax_psi, 31)
    im = ax.contourf(r_km, z_km, psi_rz, levels=levels_psi,
                     cmap=SE_CMAP, extend="both")
    ax.contour(r_km, z_km, psi_rz, levels=levels_psi[::3],
               colors='k', alpha=0.4, linewidths=0.5)
    ax.contour(r_km, z_km, ut_rz, levels=11, colors='green',
               alpha=0.6, linewidths=0.8)
    plt.colorbar(im, ax=ax, fraction=0.046, label="ψ (kg m⁻¹ s⁻¹)")
    ax.set_title("Streamfunction ψ")
    ax.set_xlabel("Radius (km)")
    ax.set_ylabel("Height (km)")
    ax.set_ylim(0, 20)

    # U_se
    ax = axes[1]
    vmax_u = max(abs(np.nanmin(u_rz)), abs(np.nanmax(u_rz)), 0.1)
    levels_u = np.linspace(-vmax_u, vmax_u, 31)
    im = ax.contourf(r_km, z_km, u_rz, levels=levels_u,
                     cmap=SE_CMAP, extend="both")
    ax.contour(r_km, z_km, u_rz, levels=levels_u[::3],
               colors='k', alpha=0.4, linewidths=0.5)
    ax.contour(r_km, z_km, ut_rz, levels=11, colors='green',
               alpha=0.6, linewidths=0.8)
    plt.colorbar(im, ax=ax, fraction=0.046, label="U_se (m/s)")
    ax.set_title("Radial Secondary Circulation U_se")
    ax.set_xlabel("Radius (km)")
    ax.set_ylim(0, 20)

    # W_se
    ax = axes[2]
    w_max = float(np.nanpercentile(np.abs(w_rz), 99))
    levels_w = np.linspace(-w_max, w_max, 31)
    im = ax.contourf(r_km, z_km, w_rz, levels=levels_w,
                     cmap=SE_CMAP, extend="both")
    ax.contour(r_km, z_km, w_rz, levels=levels_w[::3],
               colors='k', alpha=0.4, linewidths=0.5)
    ax.contour(r_km, z_km, ut_rz, levels=11, colors='green',
               alpha=0.6, linewidths=0.8)
    plt.colorbar(im, ax=ax, fraction=0.046, label="W_se (m/s)")
    ax.set_title("Vertical Secondary Circulation W_se")
    ax.set_xlabel("Radius (km)")
    ax.set_ylim(0, 20)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] SE 解场图已保存: {out_png}")


# ==============================================================================
# SE 强迫项图
# ==============================================================================

def plot_se_forcing(
    r_km: np.ndarray,
    z_km: np.ndarray,
    forcing_total: np.ndarray,
    forcing_thermal: np.ndarray,
    forcing_momentum: np.ndarray,
    q_2d: np.ndarray,
    out_png: Path,
    forcing_evap: Optional[np.ndarray] = None,
    dpi: int = 150,
) -> None:
    """
    SE 强迫项图: 总强迫、热力强迫、动量强迫、Q 场（+可选蒸发面板）。

    Parameters
    ----------
    forcing_evap : np.ndarray or None
        如果提供，显示为第 5 列（蒸发冷却强迫项）。
    """
    n_cols = 5 if forcing_evap is not None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6),
                              constrained_layout=True)

    panels = [
        (forcing_total, "Total Forcing F", -1),
        (forcing_thermal, "Thermal Forcing", -1),
        (forcing_momentum, "Momentum Forcing", -1),
        (q_2d, "Q (K/s)", 0),
    ]
    if forcing_evap is not None:
        panels.append((forcing_evap, "Evap Cooling Forcing", -1))

    for ax, (field, title, vcenter) in zip(axes, panels):
        vmax_val = float(np.nanpercentile(np.abs(field), 99))
        vmin_val = -vmax_val if vcenter == -1 else float(np.nanmin(field))
        if vcenter == 0:
            vmax_val = float(np.nanmax(field))

        if vcenter == -1:
            norm = TwoSlopeNorm(vcenter=0, vmin=vmin_val, vmax=vmax_val)
            cmap = SE_CMAP
        else:
            norm = None
            cmap = "RdBu_r" if vmin_val < 0 < vmax_val else "viridis"

        levels = np.linspace(vmin_val, vmax_val, 31)
        im = ax.contourf(r_km, z_km, field, levels=levels,
                         cmap=cmap, norm=norm, extend="both")
        ax.contour(r_km, z_km, field, levels=levels[::3],
                   colors='k', alpha=0.4, linewidths=0.5)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(title)
        ax.set_xlabel("Radius (km)")
        ax.set_ylabel("Height (km)")
        ax.set_ylim(0, 20)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] SE 强迫项图已保存: {out_png}")


# ==============================================================================
# 台风中心轨迹图
# ==============================================================================

def plot_center_tracks(
    df,
    out_png: Path,
    title: str = "Typhoon Center Tracks",
    dpi: int = 150,
) -> None:
    """绘制台风中心轨迹图。"""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)

    if "x" in df.columns and "y" in df.columns:
        ax.plot(df["x"], df["y"], 'b.-', linewidth=0.8, markersize=2)
        ax.plot(df["x"].iloc[0], df["y"].iloc[0], 'go', markersize=8,
                label='Start')
        ax.plot(df["x"].iloc[-1], df["y"].iloc[-1], 'ro', markersize=8,
                label='End')

    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# 3D 中心轨迹图
# ==============================================================================

def plot_centers_3d(
    centers: List[Dict[str, object]],
    out_png: Optional[Path] = None,
    title: str = "Typhoon Center 3D Profile",
    dpi: int = 150,
) -> None:
    """绘制台风中心随高度的 3D 轨迹图。"""
    xs = [float(c.get("x", np.nan)) for c in centers]
    ys = [float(c.get("y", np.nan)) for c in centers]
    zs = [float(c.get("zh_value", c.get("z", np.nan))) for c in centers]

    valid = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
    xs = np.array(xs)[valid]
    ys = np.array(ys)[valid]
    zs = np.array(zs)[valid]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=dpi)

    # x-z 投影
    ax = axes[0]
    ax.plot(xs, zs, 'b.-', linewidth=1.5)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("Height (km)")
    ax.set_title(f"{title} - x-z projection")
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # y-z 投影
    ax = axes[1]
    ax.plot(ys, zs, 'r.-', linewidth=1.5)
    ax.set_xlabel("y (km)")
    ax.set_ylabel("Height (km)")
    ax.set_title(f"{title} - y-z projection")
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    fig.suptitle(title, fontweight="bold")

    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ==============================================================================
# 预算项分组面板图
# ==============================================================================

def plot_budget_grouped_panels(
    r_km: np.ndarray,
    z_km: np.ndarray,
    terms: Dict[str, np.ndarray],
    out_png: Path,
    z_target: Optional[float] = None,
    term_groups: Optional[Dict[str, List[str]]] = None,
    title: str = "Momentum Budget Diagnostics",
    dpi: int = 150,
) -> None:
    """
    绘制多个预算项的 R-Z 填色图面板（或径向剖面图）。

    Parameters
    ----------
    terms : dict
        {term_name: 2d_array (nz, nr)}。
    z_target : float or None
        如果指定，改为绘制该高度层的径向剖面线图。
    term_groups : dict or None
        分组定义 {group_name: [term_names]}，用于分色绘图。
    """
    if z_target is not None:
        # 径向剖面模式
        iz = find_index(z_km, z_target)
        fig, axes = plt.subplots(
            (len(term_groups) + 1) // 2, 2,
            figsize=(16, 4 * ((len(term_groups) + 1) // 2)),
            constrained_layout=True
        )
        axes = axes.flatten()

        for ax, (group_name, group_terms) in zip(axes, term_groups.items()):
            for tname in group_terms:
                if tname in terms:
                    ax.plot(r_km, terms[tname][iz, :], label=tname, linewidth=1.5)
            ax.axhline(y=0, color='grey', linestyle=':', alpha=0.5)
            ax.set_xlabel("Radius (km)")
            ax.set_ylabel("m s⁻²")
            ax.set_title(f"{group_name} at z≈{z_km[iz]:.1f} km")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        for ax in axes[len(term_groups):]:
            ax.set_visible(False)
    else:
        # R-Z 填色图模式
        n_terms = len(terms)
        n_cols = min(3, n_terms)
        n_rows = (n_terms + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(6 * n_cols, 5 * n_rows),
                                  constrained_layout=True)
        if n_terms == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for ax, (tname, field) in zip(axes, terms.items()):
            # 安全计算色阶范围
            valid = field[np.isfinite(field)]
            if len(valid) == 0:
                vmax = 1.0
            else:
                vmax = float(np.nanpercentile(np.abs(valid), 99))
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0
            levels = np.linspace(-vmax, vmax, 31)
            if len(levels) < 2 or levels[0] >= levels[-1]:
                levels = np.linspace(-vmax, vmax, 31)
            im = ax.contourf(r_km, z_km, field, levels=levels,
                             cmap=SE_CMAP, extend="both")
            ax.contour(r_km, z_km, field, levels=levels[::3],
                       colors='k', alpha=0.3, linewidths=0.5)
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(tname, fontsize=10)
            ax.set_xlabel("Radius (km)")
            ax.set_ylabel("Height (km)")
            ax.set_ylim(0, 20)

        for ax in axes[n_terms:]:
            ax.set_visible(False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] 预算诊断图已保存: {out_png}")


# ==============================================================================
# 单个诊断项 R-Z 图
# ==============================================================================

def plot_single_diagnostic_rz(
    r_km: np.ndarray,
    z_km: np.ndarray,
    field: np.ndarray,
    out_png: Path,
    var_name: str = "diagnostic",
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    dpi: int = 150,
) -> None:
    """
    绘制单个诊断项的 R-Z 填色图。

    这是最基础的单图绘制函数，适用于快速查看任意二维场。
    """
    fig, ax = plt.subplots(figsize=(10, 7), dpi=dpi)

    if vmin is None:
        vmax_abs = float(np.nanpercentile(np.abs(field), 99))
        vmin, vmax = -vmax_abs, vmax_abs

    levels = np.linspace(vmin, vmax, 31)
    norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax) \
        if vmin < 0 < vmax else None

    im = ax.contourf(r_km, z_km, field, levels=levels,
                     cmap=cmap, norm=norm, extend="both")
    ax.contour(r_km, z_km, field, levels=levels[::3],
               colors='k', alpha=0.4, linewidths=0.5)
    plt.colorbar(im, ax=ax, fraction=0.046, label=var_name)

    ax.set_xlabel("Radius (km)")
    ax.set_ylabel("Height (km)")
    ax.set_title(title or var_name)
    ax.set_ylim(0, min(20, float(np.nanmax(z_km))))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] 诊断图已保存: {out_png}")
