#!/usr/bin/env python
"""
单页诊断图绘制工具 — 忠实复现 radial_diagnostic_singlepage.ipynb。

功能:
  1. 径向动量收支单页图 (--panel radial)
  2. 切向动量收支单页图 (--panel tangential)
  3. 切向重组诊断图 (--panel tangential_regrouped)
  4. 自定义线性组合单窗图 (--panel combo)
  5. 涡度组合单窗图 (--panel vortex_combo)

用法:
  # 径向收支单页图 (42h-74h 时间段平均)
  python scripts/plot_singlepage_diagnostics.py --panel radial \\
      --input dataset/wind_Thompson.nc --mode time_range \\
      --start-hour 42 --end-hour 74 --output output/figures/radial_diag.png

  # 切向收支单页图
  python scripts/plot_singlepage_diagnostics.py --panel tangential \\
      --input dataset/wind_Thompson.nc --mode time_range \\
      --start-hour 42 --end-hour 74 --output output/figures/tangential_diag.png

  # 自定义组合: U_magf - U_mr
  python scripts/plot_singlepage_diagnostics.py --panel combo \\
      --input dataset/wind_Thompson.nc --mode time_range \\
      --start-hour 42 --end-hour 74 \\
      --combo-terms "1.0,U_magf -1.0,U_mr"
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, SymLogNorm
from matplotlib.cm import ScalarMappable
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter1d

# ==============================================================================
# 辅助函数
# ==============================================================================

def nearest_time_index(time_s: np.ndarray, target_s: float) -> int:
    return int(np.argmin(np.abs(time_s - target_s)))


def symmetric_levels_from_data(
    arrays: list, n_levels: int, floor: float, clip_percentile: float = 99.0
) -> Tuple[np.ndarray, float]:
    if len(arrays) == 0:
        vmax = floor
    else:
        stack = np.stack(arrays, axis=0)
        vmax = np.nanpercentile(np.abs(stack), clip_percentile)
        vmax = max(float(vmax), floor)
    levels = np.linspace(-vmax, vmax, n_levels)
    return levels, vmax


def make_symlog_levels(vmax: float, linthresh: float, n_levels: int) -> np.ndarray:
    n_half = max((n_levels - 1) // 2, 4)
    n_linear = max(3, n_half // 3)
    n_log = max(2, n_half - n_linear)
    pos_linear = np.linspace(0.0, linthresh, n_linear, endpoint=False)
    pos_log = np.geomspace(max(linthresh, 1e-14), vmax, n_log + 1)
    pos = np.unique(np.concatenate([pos_linear, pos_log]))
    levels = np.concatenate([-pos[::-1], [0.0], pos])
    levels = np.unique(levels)
    if levels.size < 7:
        levels = np.linspace(-vmax, vmax, max(n_levels, 9))
    return levels


def get_var_2d(nc: Dataset, vname: str, tidx: int,
               z_mask: np.ndarray, r_mask: np.ndarray) -> np.ndarray:
    return np.asarray(nc.variables[vname][tidx, z_mask, :][:, r_mask], dtype=float)


def get_var_2d_mean(nc: Dataset, vname: str, time_mask: np.ndarray,
                    z_mask: np.ndarray, r_mask: np.ndarray) -> np.ndarray:
    data = np.asarray(nc.variables[vname][time_mask, :, :], dtype=float)
    return np.nanmean(data[:, z_mask, :][:, :, r_mask], axis=0)


def shorten_text(text: str, max_len: int) -> str:
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "..."


def smooth_2d_field(field_2d: np.ndarray, sigma: float) -> np.ndarray:
    """沿 R 方向 (axis=1) 做高斯平滑。"""
    return gaussian_filter1d(field_2d, sigma=sigma, axis=1)


def is_smooth_target(name: str, targets: list) -> bool:
    for t in targets:
        if name == t or name == f"{t}_avg":
            return True
    return False


# ==============================================================================
# 径向动量收支单页图
# ==============================================================================

def plot_radial_diagnostics(
    nc_file: str,
    plot_mode: str = "time_range",
    target_hour: float = 60.0,
    start_hour: float = 42.0,
    end_hour: float = 74.0,
    max_r_km: float = 300.0,
    max_z_km: float = 20.0,
    n_levels_ur: int = 17,
    n_levels_diag: int = 17,
    diag_color_mode: str = "symlog",
    small_clip_percentile: float = 97.0,
    large_clip_percentile: float = 98.5,
    small_linthresh_ratio: float = 0.08,
    large_linthresh_ratio: float = 0.08,
    ncols: int = 4,
    plot_non_raw_only: bool = True,
    apply_radial_smoothing: bool = True,
    smooth_sigma: float = 2.0,
    output_png: Optional[str] = None,
) -> None:
    """
    径向动量收支单页多面板诊断图。
    """
    smooth_targets = ["pgrad_mean", "pgrad_eddy", "U_magf", "U_eagf"]

    preferred_diag_all = [
        "U_mr", "U_eh", "U_mv", "U_ev",
        "U_magf", "U_eagf", "U_dh", "U_dv", "ramp",
        "U_mr_raw", "U_eh_raw", "U_mv_raw", "U_ev_raw",
        "U_magf_raw", "U_eagf_raw", "U_dh_raw", "U_dv_raw", "ramp_raw",
        "coriolis", "pgrad_mean", "pgrad_eddy", "curv_mean", "curv_eddy",
        "br_total_raw", "tendency_model_raw", "tendency_model_adjusted",
        "residual_raw", "residual_after_allocation",
    ]
    if plot_non_raw_only:
        preferred_diag = [v for v in preferred_diag_all if not v.endswith("_raw")]
    else:
        preferred_diag = list(preferred_diag_all)

    with Dataset(nc_file, "r") as nc:
        time_s = np.asarray(nc.variables["time"][:], dtype=float)
        time_h = time_s / 3600.0
        r_arr = np.asarray(nc.variables["r"][:], dtype=float)
        z_arr = np.asarray(
            nc.variables["z"][:] if "z" in nc.variables
            else nc.variables["zh"][:], dtype=float
        )
        r_mask = r_arr <= max_r_km
        z_mask = z_arr <= max_z_km
        r_plot = r_arr[r_mask]
        z_plot = z_arr[z_mask]

        var_names = list(nc.variables.keys())
        diag_vars = [v for v in preferred_diag if v in var_names]
        if len(diag_vars) == 0:
            raise KeyError("结果文件中未找到可绘制的诊断项变量。")

        field_data = {}
        field_data_unsmoothed = {}
        field_units = {}
        field_long_name = {}

        if plot_mode == "time_point":
            target_s = target_hour * 3600.0
            t_idx = nearest_time_index(time_s, target_s)
            field_data["ur"] = get_var_2d(nc, "ur", t_idx, z_mask, r_mask)
            field_units["ur"] = getattr(nc.variables["ur"], "units", "")
            field_long_name["ur"] = getattr(nc.variables["ur"], "long_name", "ur")

            for v in diag_vars:
                data_raw = get_var_2d(nc, v, t_idx, z_mask, r_mask)
                field_data_unsmoothed[v] = data_raw
                if apply_radial_smoothing and is_smooth_target(v, smooth_targets):
                    field_data[v] = smooth_2d_field(data_raw, smooth_sigma)
                    lname_prefix = "(Smoothed) "
                else:
                    field_data[v] = data_raw
                    lname_prefix = ""
                field_units[v] = getattr(nc.variables[v], "units", "")
                field_long_name[v] = lname_prefix + getattr(nc.variables[v], "long_name", v)

            sum_weights = {
                "U_mr": -1, "U_eh": -1, "U_mv": -1, "U_ev": -1,
                "U_magf": 1, "U_eagf": 1, "U_dh": 1, "U_dv": 1, "ramp": 1
            }
            if not plot_non_raw_only:
                sum_weights = {k + "_raw": v for k, v in sum_weights.items()}
            valid_keys = [k for k in sum_weights if k in field_data_unsmoothed]
            if valid_keys:
                field_data["diag_sum"] = np.sum(
                    np.stack([field_data_unsmoothed[k] * sum_weights[k] for k in valid_keys], axis=0), axis=0)
            else:
                field_data["diag_sum"] = np.zeros_like(field_data["ur"])
            field_units["diag_sum"] = field_units.get(valid_keys[0], "") if valid_keys else ""
            field_long_name["diag_sum"] = "sum of independent forcing terms (unsmoothed)"
            panel_order = ["ur", "diag_sum"] + diag_vars
            title_mode = f"Time-point diagnostics | target={target_hour:.2f} h, actual={time_h[t_idx]:.2f} h (idx={t_idx})"

        elif plot_mode == "time_range":
            if end_hour < start_hour:
                raise ValueError("end_hour 必须 >= start_hour")
            start_s = start_hour * 3600.0
            end_s = end_hour * 3600.0
            t0 = nearest_time_index(time_s, start_s)
            t1 = nearest_time_index(time_s, end_s)
            if t1 < t0:
                t0, t1 = t1, t0
            time_mask = np.zeros_like(time_s, dtype=bool)
            time_mask[t0:t1 + 1] = True

            field_data["ur_start"] = get_var_2d(nc, "ur", t0, z_mask, r_mask)
            field_data["ur_end"] = get_var_2d(nc, "ur", t1, z_mask, r_mask)
            field_units["ur_start"] = getattr(nc.variables["ur"], "units", "")
            field_units["ur_end"] = getattr(nc.variables["ur"], "units", "")
            field_long_name["ur_start"] = "radial wind at start time"
            field_long_name["ur_end"] = "radial wind at end time"

            delta_t_s = end_s - start_s
            if delta_t_s > 0:
                field_data["ur_tendency"] = (field_data["ur_end"] - field_data["ur_start"]) / delta_t_s
            else:
                field_data["ur_tendency"] = np.zeros_like(field_data["ur_start"])
            field_units["ur_tendency"] = "m s-2"
            field_long_name["ur_tendency"] = "\u2202u/\u2202t (end-start)/dt"

            diag_avg_names = []
            for v in diag_vars:
                v_avg_name = f"{v}_avg"
                diag_avg_names.append(v_avg_name)
                data_raw = get_var_2d_mean(nc, v, time_mask, z_mask, r_mask)
                field_data_unsmoothed[v_avg_name] = data_raw
                if apply_radial_smoothing and is_smooth_target(v_avg_name, smooth_targets):
                    field_data[v_avg_name] = smooth_2d_field(data_raw, smooth_sigma)
                    lname_prefix = "(Smoothed) "
                else:
                    field_data[v_avg_name] = data_raw
                    lname_prefix = ""
                field_units[v_avg_name] = getattr(nc.variables[v], "units", "")
                field_long_name[v_avg_name] = lname_prefix + f"time-mean of {getattr(nc.variables[v], 'long_name', v)}"

            sum_weights = {
                "U_mr": -1, "U_eh": -1, "U_mv": -1, "U_ev": -1,
                "U_magf": 1, "U_eagf": 1, "U_dh": 1, "U_dv": 1, "ramp": 1
            }
            if not plot_non_raw_only:
                sum_weights = {k + "_raw": v for k, v in sum_weights.items()}
            valid_keys_avg = [f"{k}_avg" for k in sum_weights if f"{k}_avg" in field_data_unsmoothed]
            if valid_keys_avg:
                field_data["diag_sum_avg"] = np.sum(
                    np.stack([field_data_unsmoothed[k] * sum_weights[k.replace("_avg", "")]
                              for k in valid_keys_avg], axis=0), axis=0)
            else:
                field_data["diag_sum_avg"] = np.zeros_like(field_data["ur_start"])
            field_units["diag_sum_avg"] = field_units.get(valid_keys_avg[0], "") if valid_keys_avg else ""
            field_long_name["diag_sum_avg"] = "time-mean sum of independent forcing terms (unsmoothed)"
            panel_order = ["ur_start", "ur_end", "ur_tendency", "diag_sum_avg"] + diag_avg_names
            title_mode = (f"Time-range mean diagnostics | start={time_h[t0]:.2f} h, "
                          f"end={time_h[t1]:.2f} h, N={np.sum(time_mask)}")
        else:
            raise ValueError("plot_mode 只能是 'time_point' 或 'time_range'")

    # --- 统一色阶 ---
    ur_keys = [k for k in panel_order if k.startswith("ur") and k != "ur_tendency"]
    large_key_set = {"diag_sum", "diag_sum_avg", "pgrad_mean", "pgrad_mean_avg",
                     "curv_mean", "curv_mean_avg", "ur_tendency"}
    large_diag_keys = [k for k in panel_order if k in large_key_set]
    small_diag_keys = [k for k in panel_order if (k not in ur_keys and k not in large_diag_keys)]

    ur_levels, ur_abs = symmetric_levels_from_data(
        arrays=[field_data[k] for k in ur_keys], n_levels=n_levels_ur, floor=1e-8, clip_percentile=99.0)
    small_levels, small_abs = symmetric_levels_from_data(
        arrays=[field_data[k] for k in small_diag_keys], n_levels=n_levels_diag,
        floor=1e-12, clip_percentile=small_clip_percentile)
    large_levels, large_abs = symmetric_levels_from_data(
        arrays=[field_data[k] for k in large_diag_keys], n_levels=n_levels_diag,
        floor=1e-12, clip_percentile=large_clip_percentile)

    ur_norm = BoundaryNorm(ur_levels, ncolors=256, clip=True)
    if diag_color_mode == "symlog":
        small_linthresh = max(small_abs * small_linthresh_ratio, 1e-12)
        large_linthresh = max(large_abs * large_linthresh_ratio, 1e-12)
        small_levels = make_symlog_levels(small_abs, small_linthresh, n_levels_diag)
        large_levels = make_symlog_levels(large_abs, large_linthresh, n_levels_diag)
        small_norm = SymLogNorm(linthresh=small_linthresh, vmin=-small_abs, vmax=small_abs, base=10)
        large_norm = SymLogNorm(linthresh=large_linthresh, vmin=-large_abs, vmax=large_abs, base=10)
    else:
        small_norm = BoundaryNorm(small_levels, ncolors=256, clip=True)
        large_norm = BoundaryNorm(large_levels, ncolors=256, clip=True)

    print(f"ur 色阶范围: [{-ur_abs:.3f}, {ur_abs:.3f}]")
    print(f"普通诊断项色阶范围: [{-small_abs:.6f}, {small_abs:.6f}] (p{small_clip_percentile})")
    print(f"大项色阶范围: [{-large_abs:.6f}, {large_abs:.6f}] (p{large_clip_percentile})")
    if apply_radial_smoothing:
        print(f"平滑: {smooth_targets}, sigma={smooth_sigma}")

    # --- 绘图 ---
    R, Z = np.meshgrid(r_plot, z_plot)
    n_panels = len(panel_order)
    nrows = int(np.ceil(n_panels / ncols))
    fig_h = 4.2 * nrows + 4.0

    fig, axes = plt.subplots(nrows, ncols, figsize=(19, fig_h), sharex=True, sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for i, key in enumerate(panel_order):
        ax = axes_flat[i]
        data = field_data[key]
        if key in ur_keys:
            levels, norm = ur_levels, ur_norm
        elif key in large_diag_keys:
            levels, norm = large_levels, large_norm
        else:
            levels, norm = small_levels, small_norm

        ax.contourf(R, Z, data, levels=levels, cmap="RdBu_r", norm=norm, extend="both")
        contour_levels = levels[::2]
        contour_levels = contour_levels[~np.isclose(contour_levels, 0.0, atol=1e-15)]
        if contour_levels.size > 0:
            cs = ax.contour(R, Z, data, levels=contour_levels, colors="k", linewidths=0.6, alpha=0.7)
            ax.clabel(cs, inline=True, fontsize=7, fmt="%.2g")

        unit = field_units.get(key, "")
        lname = field_long_name.get(key, key)
        short_key = shorten_text(key, 22)
        short_lname = shorten_text(lname, 44)
        title1 = f"{short_key} ({unit})" if unit else short_key
        ax.set_title(f"{title1}\n{short_lname}", fontsize=8.5, pad=4)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.set_xlim(0, max_r_km)
        ax.set_ylim(0, max_z_km)

    for i, ax in enumerate(axes_flat):
        if i >= n_panels:
            ax.axis("off")
            continue
        if i % ncols == 0:
            ax.set_ylabel("Height (km)", fontsize=10)
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Radius (km)", fontsize=10)

    fig.suptitle("Radial Momentum Diagnostics (Single Page)\n" + title_mode,
                 fontsize=15, y=0.992)
    fig.subplots_adjust(left=0.055, right=0.975, bottom=0.20, top=0.93,
                        wspace=0.18, hspace=0.40)

    info_lines = [
        f"mode: {plot_mode}", f"file: {nc_file}",
        f"radius: 0-{max_r_km:.0f} km", f"height: 0-{max_z_km:.0f} km",
        f"plot_non_raw_only: {plot_non_raw_only}", f"diag_color_mode: {diag_color_mode}",
        f"apply_radial_smoothing: {apply_radial_smoothing} (sigma={smooth_sigma})",
        f"normal diag panels: {len(small_diag_keys)}", f"large diag panels: {len(large_diag_keys)}",
    ]
    fig.text(0.055, 0.145, "\n".join(info_lines), ha="left", va="top", fontsize=9)

    # 三组 colorbar
    sm_ur = ScalarMappable(norm=ur_norm, cmap="RdBu_r"); sm_ur.set_array([])
    sm_small = ScalarMappable(norm=small_norm, cmap="RdBu_r"); sm_small.set_array([])
    sm_large = ScalarMappable(norm=large_norm, cmap="RdBu_r"); sm_large.set_array([])

    cax1 = fig.add_axes([0.58, 0.12, 0.37, 0.013])
    cax2 = fig.add_axes([0.58, 0.085, 0.37, 0.013])
    cax3 = fig.add_axes([0.58, 0.05, 0.37, 0.013])

    cb1 = fig.colorbar(sm_ur, cax=cax1, orientation="horizontal", ticks=ur_levels[::2])
    cb1.set_label("radial wind panels", fontsize=9)
    cb2 = fig.colorbar(sm_small, cax=cax2, orientation="horizontal")
    cb2.set_label("normal diagnostic panels", fontsize=9)
    cb3 = fig.colorbar(sm_large, cax=cax3, orientation="horizontal")
    cb3.set_label("large diagnostic panels: sum + pgrad_mean + curv_mean + ur_tendency", fontsize=9)

    if output_png:
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=150, bbox_inches="tight")
        print(f"[INFO] 径向诊断图已保存: {output_png}")
    else:
        plt.show()
    plt.close(fig)


# ==============================================================================
# 切向动量收支单页图
# ==============================================================================

def plot_tangential_diagnostics(
    nc_file: str, plot_mode: str = "time_range",
    target_hour: float = 60.0, start_hour: float = 42.0, end_hour: float = 74.0,
    max_r_km: float = 300.0, max_z_km: float = 20.0,
    n_levels_ur: int = 17, n_levels_diag: int = 17,
    diag_color_mode: str = "symlog",
    small_clip_percentile: float = 97.0, large_clip_percentile: float = 98.5,
    small_linthresh_ratio: float = 0.08, large_linthresh_ratio: float = 0.08,
    ncols: int = 4, plot_non_raw_only: bool = True,
    apply_radial_smoothing: bool = True, smooth_sigma: float = 2.0,
    smooth_v_sum_panel: bool = False,
    output_png: Optional[str] = None,
) -> None:
    """切向动量收支单页多面板诊断图。"""
    smooth_targets = ["pgrad_mean", "pgrad_eddy", "U_magf", "U_eagf"]
    smooth_targets_t = []
    for target in smooth_targets:
        if target in ("pgrad_mean", "pgrad_eddy"):
            smooth_targets_t.append("pgrad_t")
        elif target == "U_magf":
            smooth_targets_t.append("V_magf")
        elif target == "U_eagf":
            smooth_targets_t.append("V_eagf")
        else:
            smooth_targets_t.append(target)

    with Dataset(nc_file, "r") as nc:
        time_s = np.asarray(nc.variables["time"][:], dtype=float)
        time_h = time_s / 3600.0
        r_arr = np.asarray(nc.variables["r"][:], dtype=float)
        z_arr = np.asarray(nc.variables["z"][:] if "z" in nc.variables else nc.variables["zh"][:], dtype=float)
        r_mask = r_arr <= max_r_km
        z_mask = z_arr <= max_z_km
        r_plot = r_arr[r_mask]
        z_plot = z_arr[z_mask]

        preferred_diag_t_all = [
            "V_mr", "V_eh", "V_mv", "V_ev", "V_magf", "V_eagf", "V_dh", "V_dv", "tramp",
            "V_mr_raw", "V_eh_raw", "V_mv_raw", "V_ev_raw",
            "V_magf_raw", "V_eagf_raw", "V_dh_raw", "V_dv_raw", "tramp_raw",
            "coriolis_t", "pgrad_t", "vcurv_mean", "vcurv_eddy",
            "bt_total_raw", "tendency_t_model_raw", "tendency_t_model_adjusted",
            "residual_t_raw", "residual_t_after_allocation",
        ]
        if plot_non_raw_only:
            preferred_diag_t = [v for v in preferred_diag_t_all if not v.endswith("_raw")]
        else:
            preferred_diag_t = list(preferred_diag_t_all)
        diag_vars_t = [v for v in preferred_diag_t if v in nc.variables]
        if len(diag_vars_t) == 0:
            raise KeyError("未找到切向诊断项变量。")

        field_data_t = {}
        field_data_t_unsmoothed = {}
        field_units_t = {}
        field_long_name_t = {}

        if plot_mode == "time_point":
            target_s = target_hour * 3600.0
            t_idx = nearest_time_index(time_s, target_s)
            field_data_t["vt"] = get_var_2d(nc, "ut", t_idx, z_mask, r_mask)
            field_units_t["vt"] = getattr(nc.variables["ut"], "units", "")
            field_long_name_t["vt"] = getattr(nc.variables["ut"], "long_name", "ut")

            for v in diag_vars_t:
                data_raw = get_var_2d(nc, v, t_idx, z_mask, r_mask)
                field_data_t_unsmoothed[v] = data_raw
                if apply_radial_smoothing and is_smooth_target(v, smooth_targets_t):
                    field_data_t[v] = smooth_2d_field(data_raw, smooth_sigma)
                    lname_prefix = "(Smoothed) "
                else:
                    field_data_t[v] = data_raw
                    lname_prefix = ""
                field_units_t[v] = getattr(nc.variables[v], "units", "")
                field_long_name_t[v] = lname_prefix + getattr(nc.variables[v], "long_name", v)

            sum_weights_t = {
                "V_mr": -1, "V_eh": -1, "V_mv": -1, "V_ev": -1,
                "V_magf": 1, "V_eagf": 1, "V_dh": 1, "V_dv": 1, "tramp": 1
            }
            if not plot_non_raw_only:
                sum_weights_t = {k + "_raw": v for k, v in sum_weights_t.items()}
            valid_keys_t = [k for k in sum_weights_t if k in field_data_t_unsmoothed]
            if valid_keys_t:
                field_data_t["diag_t_sum"] = np.sum(
                    np.stack([field_data_t_unsmoothed[k] * sum_weights_t[k] for k in valid_keys_t], axis=0), axis=0)
            else:
                field_data_t["diag_t_sum"] = np.zeros_like(field_data_t["vt"])
            field_long_name_t["diag_t_sum"] = "sum of tangential independent forcing terms (unsmoothed)"
            field_units_t["diag_t_sum"] = field_units_t.get(valid_keys_t[0], "") if valid_keys_t else ""
            panel_order_t = ["vt", "diag_t_sum"] + diag_vars_t
            title_mode_t = f"Tangential diagnostics | target={target_hour:.2f} h, actual={time_h[t_idx]:.2f} h"

        elif plot_mode == "time_range":
            if end_hour < start_hour:
                raise ValueError("end_hour >= start_hour")
            start_s = start_hour * 3600.0; end_s = end_hour * 3600.0
            t0 = nearest_time_index(time_s, start_s); t1 = nearest_time_index(time_s, end_s)
            if t1 < t0: t0, t1 = t1, t0
            time_mask = np.zeros_like(time_s, dtype=bool)
            time_mask[t0:t1 + 1] = True

            field_data_t["vt_start"] = get_var_2d(nc, "ut", t0, z_mask, r_mask)
            field_data_t["vt_end"] = get_var_2d(nc, "ut", t1, z_mask, r_mask)
            field_units_t["vt_start"] = getattr(nc.variables["ut"], "units", "")
            field_units_t["vt_end"] = getattr(nc.variables["ut"], "units", "")
            field_long_name_t["vt_start"] = "tangential wind at start time"
            field_long_name_t["vt_end"] = "tangential wind at end time"

            delta_t_s = end_s - start_s
            if delta_t_s > 0:
                field_data_t["vt_tendency"] = (field_data_t["vt_end"] - field_data_t["vt_start"]) / delta_t_s
            else:
                field_data_t["vt_tendency"] = np.zeros_like(field_data_t["vt_start"])
            field_units_t["vt_tendency"] = "m s-2"
            field_long_name_t["vt_tendency"] = "\u2202v_t/\u2202t (end-start)/dt"

            diag_t_avg_names = []
            for v in diag_vars_t:
                v_avg_name = f"{v}_avg"
                diag_t_avg_names.append(v_avg_name)
                data_raw = get_var_2d_mean(nc, v, time_mask, z_mask, r_mask)
                field_data_t_unsmoothed[v_avg_name] = data_raw
                if apply_radial_smoothing and is_smooth_target(v_avg_name, smooth_targets_t):
                    field_data_t[v_avg_name] = smooth_2d_field(data_raw, smooth_sigma)
                    lname_prefix = "(Smoothed) "
                else:
                    field_data_t[v_avg_name] = data_raw
                    lname_prefix = ""
                field_units_t[v_avg_name] = getattr(nc.variables[v], "units", "")
                field_long_name_t[v_avg_name] = lname_prefix + f"time-mean of {getattr(nc.variables[v], 'long_name', v)}"

            sum_weights_t = {
                "V_mr": -1, "V_eh": -1, "V_mv": -1, "V_ev": -1,
                "V_magf": 1, "V_eagf": 1, "V_dh": 1, "V_dv": 1, "tramp": 1
            }
            valid_keys_t_avg = [f"{k}_avg" for k in sum_weights_t if f"{k}_avg" in field_data_t_unsmoothed]
            if valid_keys_t_avg:
                field_data_t["diag_t_sum_avg"] = np.sum(
                    np.stack([field_data_t_unsmoothed[k] * sum_weights_t[k.replace("_avg", "")]
                              for k in valid_keys_t_avg], axis=0), axis=0)
            else:
                field_data_t["diag_t_sum_avg"] = np.zeros_like(field_data_t["vt_start"])
            field_long_name_t["diag_t_sum_avg"] = "time-mean sum of tangential independent forcing terms (unsmoothed)"
            field_units_t["diag_t_sum_avg"] = field_units_t.get(valid_keys_t_avg[0], "") if valid_keys_t_avg else ""
            panel_order_t = ["vt_start", "vt_end", "vt_tendency", "diag_t_sum_avg"] + diag_t_avg_names
            title_mode_t = (f"Tangential time-range diagnostics | start={time_h[t0]:.2f} h, "
                            f"end={time_h[t1]:.2f} h, N={np.sum(time_mask)}")
        else:
            raise ValueError("plot_mode 只能是 'time_point' 或 'time_range'")

    # 色阶
    vt_keys = [k for k in panel_order_t if k.startswith("vt") and k != "vt_tendency"]
    large_key_set_t = {"diag_t_sum", "diag_t_sum_avg", "vcurv_mean", "vcurv_mean_avg",
                       "pgrad_t", "pgrad_t_avg", "vt_tendency"}
    large_diag_keys_t = [k for k in panel_order_t if k in large_key_set_t]
    small_diag_keys_t = [k for k in panel_order_t if (k not in vt_keys and k not in large_diag_keys_t)]

    vt_levels, vt_abs = symmetric_levels_from_data(
        [field_data_t[k] for k in vt_keys], n_levels_ur, floor=1e-8, clip_percentile=99.0)
    small_levels_t, small_abs_t = symmetric_levels_from_data(
        [field_data_t[k] for k in small_diag_keys_t], n_levels_diag,
        floor=1e-12, clip_percentile=small_clip_percentile)
    large_levels_t, large_abs_t = symmetric_levels_from_data(
        [field_data_t[k] for k in large_diag_keys_t], n_levels_diag,
        floor=1e-12, clip_percentile=large_clip_percentile)

    vt_norm = BoundaryNorm(vt_levels, ncolors=256, clip=True)
    if diag_color_mode == "symlog":
        slt = max(small_abs_t * small_linthresh_ratio, 1e-12)
        llt = max(large_abs_t * large_linthresh_ratio, 1e-12)
        small_levels_t = make_symlog_levels(small_abs_t, slt, n_levels_diag)
        large_levels_t = make_symlog_levels(large_abs_t, llt, n_levels_diag)
        small_norm_t = SymLogNorm(linthresh=slt, vmin=-small_abs_t, vmax=small_abs_t, base=10)
        large_norm_t = SymLogNorm(linthresh=llt, vmin=-large_abs_t, vmax=large_abs_t, base=10)
    else:
        small_norm_t = BoundaryNorm(small_levels_t, ncolors=256, clip=True)
        large_norm_t = BoundaryNorm(large_levels_t, ncolors=256, clip=True)

    print(f"vt 色阶范围: [{-vt_abs:.3f}, {vt_abs:.3f}]")
    print(f"切向普通诊断项: [{-small_abs_t:.6f}, {small_abs_t:.6f}]")
    print(f"切向大项: [{-large_abs_t:.6f}, {large_abs_t:.6f}]")

    # 绘图
    R, Z = np.meshgrid(r_plot, z_plot)
    n_panels_t = len(panel_order_t)
    nrows_t = int(np.ceil(n_panels_t / ncols))
    fig_h_t = 4.2 * nrows_t + 4.0

    fig, axes = plt.subplots(nrows_t, ncols, figsize=(19, fig_h_t), sharex=True, sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for i, key in enumerate(panel_order_t):
        ax = axes_flat[i]
        data = field_data_t[key]
        if key in vt_keys:
            levels, norm = vt_levels, vt_norm
        elif key in large_diag_keys_t:
            levels, norm = large_levels_t, large_norm_t
        else:
            levels, norm = small_levels_t, small_norm_t

        ax.contourf(R, Z, data, levels=levels, cmap="RdBu_r", norm=norm, extend="both")
        contour_levels = levels[::2]
        contour_levels = contour_levels[~np.isclose(contour_levels, 0.0, atol=1e-15)]
        if contour_levels.size > 0:
            cs = ax.contour(R, Z, data, levels=contour_levels, colors="k", linewidths=0.6, alpha=0.7)
            ax.clabel(cs, inline=True, fontsize=7, fmt="%.2g")

        unit = field_units_t.get(key, "")
        lname = field_long_name_t.get(key, key)
        short_key = shorten_text(key, 22)
        short_lname = shorten_text(lname, 44)
        title1 = f"{short_key} ({unit})" if unit else short_key
        ax.set_title(f"{title1}\n{short_lname}", fontsize=8.5, pad=4)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.set_xlim(0, max_r_km)
        ax.set_ylim(0, max_z_km)

    for i, ax in enumerate(axes_flat):
        if i >= n_panels_t:
            ax.axis("off")
            continue
        if i % ncols == 0:
            ax.set_ylabel("Height (km)", fontsize=10)
        if i >= (nrows_t - 1) * ncols:
            ax.set_xlabel("Radius (km)", fontsize=10)

    fig.suptitle("Tangential Momentum Diagnostics (Single Page)\n" + title_mode_t,
                 fontsize=15, y=0.992)
    fig.subplots_adjust(left=0.055, right=0.975, bottom=0.20, top=0.93,
                        wspace=0.18, hspace=0.40)

    sm_vt = ScalarMappable(norm=vt_norm, cmap="RdBu_r"); sm_vt.set_array([])
    sm_small_t = ScalarMappable(norm=small_norm_t, cmap="RdBu_r"); sm_small_t.set_array([])
    sm_large_t = ScalarMappable(norm=large_norm_t, cmap="RdBu_r"); sm_large_t.set_array([])

    cax1 = fig.add_axes([0.58, 0.12, 0.37, 0.013])
    cax2 = fig.add_axes([0.58, 0.085, 0.37, 0.013])
    cax3 = fig.add_axes([0.58, 0.05, 0.37, 0.013])

    cb1 = fig.colorbar(sm_vt, cax=cax1, orientation="horizontal", ticks=vt_levels[::2])
    cb1.set_label("tangential wind panels", fontsize=9)
    cb2 = fig.colorbar(sm_small_t, cax=cax2, orientation="horizontal")
    cb2.set_label("normal tangential diagnostic panels", fontsize=9)
    cb3 = fig.colorbar(sm_large_t, cax=cax3, orientation="horizontal")
    cb3.set_label("large panels: sum + vcurv_mean + pgrad_t + vt_tendency", fontsize=9)

    if output_png:
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=150, bbox_inches="tight")
        print(f"[INFO] 切向诊断图已保存: {output_png}")
    else:
        plt.show()
    plt.close(fig)


# ==============================================================================
# 自定义线性组合单窗图
# ==============================================================================

def plot_combo_diagnostics(
    nc_file: str, combo_terms: List[Tuple[float, str]],
    plot_mode: str = "time_range",
    target_hour: float = 60.0, start_hour: float = 42.0, end_hour: float = 74.0,
    max_r_km: float = 300.0, max_z_km: float = 20.0,
    diag_color_mode: str = "symlog",
    clip_percentile: float = 98.5, linthresh_ratio: float = 0.08,
    apply_smoothing: bool = True, smooth_sigma: float = 2.0,
    norm_mode: str = "none",
    output_png: Optional[str] = None,
) -> None:
    """自定义线性组合单窗诊断图。"""
    if len(combo_terms) == 0:
        raise ValueError("combo_terms 不能为空")

    with Dataset(nc_file, "r") as nc:
        time_s = np.asarray(nc.variables["time"][:], dtype=float)
        time_h = time_s / 3600.0
        r_arr = np.asarray(nc.variables["r"][:], dtype=float)
        z_arr = np.asarray(nc.variables["z"][:] if "z" in nc.variables else nc.variables["zh"][:], dtype=float)
        r_mask = r_arr <= max_r_km
        z_mask = z_arr <= max_z_km
        r_plot = r_arr[r_mask]
        z_plot = z_arr[z_mask]

        for _, vname in combo_terms:
            if vname not in nc.variables:
                raise KeyError(f"变量不存在: {vname}")

        if plot_mode == "time_point":
            target_s = target_hour * 3600.0
            t_idx = nearest_time_index(time_s, target_s)
            combo_field = None
            for coef, vname in combo_terms:
                arr = get_var_2d(nc, vname, t_idx, z_mask, r_mask)
                if combo_field is None:
                    combo_field = np.zeros_like(arr, dtype=float)
                combo_field += float(coef) * arr
            title_time = f"target={target_hour:.2f} h, actual={time_h[t_idx]:.2f} h"
        elif plot_mode == "time_range":
            if end_hour < start_hour:
                raise ValueError("end_hour >= start_hour")
            start_s = start_hour * 3600.0; end_s = end_hour * 3600.0
            t0 = nearest_time_index(time_s, start_s); t1 = nearest_time_index(time_s, end_s)
            if t1 < t0: t0, t1 = t1, t0
            time_mask = np.zeros_like(time_s, dtype=bool)
            time_mask[t0:t1 + 1] = True
            combo_field = None
            for coef, vname in combo_terms:
                arr = get_var_2d_mean(nc, vname, time_mask, z_mask, r_mask)
                if combo_field is None:
                    combo_field = np.zeros_like(arr, dtype=float)
                combo_field += float(coef) * arr
            title_time = f"start={time_h[t0]:.2f} h, end={time_h[t1]:.2f} h, N={np.sum(time_mask)}"
        else:
            raise ValueError("plot_mode 只能是 'time_point' 或 'time_range'")

    if apply_smoothing:
        combo_field = smooth_2d_field(combo_field, smooth_sigma)

    expr_text = " ".join([f"{'+' if c>=0 else '-'}{abs(c):g}*{v}" if i > 0 else f"{c:g}*{v}"
                          for i, (c, v) in enumerate(combo_terms)])

    R, Z = np.meshgrid(r_plot, z_plot)
    levels, vmax = symmetric_levels_from_data(
        [combo_field], n_levels=17, floor=1e-12, clip_percentile=clip_percentile)

    if diag_color_mode == "symlog":
        linthresh = max(vmax * linthresh_ratio, 1e-12)
        levels = make_symlog_levels(vmax, linthresh, 17)
        norm = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)
    else:
        norm = BoundaryNorm(levels, ncolors=256, clip=True)

    fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=140)
    cf = ax.contourf(R, Z, combo_field, levels=levels, cmap="RdBu_r", norm=norm, extend="both")
    contour_levels = levels[::2]
    contour_levels = contour_levels[~np.isclose(contour_levels, 0.0, atol=1e-15)]
    if contour_levels.size > 0:
        cs = ax.contour(R, Z, combo_field, levels=contour_levels, colors="k", linewidths=0.6, alpha=0.7)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.2g")

    ax.set_xlim(0, max_r_km); ax.set_ylim(0, max_z_km)
    ax.set_xlabel("Radius (km)"); ax.set_ylabel("Height (km)")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.set_title(f"Custom Linear Combination\n{expr_text} | {plot_mode} | {title_time}", fontsize=10)
    cbar = plt.colorbar(cf, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label("m s-2")

    if output_png:
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=140, bbox_inches="tight")
        print(f"[INFO] 组合图已保存: {output_png}")
    else:
        plt.show()
    plt.close(fig)


# ==============================================================================
# CLI
# ==============================================================================

def _parse_combo_terms(text: str) -> List[Tuple[float, str]]:
    """解析 '1.0,U_magf -1.0,U_mr' 格式的组合项。"""
    parts = text.strip().split()
    terms = []
    for part in parts:
        coef_str, var_name = part.split(",", 1)
        terms.append((float(coef_str), var_name.strip()))
    return terms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="单页诊断图绘制工具")
    parser.add_argument("--panel", choices=["radial", "tangential", "combo"],
                        default="radial", help="面板类型")
    parser.add_argument("--input", default="dataset/wind_Thompson.nc", help="输入 NC 文件")
    parser.add_argument("--mode", choices=["time_point", "time_range"],
                        default="time_range", help="时间模式")
    parser.add_argument("--target-hour", type=float, default=60.0, help="单时次目标 (h)")
    parser.add_argument("--start-hour", type=float, default=42.0, help="时间段起始 (h)")
    parser.add_argument("--end-hour", type=float, default=74.0, help="时间段终止 (h)")
    parser.add_argument("--max-r-km", type=float, default=300.0)
    parser.add_argument("--max-z-km", type=float, default=20.0)
    parser.add_argument("--output", default=None, help="输出 PNG 路径")
    parser.add_argument("--combo-terms", default="1.0,U_magf -1.0,U_mr",
                        help="组合项: 'coef1,var1 coef2,var2 ...'")
    parser.add_argument("--no-smoothing", action="store_true", help="关闭径向平滑")
    parser.add_argument("--no-symlog", action="store_true", help="使用线性色阶")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    apply_smoothing = not args.no_smoothing
    color_mode = "linear" if args.no_symlog else "symlog"

    if args.panel == "radial":
        plot_radial_diagnostics(
            nc_file=args.input, plot_mode=args.mode,
            target_hour=args.target_hour, start_hour=args.start_hour, end_hour=args.end_hour,
            max_r_km=args.max_r_km, max_z_km=args.max_z_km,
            diag_color_mode=color_mode, apply_radial_smoothing=apply_smoothing,
            output_png=args.output,
        )
    elif args.panel == "tangential":
        plot_tangential_diagnostics(
            nc_file=args.input, plot_mode=args.mode,
            target_hour=args.target_hour, start_hour=args.start_hour, end_hour=args.end_hour,
            max_r_km=args.max_r_km, max_z_km=args.max_z_km,
            diag_color_mode=color_mode, apply_radial_smoothing=apply_smoothing,
            output_png=args.output,
        )
    elif args.panel == "combo":
        combo_terms = _parse_combo_terms(args.combo_terms)
        plot_combo_diagnostics(
            nc_file=args.input, combo_terms=combo_terms, plot_mode=args.mode,
            target_hour=args.target_hour, start_hour=args.start_hour, end_hour=args.end_hour,
            max_r_km=args.max_r_km, max_z_km=args.max_z_km,
            diag_color_mode=color_mode, apply_smoothing=apply_smoothing,
            output_png=args.output,
        )


if __name__ == "__main__":
    main()
