"""
Sawyer-Eliassen Equation Diagnostic Pipeline (时间段平均版)

运行示例 (Example Command):
python se_diagnostic_pipeline_timeavg.py --time-avg-start-hours 64 --time-avg-end-hours 72 --output-dir se_pipeline_output_64_72h_avg --sor-omega 1.5
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import struct

try:
    from .center_finder import find_smoothed_min_point

    HAS_CENTER_FINDER = True
except Exception:
    HAS_CENTER_FINDER = False


G = 9.806


@dataclass
class SourceMaskConfig:
    thermal_scale: float = 1.0
    momentum_scale: float = 1.0
    thermal_zero_boxes: List[Dict[str, float]] = field(default_factory=list)
    momentum_zero_boxes: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class PipelineConfig:
    input_file: str = "dataset/cm1out.nc"
    output_dir: str = "se_pipeline_output"
    time_index: int = 0
    target_time_seconds: Optional[float] = None
    target_time_hours: Optional[float] = None
    time_avg_start_hours: Optional[float] = None  # 时间段平均起始 (h)
    time_avg_end_hours: Optional[float] = None    # 时间段平均终止 (h)

    u_name: str = "u"
    v_name: str = "v"
    w_name: str = "w"
    prs_name: str = "prs"
    rho_name: str = "rho"
    theta_name: str = "th"
    psfc_name: str = "psfc"
    u_candidates: Tuple[str, ...] = ("u", "ua", "uinterp")
    v_candidates: Tuple[str, ...] = ("v", "va", "vinterp")
    w_candidates: Tuple[str, ...] = ("w", "wa", "winterp")
    prs_candidates: Tuple[str, ...] = ("prs", "pres", "p")
    rho_candidates: Tuple[str, ...] = ("rho", "rhoa", "dens")
    theta_candidates: Tuple[str, ...] = ("th", "theta", "thpert")
    psfc_candidates: Tuple[str, ...] = ("psfc", "sfcprs", "ps")

    q_name: str = "Q"
    fnu_name: str = "Fnu"
    q_candidates: Tuple[str, ...] = ("Q", "q_diab", "qheat", "th_src")
    fnu_candidates: Tuple[str, ...] = ("Fnu", "fric_radial", "mom_src", "radial_drag")

    q_override_file: str = ""
    fnu_override_file: str = ""
    q_constant: float = 0.0
    fnu_constant: float = 0.0

    max_r_km: float = 300.0
    dr_km: float = 2.0
    enforce_dr_not_finer_than_grid: bool = True
    max_z_km: float = 20.0
    center_window: int = 21
    center_method: str = "min"

    coriolis_f: float = 5.0e-5
    theta_floor: float = 150.0
    theta_outer_smooth_window: int = 1

    elliptic_margin: float = 0.0
    inertia_eps_ratio: float = 1.0e-3
    regularization_max_iter: int = 20

    sor_max_iter: int = 60000
    sor_omega: float = 1.8
    sor_tol: float = 1.0e-14
    sor_verbose_every: int = 500

    write_netcdf: bool = True
    write_ieee: bool = True
    ieee_prefix: str = "SE"
    plot_solution: bool = True

    baroclinic_scale: float = 0.4  # NCL式斜压项缩放因子(bb*0.4)，增强椭圆性

    source_mask: SourceMaskConfig = field(default_factory=SourceMaskConfig)


def _destagger_axis(data: np.ndarray, axis: int) -> np.ndarray:
    sl0 = [slice(None)] * data.ndim
    sl1 = [slice(None)] * data.ndim
    sl0[axis] = slice(0, -1)
    sl1[axis] = slice(1, None)
    return 0.5 * (data[tuple(sl0)] + data[tuple(sl1)])


def _destagger_to_scalar_grid(data: np.ndarray, dims_wo_time: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    out = np.asarray(data, dtype=np.float64)
    out_dims = list(dims_wo_time)
    stag_to_scalar = {"xf": "xh", "yf": "yh", "zf": "zh"}
    for stag_dim in ("xf", "yf", "zf"):
        if stag_dim in out_dims:
            axis = out_dims.index(stag_dim)
            out = _destagger_axis(out, axis)
            out_dims[axis] = stag_to_scalar[stag_dim]
    return out, out_dims


def _require_dims(dims: Sequence[str], expected: Sequence[str], var_name: str) -> None:
    if tuple(dims) != tuple(expected):
        raise ValueError(f"变量 {var_name} 维度为 {tuple(dims)}，期望 {tuple(expected)}")


def _get_time_slice(var_in: xr.DataArray, t_idx: int) -> Tuple[np.ndarray, List[str]]:
    dims = list(var_in.dims)
    indexers = {d: (t_idx if d == "time" else slice(None)) for d in dims}
    data = np.asarray(var_in.isel(indexers), dtype=np.float64)
    return data, [d for d in dims if d != "time"]


def _parse_csv_names(text: str) -> Tuple[str, ...]:
    names = [x.strip() for x in text.split(",") if x.strip()]
    return tuple(names)


def _resolve_time_index(time_vals: np.ndarray, cfg: PipelineConfig) -> Tuple[int, float, str]:
    nt = int(len(time_vals))
    if nt <= 0:
        return 0, 0.0, "fallback_empty_time"

    if cfg.target_time_hours is not None:
        target_sec = float(cfg.target_time_hours) * 3600.0
        idx = int(np.nanargmin(np.abs(time_vals - target_sec)))
        return idx, float(time_vals[idx]), f"target_time_hours={cfg.target_time_hours}"

    if cfg.target_time_seconds is not None:
        target_sec = float(cfg.target_time_seconds)
        idx = int(np.nanargmin(np.abs(time_vals - target_sec)))
        return idx, float(time_vals[idx]), f"target_time_seconds={cfg.target_time_seconds}"

    if cfg.time_index < 0 or cfg.time_index >= nt:
        raise IndexError(f"time_index={cfg.time_index} 越界，time 维长度={nt}")
    return int(cfg.time_index), float(time_vals[cfg.time_index]), "time_index"


def _resolve_time_indices_for_averaging(
    time_vals: np.ndarray, cfg: PipelineConfig
) -> Tuple[List[int], str]:
    """返回时间段内所有时间索引."""
    if cfg.time_avg_start_hours is None or cfg.time_avg_end_hours is None:
        # 回退到单时间点
        idx, _, method = _resolve_time_index(time_vals, cfg)
        return [idx], method

    t_start = float(cfg.time_avg_start_hours) * 3600.0
    t_end = float(cfg.time_avg_end_hours) * 3600.0
    mask = (time_vals >= t_start - 1e-6) & (time_vals <= t_end + 1e-6)
    indices = [int(i) for i in np.where(mask)[0]]
    if not indices:
        raise ValueError(
            f"时间段 [{cfg.time_avg_start_hours}h, {cfg.time_avg_end_hours}h] "
            f"内无时间点 (time 范围 [{time_vals.min():.0f}, {time_vals.max():.0f}]s)"
        )
    method = f"time_avg_{cfg.time_avg_start_hours}h_to_{cfg.time_avg_end_hours}h"
    return indices, method


def _first_existing_var(ds: xr.Dataset, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in ds.variables:
            return name
    return None


def _is_ascii_path(path_str: str) -> bool:
    try:
        path_str.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _find_free_drive_letter() -> Optional[str]:
    for letter in ("X", "Y", "Z", "W", "V", "U", "T", "S", "R", "Q", "P", "O"):
        if not Path(f"{letter}:/").exists():
            return letter
    return None


def _list_subst_mappings() -> Dict[str, str]:
    out = subprocess.run(["subst"], capture_output=True, text=True)
    if out.returncode != 0:
        return {}
    mapping: Dict[str, str] = {}
    for line in out.stdout.splitlines():
        s = line.strip()
        if not s or ":\\:" not in s or "=>" not in s:
            continue
        left, right = s.split("=>", 1)
        drive = left.strip()[0].upper()
        target = right.strip().rstrip("\\/")
        mapping[drive] = target
    return mapping


def _subst_map_dir(dir_path: Path) -> Tuple[Optional[str], Optional[str], bool]:
    # 返回 (drive_letter, mapped_root, created_now)。失败时 (None, None, False)。
    target_norm = str(dir_path).rstrip("\\/").lower()

    existing = _list_subst_mappings()
    for drv, tgt in existing.items():
        if tgt.lower() == target_norm:
            return drv, f"{drv}:/", False

    letter = _find_free_drive_letter()
    if letter is None:
        return None, None, False

    proc = subprocess.run(["subst", f"{letter}:", str(dir_path)], capture_output=True, text=True)
    if proc.returncode != 0:
        return None, None, False

    mapped_root = f"{letter}:/"
    if not Path(mapped_root).exists():
        return None, None, False
    return letter, mapped_root, True


def _subst_unmap(letter: Optional[str], created_now: bool) -> None:
    if (not letter) or (not created_now):
        return
    subprocess.run(["subst", f"{letter}:", "/d"], capture_output=True, text=True)


def _open_dataset_robust(input_file: str) -> Tuple[xr.Dataset, Dict[str, str], Optional[str], bool]:
    p = Path(input_file).expanduser().resolve()
    path_text = str(p)
    open_meta: Dict[str, str] = {"input_path": path_text, "open_path": path_text, "engine": "default"}

    # 先尝试直接打开
    try:
        ds = xr.open_dataset(path_text, decode_cf=False)
        return ds, open_meta, None, False
    except Exception as e_direct:
        direct_err = f"{type(e_direct).__name__}: {e_direct}"

    # 若路径含非 ASCII，尝试用 subst 映射父目录到 ASCII 盘符
    if not _is_ascii_path(path_text):
        letter, mapped_root, created_now = _subst_map_dir(p.parent)
        if letter and mapped_root:
            mapped_path = f"{mapped_root}{p.name}"
            try:
                ds = xr.open_dataset(mapped_path, decode_cf=False)
                open_meta["open_path"] = mapped_path
                open_meta["engine"] = "default+subst"
                open_meta["subst_drive"] = letter
                open_meta["subst_created_now"] = str(created_now)
                return ds, open_meta, letter, created_now
            except Exception as e_subst:
                _subst_unmap(letter, created_now)
                subst_err = f"{type(e_subst).__name__}: {e_subst}"
                raise RuntimeError(
                    "无法打开 NetCDF 文件。"
                    f"\n- 直接打开失败: {direct_err}"
                    f"\n- subst 映射后失败: {subst_err}"
                )

    raise RuntimeError(f"无法打开 NetCDF 文件: {path_text}\n- 直接打开失败: {direct_err}")


def _resolve_core_var_names(ds: xr.Dataset, cfg: PipelineConfig) -> Dict[str, str]:
    mapping: Dict[str, str] = {}

    def pick(preferred: str, candidates: Sequence[str], label: str) -> str:
        if preferred in ds.variables:
            return preferred
        found = _first_existing_var(ds, candidates)
        if found is None:
            raise KeyError(f"未找到变量 {label}。首选={preferred}, 候选={tuple(candidates)}")
        print(f"[INFO] 变量 {label} 自动匹配为: {found}")
        return found

    mapping["u"] = pick(cfg.u_name, cfg.u_candidates, "u")
    mapping["v"] = pick(cfg.v_name, cfg.v_candidates, "v")
    mapping["w"] = pick(cfg.w_name, cfg.w_candidates, "w")
    mapping["prs"] = pick(cfg.prs_name, cfg.prs_candidates, "prs")
    mapping["rho"] = pick(cfg.rho_name, cfg.rho_candidates, "rho")
    mapping["theta"] = pick(cfg.theta_name, cfg.theta_candidates, "theta")

    if cfg.psfc_name in ds.variables:
        mapping["psfc"] = cfg.psfc_name
    else:
        psfc_found = _first_existing_var(ds, cfg.psfc_candidates)
        mapping["psfc"] = psfc_found if psfc_found is not None else ""
        if psfc_found:
            print(f"[INFO] 变量 psfc 自动匹配为: {psfc_found}")
        else:
            print("[WARN] 未找到地面气压变量，将使用网格中心作为台风中心。")

    return mapping


def _compute_radial_bin_index(r2d: np.ndarray, r_bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    nr = len(r_bins) - 1
    idx = np.digitize(r2d.ravel(), r_bins) - 1
    valid = (idx >= 0) & (idx < nr)
    return idx, valid


def _azimuthal_average_by_radius(data_3d: np.ndarray, bin_index_1d: np.ndarray, valid_mask_1d: np.ndarray, nr: int) -> np.ndarray:
    nz = data_3d.shape[0]
    out = np.full((nz, nr), np.nan, dtype=np.float64)
    for level in range(nz):
        flat = data_3d[level].ravel()
        use = valid_mask_1d & np.isfinite(flat)
        if not np.any(use):
            continue
        idx = bin_index_1d[use]
        values = flat[use]
        count = np.bincount(idx, minlength=nr)
        summation = np.bincount(idx, weights=values, minlength=nr)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[level] = summation / count
    return out


def _expand_azimuthal_mean_to_xy(
    mean_zr: np.ndarray,
    bin_index_1d: np.ndarray,
    valid_mask_1d: np.ndarray,
    ny: int,
    nx: int,
) -> np.ndarray:
    nz, _ = mean_zr.shape
    out = np.full((nz, ny, nx), np.nan, dtype=np.float64)
    for k in range(nz):
        flat = out[k].ravel()
        flat[valid_mask_1d] = mean_zr[k, bin_index_1d[valid_mask_1d]]
    return out


def _safe_gradient(field_2d: np.ndarray, coords_1d: np.ndarray, axis: int) -> np.ndarray:
    if field_2d.shape[axis] < 2:
        return np.zeros_like(field_2d, dtype=np.float64)
    return np.gradient(field_2d, coords_1d, axis=axis, edge_order=1)


def _build_box_mask(r_km: np.ndarray, z_km: np.ndarray, box: Dict[str, float]) -> np.ndarray:
    r_min = float(box.get("r_min_km", -np.inf))
    r_max = float(box.get("r_max_km", np.inf))
    z_min = float(box.get("z_min_km", -np.inf))
    z_max = float(box.get("z_max_km", np.inf))

    rr = r_km[None, :]
    zz = z_km[:, None]
    return (rr >= r_min) & (rr <= r_max) & (zz >= z_min) & (zz <= z_max)


def _apply_source_masks(
    q_2d: np.ndarray,
    fnu_2d: np.ndarray,
    r_km: np.ndarray,
    z_km: np.ndarray,
    mask_cfg: SourceMaskConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    q_mod = np.array(q_2d, copy=True, dtype=np.float64)
    fnu_mod = np.array(fnu_2d, copy=True, dtype=np.float64)

    q_mod *= float(mask_cfg.thermal_scale)
    fnu_mod *= float(mask_cfg.momentum_scale)

    for box in mask_cfg.thermal_zero_boxes:
        q_mod[_build_box_mask(r_km, z_km, box)] = 0.0

    for box in mask_cfg.momentum_zero_boxes:
        fnu_mod[_build_box_mask(r_km, z_km, box)] = 0.0

    return q_mod, fnu_mod


def _find_center(psfc_2d: np.ndarray, xh_km: np.ndarray, yh_km: np.ndarray, cfg: PipelineConfig) -> Tuple[float, float]:
    if HAS_CENTER_FINDER:
        try:
            param_names = list(inspect.signature(find_smoothed_min_point).parameters.keys())
            if len(param_names) > 0 and param_names[0] != "nc_file":
                center = find_smoothed_min_point(
                    psfc_2d,
                    xh_km,
                    yh_km,
                    window=cfg.center_window,
                    center_method=cfg.center_method,
                )
                return float(center["x"]), float(center["y"])
        except Exception as e:
            print(f"[WARN] 平滑找中心失败，回退到psfc最小值定位: {type(e).__name__}")

    iy, ix = np.unravel_index(np.nanargmin(psfc_2d), psfc_2d.shape)
    return float(xh_km[ix]), float(yh_km[iy])


def _moving_average_1d(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    pad = w // 2
    arr_pad = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=np.float64) / w
    out = np.convolve(arr_pad, kernel, mode="valid")
    return out


def _smooth_2d_3point(field: np.ndarray) -> np.ndarray:
    """参考NCL: 3点滑动平均, z方向和r方向各做一遍, 边界保留原值."""
    out = np.array(field, copy=True, dtype=np.float64)
    nz, nr = out.shape
    # z-direction
    if nz >= 3:
        for i in range(1, nz - 1):
            out[i, :] = (field[i - 1, :] + field[i, :] + field[i + 1, :]) / 3.0
    # r-direction
    if nr >= 3:
        tmp = np.array(out, copy=True)
        for j in range(1, nr - 1):
            out[:, j] = (tmp[:, j - 1] + tmp[:, j] + tmp[:, j + 1]) / 3.0
    return out


def _repair_nan_2d(field: np.ndarray) -> np.ndarray:
    """参考NCL的NaN修复逻辑: 用邻居(左/上)填充坏值, 全无邻居则置0."""
    out = np.array(field, copy=True, dtype=np.float64)
    nz, nr = out.shape
    n_bad = 0
    for k in range(nz):
        for i in range(nr):
            v = out[k, i]
            if (not np.isfinite(v)) or np.ma.is_masked(v):
                if i > 0 and np.isfinite(out[k, i - 1]):
                    out[k, i] = out[k, i - 1]
                elif k > 0 and np.isfinite(out[k - 1, i]):
                    out[k, i] = out[k - 1, i]
                else:
                    out[k, i] = 0.0
                n_bad += 1
    if n_bad > 0:
        print(f"[INFO] _repair_nan_2d: 修复了 {n_bad} 个非法值")
    return out


def azimuthal_average_from_3d(cfg: PipelineConfig) -> Dict[str, np.ndarray]:
    ds, open_meta, subst_letter, subst_created_now = _open_dataset_robust(cfg.input_file)
    try:
        var_map = _resolve_core_var_names(ds, cfg)

        nt = int(ds.sizes.get("time", 1))

        xh = np.asarray(ds["xh"], dtype=np.float64)
        yh = np.asarray(ds["yh"], dtype=np.float64)
        zh = np.asarray(ds["zh"], dtype=np.float64)

        dx_native_km = float(np.median(np.abs(np.diff(xh)))) if xh.size > 1 else float(cfg.dr_km)
        dy_native_km = float(np.median(np.abs(np.diff(yh)))) if yh.size > 1 else float(cfg.dr_km)
        dr_used_km = float(cfg.dr_km)
        dr_floor_km = max(1.0e-6, min(dx_native_km, dy_native_km))
        if cfg.enforce_dr_not_finer_than_grid and dr_used_km < dr_floor_km:
            print(
                "[WARN] 径向分箱过细会导致欠采样条纹: "
                f"dr_km={dr_used_km:.3f} < native_grid_min={dr_floor_km:.3f}. "
                f"已自动调整为 dr_km={dr_floor_km:.3f}。"
            )
            dr_used_km = dr_floor_km

        if "time" in ds.variables:
            time_vals = np.asarray(ds["time"], dtype=np.float64)
        else:
            time_vals = np.array([0.0], dtype=np.float64)

        time_idx, time_selected_sec, time_selection_method = _resolve_time_index(time_vals, cfg)
        print(
            "[INFO] 时间选择: "
            f"method={time_selection_method}, index={time_idx}, "
            f"selected_time_seconds={time_selected_sec:.3f}"
        )

        z_keep = zh <= cfg.max_z_km
        z_sel = zh[z_keep]
        z_m = z_sel * 1000.0

        def load_3d(var_name: str, t_idx: int) -> np.ndarray:
            data, dims = _get_time_slice(ds[var_name], t_idx)
            data_s, dims_s = _destagger_to_scalar_grid(data, dims)
            _require_dims(dims_s, ("zh", "yh", "xh"), var_name)
            return data_s[z_keep]

        # ---------- 时间索引确定 ----------
        time_indices, time_selection_method = _resolve_time_indices_for_averaging(time_vals, cfg)
        n_times = len(time_indices)
        if n_times > 1:
            print(f"[INFO] 时间段平均: {n_times} 个时次, "
                  f"time_index {time_indices[0]}→{time_indices[-1]}")
        else:
            print(f"[INFO] 单时次: index={time_indices[0]}")

        # ---------- 累积器 (z, r) ----------
        r_bins = np.arange(0.0, cfg.max_r_km + dr_used_km, dr_used_km)
        r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
        nr = len(r_centers)

        accum = {
            "ur": np.zeros((len(z_sel), nr), dtype=np.float64),
            "ut": np.zeros((len(z_sel), nr), dtype=np.float64),
            "w": np.zeros((len(z_sel), nr), dtype=np.float64),
            "prs": np.zeros((len(z_sel), nr), dtype=np.float64),
            "rho": np.zeros((len(z_sel), nr), dtype=np.float64),
            "theta": np.zeros((len(z_sel), nr), dtype=np.float64),
            "Q": np.zeros((len(z_sel), nr), dtype=np.float64),
        }
        # 动量预算分量
        budget_accum: Dict[str, np.ndarray] = {}
        budget_pairs_used: List[str] = []
        for suffix in ("hadv", "vadv", "pgrad", "cor", "hidiff", "hturb", "vidiff", "vturb", "rdamp"):
            ub_name = f"ub_{suffix}"
            vb_name = f"vb_{suffix}"
            if ub_name in ds.variables and vb_name in ds.variables:
                budget_accum[suffix] = np.zeros((len(z_sel), nr), dtype=np.float64)
                budget_pairs_used.append(f"{ub_name}+{vb_name}")
        xc_km_accum, yc_km_accum = 0.0, 0.0

        # ---------- 时间循环 ----------
        for ti, t_idx in enumerate(time_indices):
            if ti % 3 == 0:
                print(f"  [INFO] 处理时次 {ti+1}/{n_times}...")
            # 加载3D场
            u = load_3d(var_map["u"], t_idx)
            v = load_3d(var_map["v"], t_idx)
            w = load_3d(var_map["w"], t_idx)
            prs = load_3d(var_map["prs"], t_idx)
            rho = load_3d(var_map["rho"], t_idx)
            theta = load_3d(var_map["theta"], t_idx)

            # 台风中心
            if var_map["psfc"]:
                psfc_data, psfc_dims = _get_time_slice(ds[var_map["psfc"]], t_idx)
                psfc_s, psfc_dims_s = _destagger_to_scalar_grid(psfc_data, psfc_dims)
                xc_km, yc_km = _find_center(psfc_s, xh, yh, cfg)
            else:
                xc_km = float(0.5 * (xh.min() + xh.max()))
                yc_km = float(0.5 * (yh.min() + yh.max()))
            xc_km_accum += xc_km / n_times
            yc_km_accum += yc_km / n_times

            # 柱坐标投影
            X, Y = np.meshgrid(xh, yh)
            r2d_km = np.sqrt((X - xc_km) ** 2 + (Y - yc_km) ** 2)
            theta_ang = np.arctan2((Y - yc_km) * 1000.0, (X - xc_km) * 1000.0)
            cos_t = np.cos(theta_ang)
            sin_t = np.sin(theta_ang)

            ur_3d = u * cos_t[None, :, :] + v * sin_t[None, :, :]
            ut_3d = -u * sin_t[None, :, :] + v * cos_t[None, :, :]

            bin_index, valid_mask = _compute_radial_bin_index(r2d_km, r_bins)

            for key, data3d in [("ur", ur_3d), ("ut", ut_3d), ("w", w),
                                 ("prs", prs), ("rho", rho), ("theta", theta)]:
                accum[key] += _azimuthal_average_by_radius(data3d, bin_index, valid_mask, nr) / n_times

            ptb_3d = load_3d("ptb_mp", t_idx) if "ptb_mp" in ds.variables else np.zeros_like(theta)
            accum["Q"] += _azimuthal_average_by_radius(ptb_3d, bin_index, valid_mask, nr) / n_times

            for suffix in budget_accum:
                ub_name = f"ub_{suffix}"
                vb_name = f"vb_{suffix}"
                ub = load_3d(ub_name, t_idx)
                vb = load_3d(vb_name, t_idx)
                t_term = -ub * sin_t[None, :, :] + vb * cos_t[None, :, :]
                budget_accum[suffix] += _azimuthal_average_by_radius(t_term, bin_index, valid_mask, nr) / n_times

        # ---------- 用累积的平均场计算动量源 Fnu ----------
        ur_avg = accum["ur"]
        ut_avg = accum["ut"]
        w_avg = accum["w"]
        theta_avg = accum["theta"]
        q_avg = accum["Q"]
        q_used = "ptb_mp_time_avg"

        r_m = r_centers * 1000.0
        r_safe = np.maximum(r_m, 0.5 * np.min(np.diff(r_m)) if len(r_m) > 1 else 1.0)

        # 动量源 Fnu: 同原始逻辑
        if "hadv" in budget_accum and "vadv" in budget_accum:
            thadv = budget_accum.get("hadv", np.zeros_like(ut_avg))
            tvadv = budget_accum.get("vadv", np.zeros_like(ut_avg))
            thidiff = budget_accum.get("hidiff", np.zeros_like(ut_avg))
            thturb = budget_accum.get("hturb", np.zeros_like(ut_avg))
            tvidiff = budget_accum.get("vidiff", np.zeros_like(ut_avg))
            tvturb = budget_accum.get("vturb", np.zeros_like(ut_avg))
            trdamp = budget_accum.get("rdamp", np.zeros_like(ut_avg))

            dut_dr = _safe_gradient(ut_avg, r_m, axis=1)
            dut_dz = _safe_gradient(ut_avg, z_m, axis=0)
            vcurv_mean = (ur_avg * ut_avg) / r_safe[None, :]

            # 涡动曲率项: 由于平均了方位角平均场, 无法直接计算 ur'ut' 涡动项.
            # 简化为仅用平均场近似 (时间平均已消除大部分涡动信号).
            vcurv_eddy = np.zeros_like(ut_avg)

            V_mr = ur_avg * dut_dr
            V_mv = w_avg * dut_dz
            V_eh = -thadv - V_mr - vcurv_mean - vcurv_eddy
            V_ev = -tvadv - V_mv
            V_dh = thidiff + thturb
            V_dv = tvidiff + tvturb
            tramp = trdamp

            V_mzeta = V_mr - (-vcurv_mean + budget_accum.get("cor", np.zeros_like(ut_avg)))
            V_ezeta = V_eh - vcurv_eddy
            fnu_avg = -V_ezeta - V_ev + V_dh + V_dv + tramp
            fnu_used = "diagnosed_time_avg"
        else:
            fnu_avg = np.zeros_like(theta_avg)
            fnu_used = "zero_time_avg"
            V_mzeta = np.zeros_like(ut_avg)
            V_ezeta = np.zeros_like(ut_avg)
            V_ev = np.zeros_like(ut_avg)
            V_dh = np.zeros_like(ut_avg)
            V_dv = np.zeros_like(ut_avg)
            tramp = np.zeros_like(ut_avg)

        q_avg = np.nan_to_num(q_avg)
        fnu_avg = np.nan_to_num(fnu_avg)
        V_mzeta = np.nan_to_num(V_mzeta); V_ezeta = np.nan_to_num(V_ezeta)
        V_ev = np.nan_to_num(V_ev); V_dh = np.nan_to_num(V_dh)
        V_dv = np.nan_to_num(V_dv); tramp = np.nan_to_num(tramp)

        dtheta_dt_avg = np.zeros_like(q_avg)
        eddy_th_adv_avg = np.zeros_like(q_avg)

        return {
            "z_km": z_sel,
            "r_km": r_centers,
            "center_x_km": np.array([xc_km_accum], dtype=np.float64),
            "center_y_km": np.array([yc_km_accum], dtype=np.float64),
            "ur": accum["ur"],
            "ut": accum["ut"],
            "w": accum["w"],
            "prs": accum["prs"],
            "rho": accum["rho"],
            "theta": accum["theta"],
            "Q": q_avg,
            "Fnu": fnu_avg,
            "Q_dtheta_dt": dtheta_dt_avg,
            "Q_eddy_adv": eddy_th_adv_avg,
            "V_mzeta": V_mzeta,
            "V_ezeta": V_ezeta,
            "V_ev": V_ev,
            "V_dh": V_dh,
            "V_dv": V_dv,
            "tramp": tramp,
            "Q_used": np.array([q_used]),
            "Fnu_used": np.array([fnu_used]),
            "momentum_budget_pairs_used": np.array(budget_pairs_used, dtype=object),
            "dataset_open_path": np.array([open_meta.get("open_path", "")]),
            "dataset_engine": np.array([open_meta.get("engine", "")]),
            "time_index_used": np.array([time_indices[0]], dtype=np.int64),
            "time_seconds_used": np.array([time_vals[time_indices[0]]], dtype=np.float64),
            "time_selection_method": np.array([time_selection_method]),
            "time_avg_n_samples": np.array([n_times], dtype=np.int64),
            "dr_km_requested": np.array([float(cfg.dr_km)], dtype=np.float64),
            "dr_km_used": np.array([dr_used_km], dtype=np.float64),
            "grid_dx_km": np.array([dx_native_km], dtype=np.float64),
            "grid_dy_km": np.array([dy_native_km], dtype=np.float64),
            "u_used": np.array([var_map["u"]]),
            "v_used": np.array([var_map["v"]]),
            "w_used": np.array([var_map["w"]]),
            "prs_used": np.array([var_map["prs"]]),
            "rho_used": np.array([var_map["rho"]]),
            "theta_used": np.array([var_map["theta"]]),
            "psfc_used": np.array([var_map["psfc"] if var_map["psfc"] else "domain_center"]),
        }
    finally:
        ds.close()
        _subst_unmap(subst_letter, subst_created_now)


def invert_theta_from_thermal_wind(
    ut_2d: np.ndarray,
    theta_2d_model: np.ndarray,
    r_m: np.ndarray,
    z_m: np.ndarray,
    f: float,
    theta_floor: float,
    smooth_window: int,
) -> np.ndarray:
    # 以外边界位温作为积分边界，使用近似梯度风热成风关系反算平衡位温。
    vt = np.asarray(ut_2d, dtype=np.float64)
    theta_model = np.asarray(theta_2d_model, dtype=np.float64)

    r_safe = np.maximum(r_m, 0.5 * np.min(np.diff(r_m)) if len(r_m) > 1 else 1.0)
    gradwind = vt**2 / r_safe[None, :] + f * vt
    dgradwind_dz = _safe_gradient(gradwind, z_m, axis=0)

    theta_outer = theta_model[:, -1]
    theta_outer = _moving_average_1d(theta_outer, smooth_window)
    theta_outer = np.where(np.isfinite(theta_outer), theta_outer, np.nanmedian(theta_model, axis=1))
    theta_outer = np.maximum(theta_outer, theta_floor)

    dtheta_dr = -(theta_outer[:, None] / G) * dgradwind_dz

    nz, nr = vt.shape
    theta_bal = np.full((nz, nr), np.nan, dtype=np.float64)
    theta_bal[:, -1] = theta_outer

    if nr > 1:
        dr = np.diff(r_m)
        for j in range(nr - 2, -1, -1):
            theta_bal[:, j] = theta_bal[:, j + 1] - 0.5 * (dtheta_dr[:, j + 1] + dtheta_dr[:, j]) * dr[j]

    theta_bal = np.where(np.isfinite(theta_bal), theta_bal, theta_model)
    theta_bal = np.maximum(theta_bal, theta_floor)
    return theta_bal


def build_se_diagnostic_fields(
    ut_2d: np.ndarray,
    theta_bal_2d: np.ndarray,
    rho_2d: np.ndarray,
    q_2d: np.ndarray,
    fnu_2d: np.ndarray,
    r_m: np.ndarray,
    z_m: np.ndarray,
    f: float,
) -> Dict[str, np.ndarray]:
    vt = np.asarray(ut_2d, dtype=np.float64)
    theta = np.asarray(theta_bal_2d, dtype=np.float64)
    rho = np.asarray(rho_2d, dtype=np.float64)

    r_safe = np.maximum(r_m, 0.5 * np.min(np.diff(r_m)) if len(r_m) > 1 else 1.0)

    # 约定: chi = 1/theta_b, C = d(vt)/dz。
    chi = 1.0 / np.maximum(theta, 1.0)
    C = _safe_gradient(vt, z_m, axis=0)

    # NCL式梯度风项: ct = (vt/r + f) * vt, 用于斜压系数K2/K3
    ct_gradwind = vt**2 / r_safe[None, :] + f * vt

    rv = vt * r_safe[None, :]
    zeta_abs = _safe_gradient(rv, r_m, axis=1) / r_safe[None, :]
    zeta = zeta_abs - f
    xi = f + 2.0 * vt / r_safe[None, :]
    inertial_stability = xi * (zeta + f)

    return {
        "chi": chi,
        "C": C,
        "ct": ct_gradwind,
        "xi": xi,
        "zeta": zeta,
        "inertial_stability": inertial_stability,
        "rho": rho,
        "Q": np.asarray(q_2d, dtype=np.float64),
        "Fnu": np.asarray(fnu_2d, dtype=np.float64),
    }


def build_se_coefficients(
    fields: Dict[str, np.ndarray],
    r_m: np.ndarray,
    z_m: np.ndarray,
    K1_override: Optional[np.ndarray] = None,
    K2_override: Optional[np.ndarray] = None,
    K3_override: Optional[np.ndarray] = None,
    baroclinic_scale: float = 0.4,
) -> Dict[str, np.ndarray]:
    chi = fields["chi"]
    C = fields["C"]
    xi = fields["xi"]
    zeta = fields["zeta"]
    rho = np.maximum(fields["rho"], 1.0e-8)
    Q = fields["Q"]
    Fnu = fields["Fnu"]

    # 参考NCL: 3点平滑 chi 和 rho, 抑制梯度噪声
    chi = _smooth_2d_3point(chi)
    rho = _smooth_2d_3point(rho)

    chi_r = _safe_gradient(chi, r_m, axis=1)
    chi_z = _safe_gradient(chi, z_m, axis=0)

    K1 = -G * chi_z if K1_override is None else K1_override

    # NCL式K2: 使用梯度风项 ct = (vt/r + f)*vt 替代 d(vt)/dz, 并乘 baroclinic_scale
    # 在 build_se_diagnostic_fields 中 C = d(vt)/dz, 这里额外构造 ct 用于K2
    if "ct" in fields and K2_override is None:
        ct_field = fields["ct"]
        K2 = -_safe_gradient(chi * ct_field, z_m, axis=0)
        K2 *= baroclinic_scale  # NCL bb*0.4
    else:
        K2 = -_safe_gradient(chi * C, z_m, axis=0) if K2_override is None else K2_override

    # NCL式K3: 使用 ct 替代 C 作为斜压耦合项
    if "ct" in fields and K3_override is None:
        ct_field = fields["ct"]
        K3 = chi * fields["inertial_stability"] + ct_field * chi_r
    else:
        K3 = chi * fields["inertial_stability"] + C * chi_r if K3_override is None else K3_override

    r_safe = np.maximum(r_m, 0.5 * np.min(np.diff(r_m)) if len(r_m) > 1 else 1.0)
    M = 1.0 / (rho * r_safe[None, :])

    A = K1 * M
    B = 2.0 * K2 * M
    Cc = K3 * M

    D = _safe_gradient(A, r_m, axis=1) + _safe_gradient(K2 * M, z_m, axis=0)
    E = _safe_gradient(K2 * M, r_m, axis=1) + _safe_gradient(Cc, z_m, axis=0)

    thermal_flux = (chi**2) * Q
    momentum_flux = chi * xi * Fnu

    forcing_thermal = G * _safe_gradient(thermal_flux, r_m, axis=1) + _safe_gradient(C * thermal_flux, z_m, axis=0)
    forcing_momentum = -_safe_gradient(momentum_flux, z_m, axis=0)
    F_term = forcing_thermal + forcing_momentum

    discriminant = 4.0 * A * Cc - B**2

    return {
        "A": A,
        "B": B,
        "C": Cc,
        "D": D,
        "E": E,
        "F": F_term,
        "forcing_total": F_term,
        "forcing_thermal": forcing_thermal,
        "forcing_momentum": forcing_momentum,
        "discriminant": discriminant,
        "K1": K1,
        "K2": K2,
        "K3": K3,
    }


def regularize_inertial_stability_for_ellipticity(
    fields: Dict[str, np.ndarray],
    r_m: np.ndarray,
    z_m: np.ndarray,
    margin: float,
    eps_ratio: float,
    max_iter: int,
    baroclinic_scale: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    chi = _smooth_2d_3point(fields["chi"])  # NCL式平滑抑制噪声
    C = fields["C"]
    xi = fields["xi"]
    zeta = fields["zeta"]
    f_cor = 5e-5

    # NCL式涡度底板: vor >= fc*f (fc=0.01), 确保惯性稳定度为正
    zeta_reg = np.maximum(zeta, 0.01 * f_cor)

    chi_r = _safe_gradient(chi, r_m, axis=1)
    chi_z = _safe_gradient(chi, z_m, axis=0)

    K1 = -G * chi_z

    # 上层海绵: 15km以上对 K1/K2/K3 同步指数增强, 维持椭圆性
    z_sponge_start = 15000.0
    z_sponge_scale = 500.0   # e-folding 500m, 20km处放大 exp(10)≈22000x
    sponge = np.where(z_m > z_sponge_start, np.exp((z_m - z_sponge_start) / z_sponge_scale), 1.0)
    sponge_2d = sponge[:, None]

    K1_reg = np.maximum(K1, 1.0e-10) * sponge_2d

    # NCL式K3: I2 = chi * xi * (zeta_reg + f_cor) + ct * chi_r
    if "ct" in fields:
        ct_field = fields["ct"]
        I2 = chi * xi * (zeta_reg + f_cor) + ct_field * chi_r
    else:
        I2 = chi * xi * (zeta_reg + f_cor) + C * chi_r

    I2_max = float(np.nanmax(np.abs(I2)))
    small_pos = I2_max * 1e-3
    I2_reg = np.copy(I2)
    bad_I2 = I2_reg < 0
    I2_reg[bad_I2] = small_pos
    K3_reg = I2_reg

    # NCL式K2: 使用 ct 替代 C, 并先乘 baroclinic_scale
    if "ct" in fields:
        ct_field = fields["ct"]
        K2_reg = -_safe_gradient(chi * ct_field, z_m, axis=0)
        K2_reg *= baroclinic_scale
    else:
        K2_reg = -_safe_gradient(chi * C, z_m, axis=0)

    # 海绵层同步放大 K2 和 K3, 维持椭圆结构
    K2_reg *= sponge_2d
    K3_reg *= sponge_2d

    bad_before = 0
    n_iters = 0

    for k in range(max_iter):
        D_img = K1_reg * K3_reg - K2_reg**2
        # 判别式底板: 要求 >= max(0, 0.01 * K1*K3), 留1%安全裕度
        min_disc = np.maximum(0.01 * K1_reg * K3_reg, 1e-40)
        bad_D = (D_img <= 0) | (D_img < min_disc)
        n_bad = int(np.count_nonzero(bad_D))

        if k == 0:
            bad_before = n_bad

        if n_bad == 0:
            break

        K2_reg[bad_D] *= 0.8
        n_iters += 1

    return K1_reg, K2_reg, K3_reg, {
        "regularized": 1.0 if bad_before > 0 or np.any(bad_I2) else 0.0,
        "iterations": float(n_iters),
        "bad_I2_points": float(np.count_nonzero(bad_I2)),
        "bad_D_points_before": float(bad_before),
        "bad_D_points_after": float(n_bad),
        "baroclinic_scale": baroclinic_scale,
    }


def solve_se_sparse(
    A: np.ndarray, B: np.ndarray, C: np.ndarray,
    D: np.ndarray, E: np.ndarray, Fin: np.ndarray,
    dr: float, dz: float,
) -> np.ndarray:
    """scipy稀疏直接求解: 无条件收敛，适用于任何分辨率."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve

    nr, nz = A.shape
    N = nr * nz
    dr2, dz2 = dr*dr, dz*dz
    drdz4 = 4.0 * dr * dz
    dr2_2 = 2.0 * dr
    dz2_2 = 2.0 * dz

    rows, cols, vals = [], [], []
    rhs = np.zeros(N, dtype=np.float64)

    for i in range(nr):
        for j in range(nz):
            k = i * nz + j  # flat index

            # --- p_xx coefficients ---
            # interior: (P[i+1]+P[i-1]-2P[i])/dr² * A[i,j]
            if i == 0:
                # dpsi/dr=0: P[-1]=P[1]  →  (2P[1]-2P[0])/dr²
                c0 = -2.0/dr2; c1 = 2.0/dr2
                rows.extend([k,k]); cols.extend([k,k+nz]); vals.extend([c0*A[i,j], c1*A[i,j]])
            elif i == nr-1:
                c0 = -2.0/dr2; c1 = 2.0/dr2
                rows.extend([k,k]); cols.extend([k,k-nz]); vals.extend([c0*A[i,j], c1*A[i,j]])
            else:
                c0 = -2.0/dr2; c1 = 1.0/dr2; c2 = 1.0/dr2
                rows.extend([k,k,k]); cols.extend([k,k-nz,k+nz])
                vals.extend([c0*A[i,j], c1*A[i,j], c2*A[i,j]])

            # --- p_yy coefficients ---
            if j == 0:
                # psi=0 at bottom → P[0]=0 fixed, p_yy = (P[1]-2P[0])/dz²=(P[1]-0)/dz²
                # Actually with P[:,0]=0: p_yy = (P[i,2]+0-2P[i,1])/dz² = (P[i,2]-2P[i,1])/dz²
                c0 = -2.0/dz2; c1 = 1.0/dz2
                rows.extend([k,k]); cols.extend([k,k+1]); vals.extend([c0*C[i,j], c1*C[i,j]])
            elif j == nz-1:
                # dpsi/dz=0 at top → P[nz+1]=P[nz]
                # p_yy = (P[nz]+P[nz-1]-2P[nz])/dz² = (P[nz-1]-P[nz])/dz²
                c0 = -1.0/dz2; c1 = 1.0/dz2
                rows.extend([k,k]); cols.extend([k,k-1]); vals.extend([c0*C[i,j], c1*C[i,j]])
            else:
                c0 = -2.0/dz2; c1 = 1.0/dz2; c2 = 1.0/dz2
                rows.extend([k,k,k]); cols.extend([k,k-1,k+1])
                vals.extend([c0*C[i,j], c1*C[i,j], c2*C[i,j]])

            # --- p_xy cross-derivative (B term) ---
            # (P[i+1,j+1]-P[i+1,j-1]-P[i-1,j+1]+P[i-1,j-1])/(4*dr*dz)
            def _add_xy(ri, rj, sign):
                if 0 <= ri < nr and 0 <= rj < nz:
                    rows.append(k); cols.append(ri*nz+rj)
                    vals.append(sign * B[i,j] / drdz4)
            if i == 0:
                _add_xy(i+1, j+1 if j+1<nz else nz-2, 1.0)  # P[1,j+1]
                _add_xy(i+1, j-1, -1.0)                      # P[1,j-1]
                # P[-1,...] = P[1,...] via dpsi/dr=0 → +P[1,j+1]-P[1,j-1]-P[1,j+1]+P[1,j-1] = 0 → cancels
            elif i == nr-1:
                _add_xy(i-1, j+1 if j+1<nz else nz-2, -1.0)
                _add_xy(i-1, j-1, 1.0)
            else:
                _add_xy(i+1, j+1 if j+1<nz else nz-2, 1.0)
                _add_xy(i+1, j-1, -1.0)
                _add_xy(i-1, j+1 if j+1<nz else nz-2, -1.0)
                _add_xy(i-1, j-1, 1.0)

            # --- p_x first-order (D term) ---
            if i == 0:
                pass  # dpsi/dr=0 → p_x = 0 at inner boundary
            elif i == nr-1:
                # one-sided: (P[i]-P[i-1])/(2*dr)
                rows.extend([k,k]); cols.extend([k,k-nz])
                vals.extend([D[i,j]/dr2_2, -D[i,j]/dr2_2])
            else:
                rows.extend([k,k]); cols.extend([k+nz,k-nz])
                vals.extend([D[i,j]/dr2_2, -D[i,j]/dr2_2])

            # --- p_y first-order (E term) ---
            if j == 0:
                pass  # psi=0 → p_y unaffected at bottom (P[0]=0 is enforced by Dirichlet row)
                # Actually need one-sided: (P[1]-0)/(2*dz)
                rows.append(k); cols.append(k+1)
                vals.append(E[i,j]/dz2_2)
            elif j == nz-1:
                # dpsi/dz=0 → p_y = 0 at top
                pass
            else:
                rows.extend([k,k]); cols.extend([k+1,k-1])
                vals.extend([E[i,j]/dz2_2, -E[i,j]/dz2_2])

            # RHS
            rhs[k] = Fin[i, j]

    M = csr_matrix((vals, (rows, cols)), shape=(N, N))
    p_flat = spsolve(M, rhs)
    P = np.zeros((nr, nz + 2), dtype=np.float64)
    P[:, 1:-1] = p_flat.reshape(nr, nz)
    P[:, 0] = 0.0
    P[:, -1] = P[:, -2]
    return P



def solve_se_sor(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    E: np.ndarray,
    F: np.ndarray,
    dr: float,
    dz: float,
    max_iter: int,
    omega: float,
    tol: float,
    verbose_every: int,
) -> np.ndarray:
    nr, nz = A.shape
    dr2 = dr ** 2
    dz2 = dz ** 2
    drdz4 = 4.0 * dr * dz
    dr2_2 = 2.0 * dr
    dz2_2 = 2.0 * dz

    max_retries = 4
    current_retry = 0
    current_omega = omega

    while current_retry <= max_retries:
        P = np.zeros((nr, nz + 2), dtype=np.float64)
        converged = False
        nonfinite_failure = False
        if current_retry > 0:
            print(f"Retrying SOR with omega={current_omega} (retry {current_retry}/{max_retries})")

        for it in range(1, max_iter + 1):
            max_res = 0.0

            # --- 向量化 i-loop: 逐行用 numpy 处理所有 j ---
            for i in range(1, nr - 1):
                js = slice(1, nz + 1)   # interior j indices: 1..nz
                jj = slice(0, nz)       # corresponding A/C indices: 0..nz-1

                # 上下邻居
                if i == 1:
                    ip1 = i + 1
                    im1 = 0  # P[0,:] = P[1,:] enforced after each iter
                elif i == nr - 2:
                    ip1 = nr - 1  # P[-1,:] = P[-2,:]
                    im1 = i - 1
                else:
                    ip1 = i + 1
                    im1 = i - 1

                # 二阶导数 p_xx (i==nr-2 用单侧差分)
                if i == nr - 2:
                    p_xx = (P[im1, js] - P[i, js]) / dr2  # (P[i-1] + P[i] - 2P[i]) / dr²
                    p_x = (P[i, js] - P[im1, js]) / dr2_2
                else:
                    p_xx = (P[ip1, js] + P[im1, js] - 2.0 * P[i, js]) / dr2
                    p_x = (P[ip1, js] - P[im1, js]) / dr2_2

                # 交叉导数 p_xy (四个角)
                p_xy = (P[ip1, 2:nz+2] - P[ip1, 0:nz]
                        - P[im1, 2:nz+2] + P[im1, 0:nz]) / drdz4

                # 二阶导数 p_yy 和 p_y
                p_yy = (P[i, 2:nz+2] + P[i, 0:nz] - 2.0 * P[i, js]) / dz2
                p_y = (P[i, 2:nz+2] - P[i, 0:nz]) / dz2_2

                # 残差
                residual = (A[i, jj] * p_xx + B[i, jj] * p_xy
                            + C[i, jj] * p_yy + D[i, jj] * p_x
                            + E[i, jj] * p_y - F[i, jj])

                eij = -2.0 * A[i, jj] / dr2 - 2.0 * C[i, jj] / dz2

                if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(eij)):
                    nonfinite_failure = True
                    break

                valid = np.abs(eij) >= 1.0e-30
                delta = np.zeros_like(residual)
                delta[valid] = current_omega * residual[valid] / eij[valid]
                np.clip(delta, -1.0e5, 1.0e5, out=delta)

                P[i, js] -= delta

                abs_res = np.abs(residual)
                if np.any(valid):
                    row_max = float(np.nanmax(abs_res[valid]))
                    if np.isfinite(row_max) and row_max > max_res:
                        max_res = row_max

            if nonfinite_failure:
                print(f"SOR failed at iter={it}: encountered non-finite residual/update")
                break

            # 边界条件
            P[0, :] = P[1, :]
            P[-1, :] = P[-2, :]
            P[:, 0] = 0.0
            P[:, -1] = P[:, -2]  # dpsi/dz=0 自由滑移顶

            if verbose_every > 0 and (it == 1 or it % verbose_every == 0):
                print(f"iter={it:6d}, max_res={max_res:.3e}")

            if np.isfinite(max_res) and max_res < tol:
                print(f"SOR converged at iter={it}, max_res={max_res:.3e}")
                converged = True
                break
        else:
            print(f"SOR not converged after {max_iter} iterations")

        if nonfinite_failure or (not np.all(np.isfinite(P))):
            if current_retry < max_retries:
                current_retry += 1
                current_omega *= 0.5
                continue
            else:
                raise FloatingPointError(
                    "SOR求解产生非有限值(NaN/Inf)。"
                    "当前方程在该配置下可能非椭圆或条件数过差，请检查源项/正则化/系数符号。"
                )
        break

    if (not converged) and max_iter > 0:
        print("[WARN] SOR未满足残差阈值，输出为未完全收敛解。")

    return P


def psi_to_uw(psi: np.ndarray, rho_ext: np.ndarray, r_m: np.ndarray, dr: float, dz: float) -> Tuple[np.ndarray, np.ndarray]:
    nr, nzp2 = psi.shape
    nz = nzp2 - 2
    U = np.zeros_like(psi)
    W = np.zeros_like(psi)

    r_safe = np.maximum(r_m, 0.5 * dr)

    for iz in range(1, nz + 1):
        for ir in range(1, nr - 1):
            denom = 2.0 * dr * r_safe[ir] * max(rho_ext[ir, iz], 1.0e-8)
            W[ir, iz] = (psi[ir + 1, iz] - psi[ir - 1, iz]) / denom
        W[0, iz] = W[1, iz]
        W[-1, iz] = W[-2, iz]

    W[:, 0] = 0.0
    W[:, -1] = 0.0

    for ir in range(1, nr):
        denom0 = dz * r_safe[ir] * 0.5 * (rho_ext[ir, 0] + rho_ext[ir, 1])
        denom1 = dz * r_safe[ir] * 0.5 * (rho_ext[ir, -1] + rho_ext[ir, -2])
        if abs(denom0) > 0:
            U[ir, 0] = -(psi[ir, 1] - psi[ir, 0]) / denom0
        if abs(denom1) > 0:
            U[ir, -1] = -(psi[ir, -1] - psi[ir, -2]) / denom1
        for iz in range(1, nz + 1):
            denom = 2.0 * dz * r_safe[ir] * max(rho_ext[ir, iz], 1.0e-8)
            U[ir, iz] = -(psi[ir, iz + 1] - psi[ir, iz - 1]) / denom

    U[0, :] = 0.0
    return U, W


def _to_solver_layout_zr_to_rz(field_zr: np.ndarray) -> np.ndarray:
    return np.asarray(field_zr, dtype=np.float64).T


def _rho_ext_from_rho_zr(rho_zr: np.ndarray) -> np.ndarray:
    rho_rz = _to_solver_layout_zr_to_rz(rho_zr)
    nr, nz = rho_rz.shape
    ext = np.zeros((nr, nz + 2), dtype=np.float64)
    ext[:, 1:-1] = rho_rz
    ext[:, 0] = rho_rz[:, 0]
    ext[:, -1] = rho_rz[:, -1]
    return ext


def _load_override_zr(file_path: str, z_len: int, r_len: int, var_hint: str) -> np.ndarray:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"覆盖文件不存在: {file_path}")

    suffix = path.suffix.lower()
    arr = None

    if suffix == ".npy":
        arr = np.asarray(np.load(path), dtype=np.float64)
    elif suffix == ".npz":
        payload = np.load(path)
        if var_hint in payload:
            arr = np.asarray(payload[var_hint], dtype=np.float64)
        elif len(payload.files) > 0:
            arr = np.asarray(payload[payload.files[0]], dtype=np.float64)
        else:
            raise ValueError(f"npz 文件为空: {file_path}")
    elif suffix in (".nc", ".nc4", ".cdf"):
        ds = xr.open_dataset(path, decode_cf=False)
        try:
            pick = var_hint if var_hint in ds.variables else (list(ds.data_vars.keys())[0] if len(ds.data_vars) > 0 else None)
            if pick is None:
                raise ValueError(f"NetCDF 无数据变量: {file_path}")
            da = ds[pick]
            data = np.asarray(da, dtype=np.float64)
            dims = list(da.dims)
            if "time" in dims:
                it = dims.index("time")
                data = np.take(data, indices=0, axis=it)
                dims.pop(it)
            if tuple(dims) == ("zh", "radius"):
                arr = data
            elif tuple(dims) == ("radius", "zh"):
                arr = data.T
            else:
                raise ValueError(f"NetCDF 维度需为 ('zh','radius') 或 ('radius','zh')，实际 {tuple(dims)}")
        finally:
            ds.close()
    else:
        raise ValueError(f"不支持的覆盖文件格式: {file_path}")

    if arr.ndim != 2:
        raise ValueError(f"覆盖场必须是二维，实际 ndim={arr.ndim}")
    if arr.shape != (z_len, r_len):
        raise ValueError(f"覆盖场尺寸应为 (nz,nr)=({z_len},{r_len})，实际 {arr.shape}")
    return arr


def _resolve_sources_zr(avg: Dict[str, np.ndarray], cfg: PipelineConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
    nz, nr = avg["theta"].shape
    q = np.array(avg["Q"], copy=True, dtype=np.float64)
    fnu = np.array(avg["Fnu"], copy=True, dtype=np.float64)
    q_src = str(avg.get("Q_used", np.array(["unknown"]))[0])
    fnu_src = str(avg.get("Fnu_used", np.array(["unknown"]))[0])

    if cfg.q_override_file:
        q = _load_override_zr(cfg.q_override_file, nz, nr, "Q")
        q_src = f"override:{cfg.q_override_file}"
    elif q_src == "zero" and cfg.q_constant != 0.0:
        q = np.full((nz, nr), float(cfg.q_constant), dtype=np.float64)
        q_src = f"constant:{cfg.q_constant}"

    if cfg.fnu_override_file:
        fnu = _load_override_zr(cfg.fnu_override_file, nz, nr, "Fnu")
        fnu_src = f"override:{cfg.fnu_override_file}"
    elif fnu_src == "zero" and cfg.fnu_constant != 0.0:
        fnu = np.full((nz, nr), float(cfg.fnu_constant), dtype=np.float64)
        fnu_src = f"constant:{cfg.fnu_constant}"

    return q, fnu, {"Q_source": q_src, "Fnu_source": fnu_src}


def _write_fortran_unformatted_real32(path: Path, arr: np.ndarray) -> None:
    data = np.asarray(arr, dtype=np.float32, order="F")
    payload = data.tobytes(order="F")
    nbytes = len(payload)
    with open(path, "wb") as f:
        f.write(struct.pack("<i", nbytes))
        f.write(payload)
        f.write(struct.pack("<i", nbytes))


def _safe_write_netcdf(ds_out: xr.Dataset, out_path: Path) -> None:
    out_path = out_path.resolve()
    try:
        ds_out.to_netcdf(out_path)
        return
    except Exception as e_direct:
        direct_err = f"{type(e_direct).__name__}: {e_direct}"

    letter, mapped_root, created_now = _subst_map_dir(out_path.parent)
    if not letter or not mapped_root:
        raise RuntimeError(f"NetCDF写出失败: {direct_err}")

    mapped_file = Path(f"{mapped_root}{out_path.name}")
    try:
        ds_out.to_netcdf(mapped_file)
    except Exception as e_subst:
        raise RuntimeError(
            "NetCDF写出失败。"
            f"\n- 直接写出失败: {direct_err}"
            f"\n- subst映射写出失败: {type(e_subst).__name__}: {e_subst}"
        )
    finally:
        _subst_unmap(letter, created_now)


def plot_solution(r_km: np.ndarray, z_km: np.ndarray, psi_rz: np.ndarray, u_rz: np.ndarray, w_rz: np.ndarray, ut_rz: np.ndarray, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    
    # 绘制 psi, U_se, W_se, Vt (2行2列)
    fields = [psi_rz.T, u_rz.T, w_rz.T, ut_rz]
    titles = ["psi", "U_se (radial)", "W_se (vertical)", "Vt (tangential)"]
    axes_flat = axes.flatten()
    
    for ax, ff, ttl in zip(axes_flat, fields, titles):
        if ttl == "psi":
            # 流函数: 仅等值线, 无填色
            cs = ax.contour(r_km, z_km, ff, levels=8, colors='k', linewidths=1.0)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
        elif "radial" in ttl:
            # U_se: 以出流层(z>8km)最大绝对值为对称colorbar上界, 避免边界层大值压缩高层颜色
            z_mask = z_km >= 8.0
            ff_upper = ff[z_mask, :]
            vmax_upper = max(abs(np.nanmin(ff_upper)), abs(np.nanmax(ff_upper)), 0.5)
            levels = np.linspace(-vmax_upper, vmax_upper, 31)
            im = ax.contourf(r_km, z_km, ff, levels=levels, cmap="RdBu_r", extend="both")
            ax.contour(r_km, z_km, ff, levels=levels[::3], colors='k', alpha=0.4, linewidths=0.6)
        elif "vertical" in ttl:
            vmin, vmax = float(np.nanmin(ff)), float(np.nanmax(ff))
            levels_pos = np.linspace(0, vmax, 16)[1:] if vmax > 0 else []
            levels_neg = np.linspace(vmin, 0, 16)[:-1] if vmin < 0 else []
            levels = np.unique(np.concatenate([levels_neg, levels_pos]))
            im = ax.contourf(r_km, z_km, ff, levels=levels, cmap="RdBu_r", extend="both")
            ax.contour(r_km, z_km, ff, levels=levels[::2], colors='k', alpha=0.4, linewidths=0.6)
        else:
            levels = np.linspace(-20, 20, 41)
            im = ax.contourf(r_km, z_km, ff, levels=levels, cmap="RdBu_r", extend="both")
            ax.contour(r_km, z_km, ff, levels=levels[::3], colors='k', alpha=0.4, linewidths=0.6)
            
        ax.set_title(ttl)
        ax.set_xlabel("Radius (km)")
        ax.set_ylabel("Height (km)")
        ax.set_ylim(0, 20)
        ax.set_box_aspect(3/4)  # 子图数据区 4:3 横版
        if ttl != "psi":
            plt.colorbar(im, ax=ax, fraction=0.046)

    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def plot_forcing(r_km: np.ndarray, z_km: np.ndarray, forcing_total: np.ndarray, forcing_thermal: np.ndarray, forcing_momentum: np.ndarray, q_2d: np.ndarray, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    from matplotlib.colors import TwoSlopeNorm
    fields = [forcing_total, forcing_thermal, forcing_momentum, q_2d]
    titles = ["Total forcing", "Thermal forcing", "Momentum-source forcing", "Diabatic heating Q (K/s)"]
    axes_flat = axes.flatten()

    for ax, ff, ttl in zip(axes_flat, fields, titles):
        vmin, vmax = float(np.nanmin(ff)), float(np.nanmax(ff))
        levels_pos = np.linspace(0, vmax, 16)[1:] if vmax > 0 else []
        levels_neg = np.linspace(vmin, 0, 16)[:-1] if vmin < 0 else []
        levels = np.unique(np.concatenate([levels_neg, levels_pos]))
        if len(levels) < 2:
            levels = np.linspace(vmin, vmax, 31)
        norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax) if (vmin<0 and vmax>0) else None
        im = ax.contourf(r_km, z_km, ff, levels=levels, cmap="RdBu_r", norm=norm, extend="both")
        ax.contour(r_km, z_km, ff, levels=levels[::2], colors='k', alpha=0.4, linewidths=0.6)
        ax.set_title(ttl)
        ax.set_xlabel("Radius (km)")
        ax.set_ylabel("Height (km)")
        ax.set_ylim(0, 20)
        ax.set_box_aspect(3/4)
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def run_pipeline(cfg: PipelineConfig) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    avg = azimuthal_average_from_3d(cfg)
    r_km = avg["r_km"]
    z_km = avg["z_km"]
    r_m = r_km * 1000.0
    z_m = z_km * 1000.0

    q_base, fnu_base, source_info = _resolve_sources_zr(avg, cfg)
    q_mod, fnu_mod = _apply_source_masks(q_base, fnu_base, r_km, z_km, cfg.source_mask)

    theta_bal = invert_theta_from_thermal_wind(
        ut_2d=avg["ut"],
        theta_2d_model=avg["theta"],
        r_m=r_m,
        z_m=z_m,
        f=cfg.coriolis_f,
        theta_floor=cfg.theta_floor,
        smooth_window=cfg.theta_outer_smooth_window,
    )

    fields = build_se_diagnostic_fields(
        ut_2d=avg["ut"],
        theta_bal_2d=theta_bal,
        rho_2d=avg["rho"],
        q_2d=q_mod,
        fnu_2d=fnu_mod,
        r_m=r_m,
        z_m=z_m,
        f=cfg.coriolis_f,
    )

    K1_reg, K2_reg, K3_reg, reg_info = regularize_inertial_stability_for_ellipticity(
        fields=fields,
        r_m=r_m,
        z_m=z_m,
        margin=cfg.elliptic_margin,
        eps_ratio=cfg.inertia_eps_ratio,
        max_iter=cfg.regularization_max_iter,
        baroclinic_scale=cfg.baroclinic_scale,
    )

    coef = build_se_coefficients(
        fields, r_m, z_m, 
        K1_override=K1_reg,
        K2_override=K2_reg,
        K3_override=K3_reg,
        baroclinic_scale=cfg.baroclinic_scale,
    )

    # 参考NCL: 修复系数中的NaN/Inf (用邻居值填充)
    for key in ("A", "B", "C", "D", "E", "F"):
        coef[key] = _repair_nan_2d(coef[key])
    coef["discriminant"] = _repair_nan_2d(coef["discriminant"])
    coef["forcing_total"] = _repair_nan_2d(coef["forcing_total"])
    coef["forcing_thermal"] = _repair_nan_2d(coef["forcing_thermal"])
    coef["forcing_momentum"] = _repair_nan_2d(coef["forcing_momentum"])

    disc = coef["discriminant"]
    margin_eff = float(reg_info.get("effective_margin", cfg.elliptic_margin))
    bad_disc = (~np.isfinite(disc)) | (disc <= margin_eff)
    n_bad_disc = int(np.count_nonzero(bad_disc))
    severe_bad = (~np.isfinite(disc)) | (disc < -max(abs(margin_eff), 1.0e-20))
    n_severe_bad = int(np.count_nonzero(severe_bad))
    if n_severe_bad > 0:
        min_disc = float(np.nanmin(disc)) if np.isfinite(disc).any() else float("nan")
        raise FloatingPointError(
            "正则化后仍存在显著非椭圆网格点，停止进入SOR。"
            f" severe_bad_points={n_severe_bad}, min_discriminant={min_disc:.3e}, "
            f"effective_margin={margin_eff:.3e}."
            "建议检查源项尺度或增大 --inertia-eps-ratio / --regularization-max-iter。"
        )
    if n_bad_disc > 0:
        min_disc = float(np.nanmin(disc)) if np.isfinite(disc).any() else float("nan")
        print(
            "[WARN] 正则化后仍有弱非椭圆点，继续尝试SOR: "
            f"bad_points={n_bad_disc}, min_discriminant={min_disc:.3e}, "
            f"effective_margin={margin_eff:.3e}"
        )

    A_rz = _to_solver_layout_zr_to_rz(coef["A"])
    B_rz = _to_solver_layout_zr_to_rz(coef["B"])
    C_rz = _to_solver_layout_zr_to_rz(coef["C"])
    D_rz = _to_solver_layout_zr_to_rz(coef["D"])
    E_rz = _to_solver_layout_zr_to_rz(coef["E"])
    F_rz = _to_solver_layout_zr_to_rz(coef["F"])

    dr_val = float(np.mean(np.diff(r_m)))
    dz_val = float(np.mean(np.diff(z_m)))

    psi = solve_se_sor(
        A=A_rz, B=B_rz, C=C_rz, D=D_rz, E=E_rz, F=F_rz,
        dr=dr_val, dz=dz_val,
        max_iter=cfg.sor_max_iter, omega=cfg.sor_omega,
        tol=cfg.sor_tol, verbose_every=cfg.sor_verbose_every,
    )

    rho_ext = _rho_ext_from_rho_zr(avg["rho"])
    U_se, W_se = psi_to_uw(
        psi=psi,
        rho_ext=rho_ext,
        r_m=r_m,
        dr=float(np.mean(np.diff(r_m))),
        dz=float(np.mean(np.diff(z_m))),
    )

    plot_forcing(
        r_km=r_km,
        z_km=z_km,
        forcing_total=coef["forcing_total"],
        forcing_thermal=coef["forcing_thermal"],
        forcing_momentum=coef["forcing_momentum"],
        q_2d=q_mod,
        out_png=out_dir / "se_forcing_terms.png",
    )

    if cfg.plot_solution:
        plot_solution(
            r_km=r_km,
            z_km=z_km,
            psi_rz=psi[:, 1:-1],
            u_rz=U_se[:, 1:-1],
            w_rz=W_se[:, 1:-1],
            ut_rz=avg["ut"],  # 传入切向风
            out_png=out_dir / "se_solution_fields.png",
        )

    if cfg.write_ieee:
        _write_fortran_unformatted_real32(out_dir / f"{cfg.ieee_prefix}-A.ieee", A_rz)
        _write_fortran_unformatted_real32(out_dir / f"{cfg.ieee_prefix}-B.ieee", B_rz)
        _write_fortran_unformatted_real32(out_dir / f"{cfg.ieee_prefix}-C.ieee", C_rz)
        _write_fortran_unformatted_real32(out_dir / f"{cfg.ieee_prefix}-D.ieee", D_rz)
        _write_fortran_unformatted_real32(out_dir / f"{cfg.ieee_prefix}-E.ieee", E_rz)
        _write_fortran_unformatted_real32(out_dir / f"{cfg.ieee_prefix}-F.ieee", F_rz)
        _write_fortran_unformatted_real32(out_dir / f"{cfg.ieee_prefix}-rho.ieee", rho_ext)
        _write_fortran_unformatted_real32(out_dir / f"{cfg.ieee_prefix}-psi.ieee", psi)
        _write_fortran_unformatted_real32(out_dir / f"{cfg.ieee_prefix}-U.ieee", U_se)
        _write_fortran_unformatted_real32(out_dir / f"{cfg.ieee_prefix}-W.ieee", W_se)

    np.savez_compressed(
        out_dir / "se_pipeline_products.npz",
        r_km=r_km,
        z_km=z_km,
        ur=avg["ur"],
        ut=avg["ut"],
        w=avg["w"],
        prs=avg["prs"],
        rho=avg["rho"],
        theta_model=avg["theta"],
        theta_bal=theta_bal,
        Q=q_mod,
        Fnu=fnu_mod,
        Q_dtheta_dt=avg.get("Q_dtheta_dt", np.zeros_like(q_mod)),
        Q_eddy_adv=avg.get("Q_eddy_adv", np.zeros_like(q_mod)),
        V_mzeta=avg.get("V_mzeta", np.zeros_like(fnu_mod)),
        V_ezeta=avg.get("V_ezeta", np.zeros_like(fnu_mod)),
        V_ev=avg.get("V_ev", np.zeros_like(fnu_mod)),
        V_dh=avg.get("V_dh", np.zeros_like(fnu_mod)),
        V_dv=avg.get("V_dv", np.zeros_like(fnu_mod)),
        tramp=avg.get("tramp", np.zeros_like(fnu_mod)),
        A=coef["A"],
        B=coef["B"],
        C=coef["C"],
        D=coef["D"],
        E=coef["E"],
        F=coef["F"],
        forcing_total=coef["forcing_total"],
        forcing_thermal=coef["forcing_thermal"],
        forcing_momentum=coef["forcing_momentum"],
        discriminant=coef["discriminant"],
        inertial_stability_raw=fields["inertial_stability"],
        inertial_stability_reg=K3_reg / np.maximum(fields["chi"], 1e-10), # approximate back to expected variable logic
        psi=psi,
        U_se=U_se,
        W_se=W_se,
    )

    if cfg.write_netcdf:
        ds_out = xr.Dataset(
            coords={
                "zh": ("zh", z_km),
                "radius": ("radius", r_km),
                "zh_ext": ("zh_ext", np.arange(len(z_km) + 2, dtype=np.float64)),
                "radius_rz": ("radius_rz", r_km),
            },
            data_vars={
                "ur": (("zh", "radius"), avg["ur"]),
                "ut": (("zh", "radius"), avg["ut"]),
                "w": (("zh", "radius"), avg["w"]),
                "prs": (("zh", "radius"), avg["prs"]),
                "rho": (("zh", "radius"), avg["rho"]),
                "theta_model": (("zh", "radius"), avg["theta"]),
                "theta_bal": (("zh", "radius"), theta_bal),
                "Q": (("zh", "radius"), q_mod),
                "Fnu": (("zh", "radius"), fnu_mod),
                "Q_dtheta_dt": (("zh", "radius"), avg.get("Q_dtheta_dt", np.zeros_like(q_mod))),
                "Q_eddy_adv": (("zh", "radius"), avg.get("Q_eddy_adv", np.zeros_like(q_mod))),
                "V_mzeta": (("zh", "radius"), avg.get("V_mzeta", np.zeros_like(fnu_mod))),
                "V_ezeta": (("zh", "radius"), avg.get("V_ezeta", np.zeros_like(fnu_mod))),
                "V_ev": (("zh", "radius"), avg.get("V_ev", np.zeros_like(fnu_mod))),
                "V_dh": (("zh", "radius"), avg.get("V_dh", np.zeros_like(fnu_mod))),
                "V_dv": (("zh", "radius"), avg.get("V_dv", np.zeros_like(fnu_mod))),
                "tramp": (("zh", "radius"), avg.get("tramp", np.zeros_like(fnu_mod))),
                "A": (("zh", "radius"), coef["A"]),
                "B": (("zh", "radius"), coef["B"]),
                "C": (("zh", "radius"), coef["C"]),
                "D": (("zh", "radius"), coef["D"]),
                "E": (("zh", "radius"), coef["E"]),
                "F": (("zh", "radius"), coef["F"]),
                "forcing_total": (("zh", "radius"), coef["forcing_total"]),
                "forcing_thermal": (("zh", "radius"), coef["forcing_thermal"]),
                "forcing_momentum": (("zh", "radius"), coef["forcing_momentum"]),
                "discriminant": (("zh", "radius"), coef["discriminant"]),
                "inertial_stability_raw": (("zh", "radius"), fields["inertial_stability"]),
                "inertial_stability_reg": (("zh", "radius"), K3_reg / np.maximum(fields["chi"], 1e-10)),
                "psi": (("radius_rz", "zh_ext"), psi),
                "U_se": (("radius_rz", "zh_ext"), U_se),
                "W_se": (("radius_rz", "zh_ext"), W_se),
            },
            attrs={
                "center_x_km": float(avg["center_x_km"][0]),
                "center_y_km": float(avg["center_y_km"][0]),
                "Q_source": source_info["Q_source"],
                "Fnu_source": source_info["Fnu_source"],
            },
        )
        _safe_write_netcdf(ds_out, out_dir / "se_pipeline_products.nc")

    summary = {
        "center_x_km": float(avg["center_x_km"][0]),
        "center_y_km": float(avg["center_y_km"][0]),
        "time_index_used": int(avg.get("time_index_used", np.array([cfg.time_index]))[0]),
        "time_seconds_used": float(avg.get("time_seconds_used", np.array([0.0]))[0]),
        "time_selection_method": str(avg.get("time_selection_method", np.array(["time_index"]))[0]),
        "dr_km_requested": float(avg.get("dr_km_requested", np.array([cfg.dr_km]))[0]),
        "dr_km_used": float(avg.get("dr_km_used", np.array([cfg.dr_km]))[0]),
        "grid_dx_km": float(avg.get("grid_dx_km", np.array([np.nan]))[0]),
        "grid_dy_km": float(avg.get("grid_dy_km", np.array([np.nan]))[0]),
        "dataset_open_path": str(avg.get("dataset_open_path", np.array([""]))[0]),
        "dataset_engine": str(avg.get("dataset_engine", np.array([""]))[0]),
        "variables_used": {
            "u": str(avg.get("u_used", np.array([""]))[0]),
            "v": str(avg.get("v_used", np.array([""]))[0]),
            "w": str(avg.get("w_used", np.array([""]))[0]),
            "prs": str(avg.get("prs_used", np.array([""]))[0]),
            "rho": str(avg.get("rho_used", np.array([""]))[0]),
            "theta": str(avg.get("theta_used", np.array([""]))[0]),
            "psfc": str(avg.get("psfc_used", np.array([""]))[0]),
        },
        "source_info": source_info,
        "momentum_budget_pairs_used": [str(x) for x in avg.get("momentum_budget_pairs_used", np.array([], dtype=object))],
        "regularization": reg_info,
        "solver_bc": {
            "radial": "dpsi/dr=0 at inner and outer radius",
            "vertical": "psi=0 at bottom, dpsi/dz=0 at top (自由滑移)",
        },
        "forcing_png": str((out_dir / "se_forcing_terms.png").as_posix()),
        "solution_png": str((out_dir / "se_solution_fields.png").as_posix()) if cfg.plot_solution else "",
        "npz": str((out_dir / "se_pipeline_products.npz").as_posix()),
        "nc": str((out_dir / "se_pipeline_products.nc").as_posix()) if cfg.write_netcdf else "",
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Pipeline finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _source_mask_from_json(text_or_path: Optional[str]) -> SourceMaskConfig:
    if not text_or_path:
        return SourceMaskConfig()

    p = Path(text_or_path)
    if p.exists():
        payload = json.loads(p.read_text(encoding="utf-8"))
    else:
        payload = json.loads(text_or_path)

    return SourceMaskConfig(
        thermal_scale=float(payload.get("thermal_scale", 1.0)),
        momentum_scale=float(payload.get("momentum_scale", 1.0)),
        thermal_zero_boxes=list(payload.get("thermal_zero_boxes", [])),
        momentum_zero_boxes=list(payload.get("momentum_zero_boxes", [])),
    )


def parse_args() -> PipelineConfig:
    p = argparse.ArgumentParser(description="SE 诊断全流程: 3D场 -> 方位平均 -> 热成风反算 -> 六系数 -> 正则化 -> SE求解")
    p.add_argument("--input-file", default="dataset/cm1out.nc")
    p.add_argument("--output-dir", default="se_pipeline_output")
    p.add_argument("--time-index", type=int, default=0)
    p.add_argument("--target-time-seconds", type=float, default=None, help="按time坐标秒值选取最近时次，例如172800")
    p.add_argument("--target-time-hours", type=float, default=None, help="按小时选取最近时次，例如48")
    p.add_argument("--time-avg-start-hours", type=float, default=None, help="时间段平均起始小时")
    p.add_argument("--time-avg-end-hours", type=float, default=None, help="时间段平均终止小时")

    p.add_argument("--u-name", default="u")
    p.add_argument("--v-name", default="v")
    p.add_argument("--w-name", default="w")
    p.add_argument("--prs-name", default="prs")
    p.add_argument("--rho-name", default="rho")
    p.add_argument("--theta-name", default="th")
    p.add_argument("--psfc-name", default="psfc")
    p.add_argument("--u-candidates", default="u,ua,uinterp")
    p.add_argument("--v-candidates", default="v,va,vinterp")
    p.add_argument("--w-candidates", default="w,wa,winterp")
    p.add_argument("--prs-candidates", default="prs,pres,p")
    p.add_argument("--rho-candidates", default="rho,rhoa,dens")
    p.add_argument("--theta-candidates", default="th,theta,thpert")
    p.add_argument("--psfc-candidates", default="psfc,sfcprs,ps")
    p.add_argument("--q-name", default="Q")
    p.add_argument("--fnu-name", default="Fnu")
    p.add_argument("--q-candidates", default="Q,q_diab,qheat,th_src")
    p.add_argument("--fnu-candidates", default="Fnu,fric_radial,mom_src,radial_drag")
    p.add_argument("--q-override-file", default="", help="外部热力源二维场文件(.npy/.npz/.nc)，维度(zh,radius)")
    p.add_argument("--fnu-override-file", default="", help="外部动量源二维场文件(.npy/.npz/.nc)，维度(zh,radius)")
    p.add_argument("--q-constant", type=float, default=0.0, help="当热力源缺失且未提供覆盖文件时使用常数")
    p.add_argument("--fnu-constant", type=float, default=0.0, help="当动量源缺失且未提供覆盖文件时使用常数")

    p.add_argument("--max-r-km", type=float, default=300.0)
    p.add_argument("--dr-km", type=float, default=2.0)
    p.add_argument("--allow-fine-radial-bins", action="store_true", help="允许dr小于原始水平网格距(不建议，可能出现条纹欠采样)")
    p.add_argument("--max-z-km", type=float, default=20.0)
    p.add_argument("--center-window", type=int, default=21)
    p.add_argument("--center-method", choices=["min", "mean"], default="min")

    p.add_argument("--f", type=float, default=5.0e-5)
    p.add_argument("--theta-floor", type=float, default=150.0)
    p.add_argument("--theta-outer-smooth-window", type=int, default=1)

    p.add_argument("--elliptic-margin", type=float, default=0.0)
    p.add_argument("--inertia-eps-ratio", type=float, default=1.0e-3)
    p.add_argument("--regularization-max-iter", type=int, default=20)

    p.add_argument("--sor-max-iter", type=int, default=60000)
    p.add_argument("--sor-omega", type=float, default=1.8)
    p.add_argument("--sor-tol", type=float, default=1.0e-14)
    p.add_argument("--sor-verbose-every", type=int, default=500)

    p.add_argument("--no-write-netcdf", action="store_true")
    p.add_argument("--no-write-ieee", action="store_true")
    p.add_argument("--ieee-prefix", default="SE")
    p.add_argument("--no-plot-solution", action="store_true")

    p.add_argument("--baroclinic-scale", type=float, default=0.4,
                   help="NCL式斜压项缩放因子 (NCL bb*0.4), 默认0.4以增强椭圆性")

    p.add_argument(
        "--source-mask-json",
        default="",
        help=(
            "源项开关接口(JSON字符串或文件路径)。"
            "示例: {\"thermal_scale\":1,\"momentum_scale\":1,"
            "\"thermal_zero_boxes\":[{\"r_min_km\":0,\"r_max_km\":50,\"z_min_km\":0,\"z_max_km\":3}],"
            "\"momentum_zero_boxes\":[] }"
        ),
    )

    args = p.parse_args()

    if args.target_time_seconds is not None and args.target_time_hours is not None:
        raise ValueError("--target-time-seconds 与 --target-time-hours 不能同时指定")

    return PipelineConfig(
        input_file=args.input_file,
        output_dir=args.output_dir,
        time_index=args.time_index,
        target_time_seconds=args.target_time_seconds,
        target_time_hours=args.target_time_hours,
        time_avg_start_hours=args.time_avg_start_hours,
        time_avg_end_hours=args.time_avg_end_hours,
        u_name=args.u_name,
        v_name=args.v_name,
        w_name=args.w_name,
        prs_name=args.prs_name,
        rho_name=args.rho_name,
        theta_name=args.theta_name,
        psfc_name=args.psfc_name,
        u_candidates=_parse_csv_names(args.u_candidates),
        v_candidates=_parse_csv_names(args.v_candidates),
        w_candidates=_parse_csv_names(args.w_candidates),
        prs_candidates=_parse_csv_names(args.prs_candidates),
        rho_candidates=_parse_csv_names(args.rho_candidates),
        theta_candidates=_parse_csv_names(args.theta_candidates),
        psfc_candidates=_parse_csv_names(args.psfc_candidates),
        q_name=args.q_name,
        fnu_name=args.fnu_name,
        q_candidates=_parse_csv_names(args.q_candidates),
        fnu_candidates=_parse_csv_names(args.fnu_candidates),
        q_override_file=args.q_override_file,
        fnu_override_file=args.fnu_override_file,
        q_constant=args.q_constant,
        fnu_constant=args.fnu_constant,
        max_r_km=args.max_r_km,
        dr_km=args.dr_km,
        enforce_dr_not_finer_than_grid=(not args.allow_fine_radial_bins),
        max_z_km=args.max_z_km,
        center_window=args.center_window,
        center_method=args.center_method,
        coriolis_f=args.f,
        theta_floor=args.theta_floor,
        theta_outer_smooth_window=args.theta_outer_smooth_window,
        elliptic_margin=args.elliptic_margin,
        inertia_eps_ratio=args.inertia_eps_ratio,
        regularization_max_iter=args.regularization_max_iter,
        sor_max_iter=args.sor_max_iter,
        sor_omega=args.sor_omega,
        sor_tol=args.sor_tol,
        sor_verbose_every=args.sor_verbose_every,
        write_netcdf=(not args.no_write_netcdf),
        write_ieee=(not args.no_write_ieee),
        ieee_prefix=args.ieee_prefix,
        plot_solution=(not args.no_plot_solution),
        baroclinic_scale=args.baroclinic_scale,
        source_mask=_source_mask_from_json(args.source_mask_json),
    )


if __name__ == "__main__":
    run_pipeline(parse_args())
