"""
方位角平均与动量收支诊断 (Azimuthal Average & Budget Diagnostics)

CM1 笛卡尔网格 → 柱坐标转换 → 方位角平均 → 径向(u)和切向(v)动量收支分解。

核心流水线:
  run_budget_diagnostic(cfg: TransformConfig) -> None
    读取 CM1 输出，执行坐标变换和方位角平均，
    计算 mean/eddy 分解后的动量收支各项，输出 NetCDF。
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from netCDF4 import Dataset

from .center_finder import find_smoothed_min_point
from .coordinates import (
    destagger_axis,
    destagger_to_scalar_grid,
    ensure_2d_xy,
    require_dims,
)
from .io import copy_nc_attrs, first_existing_var

# ==============================================================================
# 常量
# ==============================================================================

U_BASE = "u"
V_BASE = "v"
W_BASE = "w"
PRS_VAR = "prs"
RHO_VAR = "rho"
PSFC_VAR = "psfc"

U_BUDGET_PREFIX = "ub_"
V_BUDGET_PREFIX = "vb_"

# ==============================================================================
# 诊断项元数据 (DIAG_META)
# ==============================================================================

DIAG_META: Dict[str, str] = {
    # --- 径向动量收支 (Radial) ---
    "U_mr": "mean radial advection magnitude: ubar * d(ubar)/dr",
    "U_eh": "eddy horizontal advection magnitude (from hadv decomposition)",
    "U_mv": "mean vertical advection magnitude: wbar * d(ubar)/dz",
    "U_ev": "eddy vertical advection magnitude (from vadv decomposition)",
    "U_magf": "mean agradient-force group: vbar^2/r + coriolis + mean pressure-gradient",
    "U_eagf": "eddy agradient-force group: v'^2/r + eddy pressure-gradient",
    "U_dh": "horizontal diffusion+turbulence: hidiff + hturb",
    "U_dv": "vertical diffusion+turbulence: vidiff + vturb",
    "ramp": "radial damping term",
    "coriolis": "radial coriolis term from projected budget",
    "pgrad_mean": "mean pressure-gradient term: -(1/rhobar) d(pbar)/dr",
    "pgrad_eddy": "eddy pressure-gradient term: total pgrad - mean pgrad",
    "curv_mean": "mean curvature term: vbar^2/r",
    "curv_eddy": "eddy curvature term: overline(v'^2)/r",
    "br_total_raw": "sum of all projected radial budget terms",
    "tendency_model_raw": "reconstructed tendency from groups before residual allocation",
    "tendency_model_adjusted": "reconstructed tendency from groups after residual allocation",
    "residual_raw": "closure residual before grouped allocation",
    "residual_after_allocation": "closure residual after grouped allocation",

    # --- 切向动量收支 (Tangential) ---
    "V_mr": "mean radial advection on tangential wind: ubar * d(vbar)/dr",
    "V_eh": "eddy horizontal advection magnitude for tangential wind",
    "V_mv": "mean vertical advection on tangential wind: wbar * d(vbar)/dz",
    "V_ev": "eddy vertical advection magnitude for tangential wind",
    "V_magf": "mean tangential force group: -ubar*vbar/r + coriolis",
    "V_eagf": "eddy tangential force group: -overline(u'v')/r + eddy azimuthal pressure-gradient",
    "V_dh": "tangential horizontal diffusion+turbulence: hidiff + hturb",
    "V_dv": "tangential vertical diffusion+turbulence: vidiff + vturb",
    "tramp": "tangential damping term",
    "vcurv_mean": "mean tangential curvature term: ubar*vbar/r",
    "vcurv_eddy": "eddy tangential curvature term: overline(u'v')/r",
    "coriolis_t": "tangential coriolis term from projected budget",
    "pgrad_t": "tangential pressure-gradient term from projected budget",
    "bt_total_raw": "sum of all projected tangential budget terms",
    "tendency_t_model_raw": "reconstructed tangential tendency before residual allocation",
    "tendency_t_model_adjusted": "reconstructed tangential tendency after residual allocation",
    "residual_t_raw": "tangential closure residual before grouped allocation",
    "residual_t_after_allocation": "tangential closure residual after grouped allocation",
}


# ==============================================================================
# 辅助函数
# ==============================================================================

def get_time_slice_nc(
    var_in, t_idx: int
) -> Tuple[np.ndarray, List[str]]:
    """从 netCDF4 变量对象提取单个时间切片。"""
    dims = list(var_in.dimensions)
    data = np.asarray(
        var_in[tuple(t_idx if d == "time" else slice(None) for d in dims)],
        dtype=np.float64
    )
    return data, [d for d in dims if d != "time"]


def pair_u_v_budget_terms(
    var_names: Iterable[str]
) -> List[Tuple[str, str, str]]:
    """
    配对 u/v 预算项。

    Returns: List of (suffix, u_name, v_name).
    """
    names = set(var_names)
    pairs: List[Tuple[str, str, str]] = []
    for u_name in sorted(n for n in names if n.startswith(U_BUDGET_PREFIX)):
        suffix = u_name[len(U_BUDGET_PREFIX):]
        v_name = f"{V_BUDGET_PREFIX}{suffix}"
        if v_name in names:
            pairs.append((suffix, u_name, v_name))
    return pairs


def compute_radial_bin_index(
    r2d: np.ndarray, r_bins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """计算每个网格点对应的径向 bin 索引。"""
    nr = len(r_bins) - 1
    idx = np.digitize(r2d.ravel(), r_bins) - 1
    valid = (idx >= 0) & (idx < nr)
    return idx, valid


def azimuthal_average_by_radius(
    data_3d: np.ndarray,
    bin_index_1d: np.ndarray,
    valid_mask_1d: np.ndarray,
    nr: int,
) -> np.ndarray:
    """按径向 bin 做方位角平均。"""
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


def eddy_variance_by_radius(
    data_3d: np.ndarray,
    mean_2d: np.ndarray,
    bin_index_1d: np.ndarray,
    valid_mask_1d: np.ndarray,
    nr: int,
) -> np.ndarray:
    """计算涡动方差 v'^2 的方位角平均。"""
    nz = data_3d.shape[0]
    out = np.full((nz, nr), np.nan, dtype=np.float64)
    for level in range(nz):
        flat = data_3d[level].ravel()
        mean_flat = np.full(flat.shape, np.nan, dtype=np.float64)
        in_range = valid_mask_1d & (bin_index_1d >= 0) & (bin_index_1d < nr)
        mean_flat[in_range] = mean_2d[level, bin_index_1d[in_range]]
        use = in_range & np.isfinite(flat) & np.isfinite(mean_flat)
        if not np.any(use):
            continue
        idx = bin_index_1d[use]
        var_vals = (flat[use] - mean_flat[use]) ** 2
        count = np.bincount(idx, minlength=nr)
        summation = np.bincount(idx, weights=var_vals, minlength=nr)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[level] = summation / count
    return out


def eddy_covariance_by_radius(
    a_3d: np.ndarray,
    b_3d: np.ndarray,
    mean_a_2d: np.ndarray,
    mean_b_2d: np.ndarray,
    bin_index_1d: np.ndarray,
    valid_mask_1d: np.ndarray,
    nr: int,
) -> np.ndarray:
    """计算涡动协方差 u'v' 的方位角平均。"""
    nz = a_3d.shape[0]
    out = np.full((nz, nr), np.nan, dtype=np.float64)
    for level in range(nz):
        a_flat = a_3d[level].ravel()
        b_flat = b_3d[level].ravel()
        mean_a_flat = np.full(a_flat.shape, np.nan, dtype=np.float64)
        mean_b_flat = np.full(b_flat.shape, np.nan, dtype=np.float64)
        in_range = valid_mask_1d & (bin_index_1d >= 0) & (bin_index_1d < nr)
        mean_a_flat[in_range] = mean_a_2d[level, bin_index_1d[in_range]]
        mean_b_flat[in_range] = mean_b_2d[level, bin_index_1d[in_range]]
        use = (
            in_range & np.isfinite(a_flat) & np.isfinite(b_flat)
            & np.isfinite(mean_a_flat) & np.isfinite(mean_b_flat)
        )
        if not np.any(use):
            continue
        idx = bin_index_1d[use]
        cov_vals = (a_flat[use] - mean_a_flat[use]) * (b_flat[use] - mean_b_flat[use])
        count = np.bincount(idx, minlength=nr)
        summation = np.bincount(idx, weights=cov_vals, minlength=nr)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[level] = summation / count
    return out


def safe_gradient(
    field_2d: np.ndarray, coords_1d: np.ndarray, axis: int
) -> np.ndarray:
    """安全计算梯度（处理单列情况）。"""
    if field_2d.shape[axis] < 2:
        return np.zeros_like(field_2d, dtype=np.float64)
    return np.gradient(field_2d, coords_1d, axis=axis, edge_order=1)


def pick_term(
    term_map: Dict[str, np.ndarray],
    candidates: Sequence[str],
    shape: Tuple[int, int],
) -> np.ndarray:
    """从 term_map 中选取第一个存在的项，否则返回零场。"""
    for name in candidates:
        if name in term_map:
            return term_map[name]
    return np.zeros(shape, dtype=np.float64)


def moving_average_1d(arr: np.ndarray, window: int) -> np.ndarray:
    """一维滑动平均。"""
    win = int(window)
    if win <= 1:
        return np.asarray(arr, dtype=np.float64)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    padded = np.pad(np.asarray(arr, dtype=np.float64), (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(padded, kernel, mode="valid")


# ==============================================================================
# 内核轴对称约束
# ==============================================================================

def apply_core_axisymmetric_constraints(
    ur_avg: np.ndarray,
    ut_avg: np.ndarray,
    ut_prime2_avg: np.ndarray,
    p_avg: np.ndarray,
    r_m: np.ndarray,
    core_radius_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    对 r ≈ 0 处施加轴对称约束，消除柱坐标奇点噪声。

    ur ~ O(r), ut ~ O(r), p ~ O(r^2), ut'^2 ~ O(r^2).
    """
    core_mask = r_m <= core_radius_m
    if not np.any(core_mask):
        return ur_avg, ut_avg, ut_prime2_avg, p_avg

    edge_idx = int(np.where(core_mask)[0][-1])
    r_core = float(r_m[edge_idx])
    if r_core <= 0.0:
        return ur_avg, ut_avg, ut_prime2_avg, p_avg

    ratio = (r_m[core_mask] / r_core).astype(np.float64)
    ratio2 = ratio ** 2

    ur_use = np.array(ur_avg, copy=True)
    ut_use = np.array(ut_avg, copy=True)
    utp2_use = np.array(ut_prime2_avg, copy=True)
    p_use = np.array(p_avg, copy=True)

    ur_edge = ur_avg[:, edge_idx][:, np.newaxis]
    ut_edge = ut_avg[:, edge_idx][:, np.newaxis]
    utp2_edge = ut_prime2_avg[:, edge_idx][:, np.newaxis]
    p0 = p_avg[:, 0][:, np.newaxis]
    p_edge = p_avg[:, edge_idx][:, np.newaxis]

    ur_use[:, core_mask] = ur_edge * ratio[np.newaxis, :]
    ut_use[:, core_mask] = ut_edge * ratio[np.newaxis, :]
    utp2_use[:, core_mask] = utp2_edge * ratio2[np.newaxis, :]
    p_use[:, core_mask] = p0 + (p_edge - p0) * ratio2[np.newaxis, :]

    return ur_use, ut_use, utp2_use, p_use


# ==============================================================================
# 分组残差分配 (Grouped Residual Allocation)
# ==============================================================================

def grouped_residual_allocation_radial(
    U_mr_raw: np.ndarray,
    U_eh_raw: np.ndarray,
    U_mv_raw: np.ndarray,
    U_ev_raw: np.ndarray,
    curv_mean: np.ndarray,
    curv_eddy: np.ndarray,
    coriolis: np.ndarray,
    pgrad_mean: np.ndarray,
    pgrad_eddy: np.ndarray,
    hadv: np.ndarray,
    vadv: np.ndarray,
    pgrad: np.ndarray,
    U_dh_raw: np.ndarray,
    U_dv_raw: np.ndarray,
    ramp_raw: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    径向动量收支的分组残差分配。

    将闭合残差按比例分配至各组内的 mean/eddy 项。
    """
    # Group-1: 水平平流
    residual_h = (-U_mr_raw - U_eh_raw) - (hadv - curv_mean - curv_eddy)
    delta_h = 0.5 * residual_h
    U_mr_adj = U_mr_raw + delta_h
    U_eh_adj = U_eh_raw + delta_h

    # Group-2: 垂直平流
    residual_v = (-U_mv_raw - U_ev_raw) - vadv
    delta_v = 0.5 * residual_v
    U_mv_adj = U_mv_raw + delta_v
    U_ev_adj = U_ev_raw + delta_v

    # Group-3: 气压梯度力
    residual_p = (pgrad_mean + pgrad_eddy) - pgrad
    delta_p = 0.5 * residual_p
    pgrad_mean_adj = pgrad_mean - delta_p
    pgrad_eddy_adj = pgrad_eddy - delta_p

    U_magf_adj = curv_mean + coriolis + pgrad_mean_adj
    U_eagf_adj = curv_eddy + pgrad_eddy_adj

    adjusted_terms = {
        "U_mr": U_mr_adj,
        "U_eh": U_eh_adj,
        "U_mv": U_mv_adj,
        "U_ev": U_ev_adj,
        "U_magf": U_magf_adj,
        "U_eagf": U_eagf_adj,
        "U_dh": U_dh_raw,
        "U_dv": U_dv_raw,
        "ramp": ramp_raw,
    }
    return adjusted_terms, pgrad_mean_adj, pgrad_eddy_adj, residual_h, residual_v, residual_p


def grouped_residual_allocation_tangential(
    V_mr_raw: np.ndarray,
    V_eh_raw: np.ndarray,
    V_mv_raw: np.ndarray,
    V_ev_raw: np.ndarray,
    V_magf_raw: np.ndarray,
    V_eagf_raw: np.ndarray,
    hadv: np.ndarray,
    vadv: np.ndarray,
    vcurv_mean: np.ndarray,
    vcurv_eddy: np.ndarray,
    V_dh_raw: np.ndarray,
    V_dv_raw: np.ndarray,
    tramp_raw: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """切向动量收支的分组残差分配。"""
    # 水平平流
    residual_h = (-V_mr_raw - V_eh_raw) - (hadv + vcurv_mean + vcurv_eddy)
    delta_h = 0.5 * residual_h
    V_mr_adj = V_mr_raw + delta_h
    V_eh_adj = V_eh_raw + delta_h

    # 垂直平流
    residual_v = (-V_mv_raw - V_ev_raw) - vadv
    delta_v = 0.5 * residual_v
    V_mv_adj = V_mv_raw + delta_v
    V_ev_adj = V_ev_raw + delta_v

    adjusted_terms = {
        "V_mr": V_mr_adj,
        "V_eh": V_eh_adj,
        "V_mv": V_mv_adj,
        "V_ev": V_ev_adj,
        "V_magf": V_magf_raw,
        "V_eagf": V_eagf_raw,
        "V_dh": V_dh_raw,
        "V_dv": V_dv_raw,
        "tramp": tramp_raw,
    }
    return adjusted_terms, residual_h, residual_v


# ==============================================================================
# 元数据检查
# ==============================================================================

def inspect_metadata_xarray(input_file: str) -> Dict[str, Dict[str, object]]:
    """检查输入文件中的变量元数据。"""
    import xarray as xr

    target_vars = {U_BASE, V_BASE, W_BASE, PRS_VAR, RHO_VAR, PSFC_VAR}
    with xr.open_dataset(input_file, decode_cf=False) as ds:
        target_vars.update(
            v for v in ds.data_vars
            if v.startswith(U_BUDGET_PREFIX) or v.startswith(V_BUDGET_PREFIX)
        )
        metadata: Dict[str, Dict[str, object]] = {}
        for name in sorted(target_vars):
            if name not in ds.variables:
                continue
            var = ds.variables[name]
            dims = list(var.dims)
            metadata[name] = {
                "name": name,
                "units": var.attrs.get("units", ""),
                "dtype": str(var.dtype),
                "dimensions": dims,
                "coordinate_variables": [d for d in dims if d in ds.variables],
            }
    return metadata


# ==============================================================================
# 主流水线
# ==============================================================================

def run_budget_diagnostic(
    input_file: str = "dataset/cm1out.nc",
    output_file: str = "dataset/typhoon_azimuthal_avg_budget.nc",
    max_r_km: float = 300.0,
    dr_km: float = 2.0,
    max_z_km: float = 20.0,
    center_window: int = 21,
    center_method: str = "min",
    enable_core_stabilization: bool = False,
    core_radius_km: float = 6.0,
    center_time_smooth_window: int = 11,
    subtract_translation_speed: bool = False,
    max_times: int | None = None,
    start_time_idx: int = 0,
    verbose: bool = True,
) -> None:
    """
    执行完整的方位角平均与动量收支诊断流水线。

    读取 CM1 笛卡尔网格输出，转换到柱坐标，做方位角平均，
    计算径向(u)和切向(v)动量收支的 mean/eddy 分解，输出 NetCDF。
    """
    if os.path.exists(output_file):
        os.remove(output_file)

    metadata = inspect_metadata_xarray(input_file)

    with Dataset(input_file, "r") as nc_in, \
         Dataset(output_file, "w", format="NETCDF4") as nc_out:

        # 验证必要变量
        required = ["time", "xh", "yh", "zh", U_BASE, V_BASE, W_BASE,
                     PRS_VAR, RHO_VAR, PSFC_VAR]
        for name in required:
            if name not in nc_in.variables:
                raise KeyError(f"输入文件缺少必要变量: {name}")

        budget_pairs = pair_u_v_budget_terms(nc_in.variables.keys())
        if not budget_pairs:
            raise ValueError("未找到可配对的 u/v 预算项（ub_* 与 vb_*）。")

        # 坐标
        xh = np.asarray(nc_in.variables["xh"][:], dtype=np.float64)
        yh = np.asarray(nc_in.variables["yh"][:], dtype=np.float64)
        zh = np.asarray(nc_in.variables["zh"][:], dtype=np.float64)
        time_values = np.asarray(nc_in.variables["time"][:], dtype=np.float64)

        z_idx = np.where(zh <= max_z_km)[0]
        if z_idx.size == 0:
            raise ValueError(f"没有高度层满足 z <= {max_z_km} km")
        z_values = zh[z_idx]
        z_m = z_values * 1000.0

        r_bins = np.arange(0.0, max_r_km + dr_km, dr_km)
        r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
        r_m = r_centers * 1000.0
        nr = len(r_centers)

        end_time_idx = len(time_values) if max_times is None else min(
            len(time_values), int(start_time_idx) + int(max_times)
        )
        total_times = end_time_idx - int(start_time_idx)

        # 处理 max_times=None 时的时间范围
        actual_time_indices = list(range(int(start_time_idx), end_time_idx))

        # --- 准备输出文件 ---
        for attr_name in nc_in.ncattrs():
            nc_out.setncattr(attr_name, nc_in.getncattr(attr_name))
        nc_out.setncattr(
            "processing",
            "Cartesian->cylindrical transform + azimuthal average + diagnostics"
        )

        nc_out.createDimension(
            "time",
            None if nc_in.dimensions["time"].isunlimited()
            else len(nc_in.dimensions["time"])
        )
        nc_out.createDimension("z", len(z_values))
        nc_out.createDimension("r", len(r_centers))

        out_vars: Dict[str, object] = {}

        var_time_in = nc_in.variables["time"]
        out_vars["time"] = nc_out.createVariable(
            "time", var_time_in.dtype, ("time",)
        )
        copy_nc_attrs(var_time_in, out_vars["time"])

        out_vars["z"] = nc_out.createVariable("z", "f4", ("z",))
        out_vars["z"][:] = z_values.astype(np.float32)
        out_vars["z"].units = "km"
        out_vars["z"].long_name = "height"

        out_vars["r"] = nc_out.createVariable("r", "f4", ("r",))
        out_vars["r"][:] = r_centers.astype(np.float32)
        out_vars["r"].units = "km"
        out_vars["r"].long_name = "radius from TC center"

        for out_name, in_name in {
            "ur": U_BASE, "ut": V_BASE, "w": W_BASE, "prs": PRS_VAR, "rho": RHO_VAR
        }.items():
            var = nc_out.createVariable(out_name, "f4", ("time", "z", "r"), zlib=True)
            copy_nc_attrs(nc_in.variables[in_name], var)
            out_vars[out_name] = var

        out_vars["ur"].long_name = "azimuthal mean radial wind (outward positive)"
        out_vars["ut"].long_name = "azimuthal mean tangential wind"
        out_vars["w"].long_name = "azimuthal mean vertical velocity"
        out_vars["prs"].long_name = "azimuthal mean pressure"
        out_vars["rho"].long_name = "azimuthal mean density"

        for suffix, u_name, v_name in budget_pairs:
            for prefix, pfx in [("br_", "radial"), ("bt_", "tangential")]:
                var_name_nc = f"{prefix}{suffix}"
                var = nc_out.createVariable(
                    var_name_nc, "f4", ("time", "z", "r"), zlib=True
                )
                var.long_name = f"azimuthal mean {pfx} budget component: {suffix}"
                var.coordinates = "time z r"
                out_vars[var_name_nc] = var

        for name, long_name in DIAG_META.items():
            var = nc_out.createVariable(name, "f4", ("time", "z", "r"), zlib=True)
            var.long_name = long_name
            var.units = "m s-2"
            var.coordinates = "time z r"
            out_vars[name] = var

        for key in (
            "U_mr", "U_eh", "U_mv", "U_ev", "U_magf", "U_eagf",
            "U_dh", "U_dv", "ramp",
            "V_mr", "V_eh", "V_mv", "V_ev", "V_magf", "V_eagf",
            "V_dh", "V_dv", "tramp",
        ):
            raw_name = f"{key}_raw"
            var = nc_out.createVariable(raw_name, "f4", ("time", "z", "r"), zlib=True)
            var.long_name = f"raw (before residual allocation): {DIAG_META[key]}"
            var.units = "m s-2"
            var.coordinates = "time z r"
            out_vars[raw_name] = var

        # --- 追踪中心轨迹 ---
        X, Y = np.meshgrid(xh, yh)

        center_x = np.zeros(total_times, dtype=np.float64)
        center_y = np.zeros(total_times, dtype=np.float64)
        for out_idx, t_idx in enumerate(actual_time_indices):
            center = find_smoothed_min_point(
                input_file, time_key=t_idx, var_name=PSFC_VAR,
                x_dim="xh", y_dim="yh", window=center_window,
                verbose=False, center_method=center_method,
            )
            center_x[out_idx] = float(center["x"])
            center_y[out_idx] = float(center["y"])

        if enable_core_stabilization and center_time_smooth_window > 1:
            center_x = moving_average_1d(center_x, center_time_smooth_window)
            center_y = moving_average_1d(center_y, center_time_smooth_window)

        # --- 计算台风移动速度 ---
        cx = np.zeros(total_times, dtype=np.float64)
        cy = np.zeros(total_times, dtype=np.float64)
        if subtract_translation_speed:
            if total_times >= 2 and np.all(np.isfinite(time_values)) \
               and np.all(np.diff(time_values) > 0):
                center_x_m = center_x * 1000.0
                center_y_m = center_y * 1000.0
                cx = np.gradient(center_x_m, time_values)
                cy = np.gradient(center_y_m, time_values)
                if verbose:
                    vmax = np.max(np.hypot(cx, cy))
                    print(f"[INFO] 台风移动速度消减已开启，最大移速={vmax:.2f} m/s")

        # --- 主时间循环 ---
        for out_idx, t_idx in enumerate(actual_time_indices):
            xc = float(center_x[out_idx])
            yc = float(center_y[out_idx])

            r2d = np.sqrt((X - xc) ** 2 + (Y - yc) ** 2)
            theta2d = np.arctan2(Y - yc, X - xc)
            bin_index, valid_mask = compute_radial_bin_index(r2d, r_bins)
            valid_mask &= r2d.ravel() <= max_r_km

            theta3d_sub = np.broadcast_to(
                theta2d[np.newaxis, :, :], (len(z_idx), len(yh), len(xh))
            )

            # 读取并去交错核心变量
            u_raw, u_dims = get_time_slice_nc(nc_in.variables[U_BASE], t_idx)
            v_raw, v_dims = get_time_slice_nc(nc_in.variables[V_BASE], t_idx)
            w_raw, w_dims = get_time_slice_nc(nc_in.variables[W_BASE], t_idx)
            p_raw, p_dims = get_time_slice_nc(nc_in.variables[PRS_VAR], t_idx)
            rho_raw, rho_dims = get_time_slice_nc(nc_in.variables[RHO_VAR], t_idx)

            u_s, u_s_dims = destagger_to_scalar_grid(u_raw, u_dims)
            v_s, v_s_dims = destagger_to_scalar_grid(v_raw, v_dims)
            w_s, w_s_dims = destagger_to_scalar_grid(w_raw, w_dims)
            p_s, p_s_dims = destagger_to_scalar_grid(p_raw, p_dims)
            rho_s, rho_s_dims = destagger_to_scalar_grid(rho_raw, rho_dims)

            require_dims(u_s_dims, ("zh", "yh", "xh"), U_BASE)
            require_dims(v_s_dims, ("zh", "yh", "xh"), V_BASE)
            require_dims(w_s_dims, ("zh", "yh", "xh"), W_BASE)
            require_dims(p_s_dims, ("zh", "yh", "xh"), PRS_VAR)
            require_dims(rho_s_dims, ("zh", "yh", "xh"), RHO_VAR)

            u_sub = u_s[z_idx]
            v_sub = v_s[z_idx]
            w_sub = w_s[z_idx]
            p_sub = p_s[z_idx]
            rho_sub = rho_s[z_idx]

            if subtract_translation_speed:
                u_sub = u_sub - cx[t_idx]
                v_sub = v_sub - cy[t_idx]

            # 投影到径向/切向
            ur_3d = u_sub * np.cos(theta3d_sub) + v_sub * np.sin(theta3d_sub)
            ut_3d = -u_sub * np.sin(theta3d_sub) + v_sub * np.cos(theta3d_sub)

            ur_avg = azimuthal_average_by_radius(ur_3d, bin_index, valid_mask, nr)
            ut_avg = azimuthal_average_by_radius(ut_3d, bin_index, valid_mask, nr)
            w_avg = azimuthal_average_by_radius(w_sub, bin_index, valid_mask, nr)
            p_avg = azimuthal_average_by_radius(p_sub, bin_index, valid_mask, nr)
            rho_avg = azimuthal_average_by_radius(rho_sub, bin_index, valid_mask, nr)

            out_vars["ur"][t_idx] = np.nan_to_num(ur_avg).astype(np.float32)
            out_vars["ut"][t_idx] = np.nan_to_num(ut_avg).astype(np.float32)
            out_vars["w"][t_idx] = np.nan_to_num(w_avg).astype(np.float32)
            out_vars["prs"][t_idx] = np.nan_to_num(p_avg).astype(np.float32)
            out_vars["rho"][t_idx] = np.nan_to_num(rho_avg).astype(np.float32)

            ut_prime2_avg = eddy_variance_by_radius(
                ut_3d, ut_avg, bin_index, valid_mask, nr
            )
            urut_prime_avg = eddy_covariance_by_radius(
                ur_3d, ut_3d, ur_avg, ut_avg, bin_index, valid_mask, nr,
            )

            # 读取预算项
            br_terms: Dict[str, np.ndarray] = {}
            bt_terms: Dict[str, np.ndarray] = {}
            for suffix, u_name, v_name in budget_pairs:
                au_raw, au_dims = get_time_slice_nc(
                    nc_in.variables[u_name], t_idx
                )
                av_raw, av_dims = get_time_slice_nc(
                    nc_in.variables[v_name], t_idx
                )
                au_s, au_s_dims = destagger_to_scalar_grid(au_raw, au_dims)
                av_s, av_s_dims = destagger_to_scalar_grid(av_raw, av_dims)
                require_dims(au_s_dims, ("zh", "yh", "xh"), u_name)
                require_dims(av_s_dims, ("zh", "yh", "xh"), v_name)
                au_sub = au_s[z_idx]
                av_sub = av_s[z_idx]
                ar_3d = au_sub * np.cos(theta3d_sub) + av_sub * np.sin(theta3d_sub)
                at_3d = -au_sub * np.sin(theta3d_sub) + av_sub * np.cos(theta3d_sub)
                ar_avg = azimuthal_average_by_radius(
                    ar_3d, bin_index, valid_mask, nr
                )
                at_avg = azimuthal_average_by_radius(
                    at_3d, bin_index, valid_mask, nr
                )
                out_vars[f"br_{suffix}"][t_idx] = np.nan_to_num(ar_avg).astype(np.float32)
                out_vars[f"bt_{suffix}"][t_idx] = np.nan_to_num(at_avg).astype(np.float32)
                br_terms[suffix] = np.nan_to_num(ar_avg)
                bt_terms[suffix] = np.nan_to_num(at_avg)

            shape_2d = ur_avg.shape

            # 提取径向预算各项
            hadv = pick_term(br_terms, ("hadv",), shape_2d)
            vadv = pick_term(br_terms, ("vadv",), shape_2d)
            pgrad = pick_term(br_terms, ("pgrad",), shape_2d)
            cor = pick_term(br_terms, ("cor",), shape_2d)
            hidiff = pick_term(br_terms, ("hidiff",), shape_2d)
            hturb = pick_term(br_terms, ("hturb",), shape_2d)
            vidiff = pick_term(br_terms, ("vidiff",), shape_2d)
            vturb = pick_term(br_terms, ("vturb",), shape_2d)
            rdamp = pick_term(br_terms, ("rdamp", "ramp"), shape_2d)

            ur_work, ut_work = ur_avg, ut_avg
            utp2_work, p_work = ut_prime2_avg, p_avg

            if enable_core_stabilization:
                core_rmax = max(core_radius_km * 1000.0, dr_km * 1000.0)
                ur_work, ut_work, utp2_work, p_work = \
                    apply_core_axisymmetric_constraints(
                        ur_avg, ut_avg, ut_prime2_avg, p_avg, r_m, core_rmax
                    )
                core_mask = r_m <= core_rmax
                if np.any(core_mask):
                    edge_idx = int(np.where(core_mask)[0][-1])
                    r_core = float(r_m[edge_idx])
                    if r_core > 0.0:
                        ratio = (r_m[core_mask] / r_core).astype(np.float64)
                        for term_arr in br_terms.values():
                            term_edge = term_arr[:, edge_idx][:, np.newaxis]
                            term_arr[:, core_mask] = term_edge * ratio[np.newaxis, :]

            dur_dr = safe_gradient(ur_work, r_m, axis=1)
            dur_dz = safe_gradient(ur_work, z_m, axis=0)
            dp_dr = safe_gradient(p_work, r_m, axis=1)

            rho_safe = np.where(np.abs(rho_avg) < 1e-12, np.nan, rho_avg)
            r_floor = max(dr_km * 500.0, 1.0) if enable_core_stabilization else 0.0
            r_safe = np.where(r_m <= r_floor, r_floor, r_m)

            curv_mean = (ut_work ** 2) / r_safe[np.newaxis, :]
            curv_eddy = utp2_work / r_safe[np.newaxis, :]
            pgrad_mean = -(1.0 / rho_safe) * dp_dr
            pgrad_eddy = pgrad - pgrad_mean
            coriolis_term = cor

            # 径向收支原始项
            U_mr_raw = ur_work * dur_dr
            U_mv_raw = w_avg * dur_dz
            U_eh_raw = -hadv - U_mr_raw + curv_mean + curv_eddy
            U_ev_raw = -vadv - U_mv_raw
            U_magf_raw = curv_mean + coriolis_term + pgrad_mean
            U_eagf_raw = curv_eddy + pgrad_eddy
            U_dh_raw = hidiff + hturb
            U_dv_raw = vidiff + vturb
            ramp_raw = rdamp

            br_total_raw = np.zeros(shape_2d)
            for arr in br_terms.values():
                br_total_raw += arr

            tendency_model_raw = (
                -U_mr_raw - U_eh_raw - U_mv_raw - U_ev_raw
                + U_magf_raw + U_eagf_raw + U_dh_raw + U_dv_raw + ramp_raw
            )
            residual_raw = br_total_raw - tendency_model_raw

            # 分组残差分配
            adj_terms, pgrad_mean_adj, pgrad_eddy_adj, _, _, _ = \
                grouped_residual_allocation_radial(
                    U_mr_raw=np.nan_to_num(U_mr_raw),
                    U_eh_raw=np.nan_to_num(U_eh_raw),
                    U_mv_raw=np.nan_to_num(U_mv_raw),
                    U_ev_raw=np.nan_to_num(U_ev_raw),
                    curv_mean=np.nan_to_num(curv_mean),
                    curv_eddy=np.nan_to_num(curv_eddy),
                    coriolis=np.nan_to_num(coriolis_term),
                    pgrad_mean=np.nan_to_num(pgrad_mean),
                    pgrad_eddy=np.nan_to_num(pgrad_eddy),
                    hadv=np.nan_to_num(hadv),
                    vadv=np.nan_to_num(vadv),
                    pgrad=np.nan_to_num(pgrad),
                    U_dh_raw=np.nan_to_num(U_dh_raw),
                    U_dv_raw=np.nan_to_num(U_dv_raw),
                    ramp_raw=np.nan_to_num(ramp_raw),
                )

            tendency_model_adjusted = (
                -adj_terms["U_mr"] - adj_terms["U_eh"]
                - adj_terms["U_mv"] - adj_terms["U_ev"]
                + adj_terms["U_magf"] + adj_terms["U_eagf"]
                + adj_terms["U_dh"] + adj_terms["U_dv"] + adj_terms["ramp"]
            )
            residual_after = br_total_raw - tendency_model_adjusted

            # 切向收支
            thadv = pick_term(bt_terms, ("hadv",), shape_2d)
            tvadv = pick_term(bt_terms, ("vadv",), shape_2d)
            tpgrad = pick_term(bt_terms, ("pgrad",), shape_2d)
            tcor = pick_term(bt_terms, ("cor",), shape_2d)
            thidiff = pick_term(bt_terms, ("hidiff",), shape_2d)
            thturb = pick_term(bt_terms, ("hturb",), shape_2d)
            tvidiff = pick_term(bt_terms, ("vidiff",), shape_2d)
            tvturb = pick_term(bt_terms, ("vturb",), shape_2d)
            trdamp = pick_term(bt_terms, ("rdamp", "ramp"), shape_2d)

            dut_dr = safe_gradient(ut_work, r_m, axis=1)
            dut_dz = safe_gradient(ut_work, z_m, axis=0)

            vcurv_mean = (ur_work * ut_work) / r_safe[np.newaxis, :]
            vcurv_eddy = urut_prime_avg / r_safe[np.newaxis, :]

            V_mr_raw = ur_work * dut_dr
            V_mv_raw = w_avg * dut_dz
            V_eh_raw = -thadv - V_mr_raw - vcurv_mean - vcurv_eddy
            V_ev_raw = -tvadv - V_mv_raw
            V_magf_raw = -vcurv_mean + tcor
            V_eagf_raw = -vcurv_eddy + tpgrad
            V_dh_raw = thidiff + thturb
            V_dv_raw = tvidiff + tvturb
            tramp_raw_v = trdamp

            bt_total_raw = np.zeros(shape_2d)
            for arr in bt_terms.values():
                bt_total_raw += arr

            tendency_t_model_raw = (
                -V_mr_raw - V_eh_raw - V_mv_raw - V_ev_raw
                + V_magf_raw + V_eagf_raw + V_dh_raw + V_dv_raw + tramp_raw_v
            )
            residual_t_raw = bt_total_raw - tendency_t_model_raw

            adj_t_terms, _, _ = grouped_residual_allocation_tangential(
                V_mr_raw=np.nan_to_num(V_mr_raw),
                V_eh_raw=np.nan_to_num(V_eh_raw),
                V_mv_raw=np.nan_to_num(V_mv_raw),
                V_ev_raw=np.nan_to_num(V_ev_raw),
                V_magf_raw=np.nan_to_num(V_magf_raw),
                V_eagf_raw=np.nan_to_num(V_eagf_raw),
                hadv=np.nan_to_num(thadv),
                vadv=np.nan_to_num(tvadv),
                vcurv_mean=np.nan_to_num(vcurv_mean),
                vcurv_eddy=np.nan_to_num(vcurv_eddy),
                V_dh_raw=np.nan_to_num(V_dh_raw),
                V_dv_raw=np.nan_to_num(V_dv_raw),
                tramp_raw=np.nan_to_num(tramp_raw_v),
            )
            tendency_t_model_adjusted = (
                -adj_t_terms["V_mr"] - adj_t_terms["V_eh"]
                - adj_t_terms["V_mv"] - adj_t_terms["V_ev"]
                + adj_t_terms["V_magf"] + adj_t_terms["V_eagf"]
                + adj_t_terms["V_dh"] + adj_t_terms["V_dv"]
                + adj_t_terms["tramp"]
            )
            residual_t_after = bt_total_raw - tendency_t_model_adjusted

            # 写入输出
            raw_terms = {
                "U_mr": U_mr_raw, "U_eh": U_eh_raw,
                "U_mv": U_mv_raw, "U_ev": U_ev_raw,
                "U_magf": U_magf_raw, "U_eagf": U_eagf_raw,
                "U_dh": U_dh_raw, "U_dv": U_dv_raw, "ramp": ramp_raw,
            }
            raw_t_terms = {
                "V_mr": V_mr_raw, "V_eh": V_eh_raw,
                "V_mv": V_mv_raw, "V_ev": V_ev_raw,
                "V_magf": V_magf_raw, "V_eagf": V_eagf_raw,
                "V_dh": V_dh_raw, "V_dv": V_dv_raw, "tramp": tramp_raw_v,
            }

            for key, value in raw_terms.items():
                out_vars[f"{key}_raw"][t_idx] = np.nan_to_num(value).astype(np.float32)
            for key, value in adj_terms.items():
                out_vars[key][t_idx] = np.nan_to_num(value).astype(np.float32)
            for key, value in raw_t_terms.items():
                out_vars[f"{key}_raw"][t_idx] = np.nan_to_num(value).astype(np.float32)
            for key, value in adj_t_terms.items():
                out_vars[key][t_idx] = np.nan_to_num(value).astype(np.float32)

            out_vars["curv_mean"][t_idx] = np.nan_to_num(curv_mean).astype(np.float32)
            out_vars["curv_eddy"][t_idx] = np.nan_to_num(curv_eddy).astype(np.float32)
            out_vars["pgrad_mean"][t_idx] = np.nan_to_num(pgrad_mean_adj).astype(np.float32)
            out_vars["pgrad_eddy"][t_idx] = np.nan_to_num(pgrad_eddy_adj).astype(np.float32)
            out_vars["coriolis"][t_idx] = np.nan_to_num(coriolis_term).astype(np.float32)
            out_vars["br_total_raw"][t_idx] = np.nan_to_num(br_total_raw).astype(np.float32)
            out_vars["tendency_model_raw"][t_idx] = np.nan_to_num(tendency_model_raw).astype(np.float32)
            out_vars["tendency_model_adjusted"][t_idx] = np.nan_to_num(tendency_model_adjusted).astype(np.float32)
            out_vars["residual_raw"][t_idx] = np.nan_to_num(residual_raw).astype(np.float32)
            out_vars["residual_after_allocation"][t_idx] = np.nan_to_num(residual_after).astype(np.float32)

            out_vars["vcurv_mean"][t_idx] = np.nan_to_num(vcurv_mean).astype(np.float32)
            out_vars["vcurv_eddy"][t_idx] = np.nan_to_num(vcurv_eddy).astype(np.float32)
            out_vars["coriolis_t"][t_idx] = np.nan_to_num(tcor).astype(np.float32)
            out_vars["pgrad_t"][t_idx] = np.nan_to_num(tpgrad).astype(np.float32)
            out_vars["bt_total_raw"][t_idx] = np.nan_to_num(bt_total_raw).astype(np.float32)
            out_vars["tendency_t_model_raw"][t_idx] = np.nan_to_num(tendency_t_model_raw).astype(np.float32)
            out_vars["tendency_t_model_adjusted"][t_idx] = np.nan_to_num(tendency_t_model_adjusted).astype(np.float32)
            out_vars["residual_t_raw"][t_idx] = np.nan_to_num(residual_t_raw).astype(np.float32)
            out_vars["residual_t_after_allocation"][t_idx] = np.nan_to_num(residual_t_after).astype(np.float32)
            out_vars["time"][t_idx] = time_values[t_idx]

            if verbose and (out_idx % 5 == 0 or out_idx == total_times - 1):
                res_norm = np.nanmean(np.abs(residual_raw))
                res_after_norm = np.nanmean(np.abs(residual_after))
                print(
                    f"[INFO] t={t_idx:04d}/{total_times - 1:04d} "
                    f"time={time_values[t_idx]:.1f}s "
                    f"center=({xc:.2f},{yc:.2f}) "
                    f"|res|={res_norm:.3e}->{res_after_norm:.3e}"
                )

            if t_idx % 5 == 0:
                nc_out.sync()

        nc_out.sync()
        print(f"[INFO] 输出已保存至: {output_file}")
