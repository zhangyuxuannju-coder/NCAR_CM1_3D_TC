"""
台风中心定位模块 (Typhoon Center Finder)

提供三种台风中心定位方法：
  - "min"            - 平滑后的极小值 (默认, 适合气压场)
  - "centroid"       - 权重质心法 (避免对单点网格极值的敏感)
  - "streamfunction" - 流函数极值法 (利用 u,v 风场求解泊松方程获得流函数)

对外接口:
  - find_smoothed_min_point()  : 在海表气压场(psfc)上定位中心
  - find_smoothed_min_prs()    : 在指定高度层气压场(prs)上定位中心
"""

from typing import Dict, Optional, Tuple
import numpy as np
from netCDF4 import Dataset

# ==============================================================================
# 模块级配置
# ==============================================================================
DEFAULT_CENTER_METHOD = "min"

# 跨时次追踪状态（用于保持中心追踪的连续性）
_TRACK_STATE: Dict[Tuple[str, str, str, str, str], Dict[str, float]] = {}


# ==============================================================================
# 内部辅助函数
# ==============================================================================

def _make_track_key(
    nc_file: str,
    var_name: str,
    x_dim: str,
    y_dim: str,
    center_method: str,
) -> Tuple[str, str, str, str, str]:
    return (str(nc_file), str(var_name), str(x_dim), str(y_dim), str(center_method).lower())


def _initial_or_previous_center(
    track_key: Tuple[str, str, str, str, str],
    time_idx: int,
    xh: np.ndarray,
    yh: np.ndarray,
    search_radius: float,
) -> Tuple[float, float, float]:
    """返回初始猜测中心：优先使用上一时刻的中心，否则用全域中心。"""
    state = _TRACK_STATE.get(track_key)
    if state is not None:
        prev_time_idx = int(state.get("time_index", -10**9))
        if time_idx >= prev_time_idx:
            return float(state["x"]), float(state["y"]), float(search_radius)

    center_x0 = float(np.nanmean(np.asarray(xh, dtype=float)))
    center_y0 = float(np.nanmean(np.asarray(yh, dtype=float)))
    domain_radius = float(max(np.ptp(np.asarray(xh, dtype=float)),
                              np.ptp(np.asarray(yh, dtype=float)))) * 2.0
    if not np.isfinite(domain_radius) or domain_radius <= 0:
        domain_radius = float(search_radius)
    return center_x0, center_y0, domain_radius


def _find_time_index(time_arr: np.ndarray, time_key) -> int:
    """time_key: int（且在范围内）视作索引，否则按值匹配最近时间。"""
    if time_key is None:
        return 0
    if isinstance(time_key, int):
        if 0 <= time_key < len(time_arr):
            return int(time_key)
    return int(np.argmin(np.abs(time_arr - float(time_key))))


def _coord_as_2d(x1d: np.ndarray, y1d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(np.asarray(x1d, dtype=float), np.asarray(y1d, dtype=float))
    return xx, yy


def _nearest_index_1d(arr: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(np.asarray(arr, dtype=float) - float(value))))


def _estimate_grid_spacing(x1d: np.ndarray, y1d: np.ndarray) -> float:
    dx = np.nanmedian(np.abs(np.diff(np.asarray(x1d, dtype=float)))) if len(x1d) > 1 else np.nan
    dy = np.nanmedian(np.abs(np.diff(np.asarray(y1d, dtype=float)))) if len(y1d) > 1 else np.nan
    vals = [v for v in (dx, dy) if np.isfinite(v) and v > 0]
    if not vals:
        return 1.0
    return float(np.mean(vals))


def _local_minimum_seed(
    field_2d: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    center_x: float,
    center_y: float,
    search_radius: float,
) -> Tuple[float, float, float]:
    """在搜索半径内找到场的最小值位置作为种子点。"""
    dist = np.hypot(xx - center_x, yy - center_y)
    mask = np.isfinite(field_2d) & (dist <= search_radius)
    if not np.any(mask):
        iy = _nearest_index_1d(yy[:, 0], center_y)
        ix = _nearest_index_1d(xx[0, :], center_x)
        return float(xx[iy, ix]), float(yy[iy, ix]), float(field_2d[iy, ix])

    idx_flat = int(np.argmin(np.where(mask, field_2d, np.inf)))
    iy, ix = np.unravel_index(idx_flat, field_2d.shape)
    return float(xx[iy, ix]), float(yy[iy, ix]), float(field_2d[iy, ix])


def _pressure_centroid_iter(
    field_2d: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    seed_x: float,
    seed_y: float,
    radius: float,
    max_iter: int,
    tol_abs: float,
) -> Tuple[float, float, int]:
    """迭代计算压力质心，提高定位精度（避免单点网格极值敏感）。"""
    curr_x, curr_y = float(seed_x), float(seed_y)

    for it in range(max_iter):
        dist = np.hypot(xx - curr_x, yy - curr_y)
        mask = np.isfinite(field_2d) & (dist <= radius)
        if not np.any(mask):
            return curr_x, curr_y, it + 1

        p_local = field_2d[mask]
        p_ref = float(np.nanmax(p_local))
        w = p_ref - p_local

        valid = np.isfinite(w) & (w > 0)
        if np.count_nonzero(valid) < 3:
            return curr_x, curr_y, it + 1

        x_local = xx[mask][valid]
        y_local = yy[mask][valid]
        w_local = w[valid]

        w_sum = float(np.sum(w_local))
        if not np.isfinite(w_sum) or w_sum <= 0:
            return curr_x, curr_y, it + 1

        new_x = float(np.sum(w_local * x_local) / w_sum)
        new_y = float(np.sum(w_local * y_local) / w_sum)

        shift = float(np.hypot(new_x - curr_x, new_y - curr_y))
        curr_x, curr_y = new_x, new_y
        if shift < tol_abs:
            return curr_x, curr_y, it + 1

    return curr_x, curr_y, max_iter


def _to_yx_2d(sl: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """确保切片为 (ny, nx) 的二维数组。"""
    arr = np.asarray(sl, dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim < 2:
        raise ValueError(f"切片维度不足，无法构造二维场: shape={arr.shape}")

    if arr.ndim > 2:
        arr = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))[0]

    if arr.shape == (nx, ny):
        return arr.T
    if arr.shape == (ny, nx):
        return arr
    if arr.shape[-2] == ny and arr.shape[-1] == nx:
        return arr
    if arr.shape[-2] == nx and arr.shape[-1] == ny:
        return arr.T
    raise ValueError(
        f"无法识别变量切片形状 {arr.shape} 与 xh/yh 长度 ({nx},{ny}) 的对应关系。"
    )


def _compute_streamfunction_track_field(
    nc: Dataset, time_idx: int, z_idx: Optional[int],
    nx: int, ny: int, dxy: float
) -> np.ndarray:
    """
    流函数极值法：从 CM1 提取 u,v 风场，用差分求涡度，
    通过离散正弦变换 (DST) 解泊松方程得流函数。
    返回 -abs(Psi)，使流函数极值为二维坑底最小值，可稳定追踪。
    """
    if "u" not in nc.variables or "v" not in nc.variables:
        raise KeyError("文件中未找到变量 'u' 或 'v'，无法计算流函数。")

    var_u = nc.variables["u"]
    var_v = nc.variables["v"]

    # 提取 u
    if z_idx is None:
        u_raw = var_u[time_idx, 0, :, :] if var_u.ndim == 4 else var_u[time_idx, :, :]
    else:
        u_raw = var_u[time_idx, z_idx, :, :]

    # 提取 v
    if z_idx is None:
        v_raw = var_v[time_idx, 0, :, :] if var_v.ndim == 4 else var_v[time_idx, :, :]
    else:
        v_raw = var_v[time_idx, z_idx, :, :]

    u_raw, v_raw = np.asarray(u_raw, dtype=float), np.asarray(v_raw, dtype=float)
    u_raw, v_raw = np.squeeze(u_raw), np.squeeze(v_raw)

    # CM1 staggered 网格内插回 xh, yh
    if u_raw.shape == (ny, nx + 1):
        u_interp = 0.5 * (u_raw[:, :-1] + u_raw[:, 1:])
    else:
        u_interp = _to_yx_2d(u_raw, nx, ny)

    if v_raw.shape == (ny + 1, nx):
        v_interp = 0.5 * (v_raw[:-1, :] + v_raw[1:, :])
    else:
        v_interp = _to_yx_2d(v_raw, nx, ny)

    # 手动中心差分计算涡度（避免边缘异常）
    dv_dx = np.zeros_like(v_interp)
    dv_dx[:, 1:-1] = (v_interp[:, 2:] - v_interp[:, :-2]) / (2.0 * dxy)
    dv_dx[:, 0] = (v_interp[:, 1] - v_interp[:, 0]) / dxy
    dv_dx[:, -1] = (v_interp[:, -1] - v_interp[:, -2]) / dxy

    du_dy = np.zeros_like(u_interp)
    du_dy[1:-1, :] = (u_interp[2:, :] - u_interp[:-2, :]) / (2.0 * dxy)
    du_dy[0, :] = (u_interp[1, :] - u_interp[0, :]) / dxy
    du_dy[-1, :] = (u_interp[-1, :] - u_interp[-2, :]) / dxy

    zeta = dv_dx - du_dy
    zeta = np.nan_to_num(zeta, nan=0.0)

    from scipy.fft import dstn, idstn
    zeta_hat = dstn(zeta, type=1)

    kx = np.arange(1, nx + 1)
    ky = np.arange(1, ny + 1)
    KX, KY = np.meshgrid(kx, ky)

    lam_x = -(2.0 / dxy * np.sin(np.pi * KX / (2.0 * (nx + 1)))) ** 2
    lam_y = -(2.0 / dxy * np.sin(np.pi * KY / (2.0 * (ny + 1)))) ** 2
    lam = lam_x + lam_y

    psi_hat = zeta_hat / lam
    psi = idstn(psi_hat, type=1)

    return -np.abs(psi)


# ==============================================================================
# 公开接口
# ==============================================================================

def find_smoothed_min_point(
    nc_file: str,
    time_key,
    var_name: str = "psfc",
    x_dim: str = "xh",
    y_dim: str = "yh",
    window: int = 21,
    verbose: bool = False,
    center_method: str = DEFAULT_CENTER_METHOD
) -> Dict[str, object]:
    """
    在海表气压场 (psfc) 上寻找平滑后最小气压中心。

    Parameters
    ----------
    nc_file : str
        netCDF 文件路径。
    time_key : int or float
        时间索引（int）或时间值（float）。
    var_name : str
        变量名，默认 'psfc'。
    x_dim, y_dim : str
        水平坐标维名，默认 'xh', 'yh'。
    window : int
        平滑窗口尺寸（默认 21，即半径 10）。
    verbose : bool
        是否打印中间信息。
    center_method : str
        "min" 使用平滑后极小值；
        "centroid" 使用极小值域加权质心；
        "streamfunction" 使用流函数极值法。

    Returns
    -------
    dict with keys:
        ix, iy          : 全局格点索引
        x, y            : 物理坐标
        smoothed_value  : 最小平滑值
        time_index      : 使用的时间索引
        time_value      : 使用的时间实际值
        central_window_indices : (ix0, ix1, iy0, iy1)
    """
    half = window // 2

    with Dataset(nc_file, "r") as nc:
        if var_name not in nc.variables:
            raise KeyError(
                f"{var_name} 不在 nc 文件中。可用变量: {list(nc.variables.keys())}"
            )
        var = nc.variables[var_name]

        if 'time' not in nc.variables:
            raise KeyError("netCDF 文件中未找到 'time' 变量。")

        time_arr = nc.variables['time'][:]
        xh = nc.variables[x_dim][:]
        yh = nc.variables[y_dim][:]

        nx = len(xh)
        ny = len(yh)

        time_idx = _find_time_index(time_arr, time_key)
        time_value = time_arr[time_idx]
        if verbose:
            print(f"使用 time index = {time_idx}, time value = {time_value}")

        sl = _to_yx_2d(np.asarray(var[time_idx, ...], dtype=float), nx, ny)
        if np.all(np.isnan(sl)):
            raise ValueError("所选 time 切片全为 NaN，无法处理。")

        xx, yy = _coord_as_2d(xh, yh)
        dxy = _estimate_grid_spacing(xh, yh)
        centroid_radius = max(float(half) * dxy, dxy)
        search_radius = max(1.2 * centroid_radius, dxy)
        tol_abs = 0.01 * dxy

        track_key = _make_track_key(nc_file, var_name, x_dim, y_dim, center_method)
        seed_center_x, seed_center_y, seed_search_radius = _initial_or_previous_center(
            track_key=track_key, time_idx=time_idx,
            xh=np.asarray(xh, dtype=float), yh=np.asarray(yh, dtype=float),
            search_radius=search_radius,
        )

        if center_method.lower() == "streamfunction":
            sl_track = _compute_streamfunction_track_field(
                nc, time_idx, None, nx, ny, dxy
            )
        else:
            sl_track = sl

        seed_x, seed_y, seed_p = _local_minimum_seed(
            sl_track, xx, yy, seed_center_x, seed_center_y,
            search_radius=seed_search_radius
        )

        if center_method.lower() in ("centroid", "streamfunction"):
            center_x, center_y, n_iter = _pressure_centroid_iter(
                sl_track, xx, yy, seed_x, seed_y,
                radius=centroid_radius, max_iter=10, tol_abs=tol_abs
            )
        else:
            center_x, center_y = seed_x, seed_y
            n_iter = 1

        _TRACK_STATE[track_key] = {
            "x": float(center_x), "y": float(center_y),
            "time_index": float(time_idx)
        }

        chosen_ix = _nearest_index_1d(xh, center_x)
        chosen_iy = _nearest_index_1d(yh, center_y)
        # 平滑窗口范围
        rx0 = max(0, chosen_ix - half)
        rx1 = min(nx - 1, chosen_ix + half)
        ry0 = max(0, chosen_iy - half)
        ry1 = min(ny - 1, chosen_iy + half)

        chosen_x = float(xh[chosen_ix])
        chosen_y = float(yh[chosen_iy])
        nb_x0 = max(0, chosen_ix - half)
        nb_x1 = min(nx - 1, chosen_ix + half)
        nb_y0 = max(0, chosen_iy - half)
        nb_y1 = min(ny - 1, chosen_iy + half)
        chosen_value = float(np.nanmean(sl[nb_y0:nb_y1 + 1, nb_x0:nb_x1 + 1]))

        result = {
            "ix": int(chosen_ix),
            "iy": int(chosen_iy),
            "x": chosen_x,
            "y": chosen_y,
            "smoothed_value": chosen_value,
            "time_index": int(time_idx),
            "time_value": float(time_value),
            "central_window_indices": (int(rx0), int(rx1), int(ry0), int(ry1))
        }

        if verbose:
            print(
                f"seed=({seed_x:.3f},{seed_y:.3f}, p={seed_p:.3f}), "
                f"center=({center_x:.3f},{center_y:.3f}), iter={n_iter}"
            )
            print("结果:", result)

        return result


def find_smoothed_min_prs(
    nc_file: str,
    time_key,
    target_zh: float,
    var_name: str = "prs",
    x_dim: str = "xh",
    y_dim: str = "yh",
    z_dim: str = "zh",
    window: int = 21,
    verbose: bool = False,
    center_method: str = DEFAULT_CENTER_METHOD
) -> Dict[str, object]:
    """
    在指定时间和高度层上，寻找平滑后最小气压中心。

    Parameters
    ----------
    nc_file : str
        netCDF 文件路径。
    time_key : int or float
        时间索引或时间值。
    target_zh : float
        目标高度（与 z_dim 单位一致）。
    var_name : str
        变量名，默认 'prs'（三维气压场）。
    x_dim, y_dim, z_dim : str
        坐标维度名。
    window : int
        平滑窗口尺寸（默认 21）。
    verbose : bool
        是否打印过程信息。
    center_method : str
        "min" 或 "centroid" 或 "streamfunction"。

    Returns
    -------
    dict : 同 find_smoothed_min_point 的返回结构。
    """
    half = window // 2

    with Dataset(nc_file, "r") as nc:
        if var_name not in nc.variables:
            raise KeyError(
                f"{var_name} 不在 nc 文件中。可用变量: {list(nc.variables.keys())}"
            )
        var = nc.variables[var_name]

        if 'time' not in nc.variables:
            raise KeyError("netCDF 文件中未找到 'time' 变量。")
        if z_dim not in nc.variables:
            raise KeyError(f"netCDF 文件中未找到 '{z_dim}' 变量。")

        time_arr = nc.variables['time'][:]
        xh = nc.variables[x_dim][:]
        yh = nc.variables[y_dim][:]
        zh_arr = nc.variables[z_dim][:]

        nx = len(xh)
        ny = len(yh)
        zh_idx = _nearest_index_1d(zh_arr, target_zh)
        zh_value = float(zh_arr[zh_idx])

        time_idx = _find_time_index(time_arr, time_key)
        time_value = time_arr[time_idx]

        if verbose:
            print(f"使用 time index={time_idx}, zh index={zh_idx} (value={zh_value})")

        # 读取三维场并提取指定高度层
        sl_3d = np.asarray(var[time_idx, ...], dtype=float)
        if sl_3d.ndim == 4:
            sl = _to_yx_2d(sl_3d[zh_idx, ...], nx, ny)
        elif sl_3d.ndim == 3:
            sl = _to_yx_2d(sl_3d[zh_idx, ...], nx, ny)
        else:
            raise ValueError(f"变量 {var_name} 维度异常: {sl_3d.shape}")

        if np.all(np.isnan(sl)):
            raise ValueError("所选切片全为 NaN，无法处理。")

        xx, yy = _coord_as_2d(xh, yh)
        dxy = _estimate_grid_spacing(xh, yh)
        centroid_radius = max(float(half) * dxy, dxy)
        search_radius = max(1.2 * centroid_radius, dxy)
        tol_abs = 0.01 * dxy

        track_key = _make_track_key(nc_file, var_name, x_dim, y_dim, center_method)
        seed_center_x, seed_center_y, seed_search_radius = _initial_or_previous_center(
            track_key=track_key, time_idx=time_idx,
            xh=np.asarray(xh, dtype=float), yh=np.asarray(yh, dtype=float),
            search_radius=search_radius,
        )

        if center_method.lower() == "streamfunction":
            sl_track = _compute_streamfunction_track_field(
                nc, time_idx, zh_idx, nx, ny, dxy
            )
        else:
            sl_track = sl

        seed_x, seed_y, seed_p = _local_minimum_seed(
            sl_track, xx, yy, seed_center_x, seed_center_y,
            search_radius=seed_search_radius
        )

        if center_method.lower() in ("centroid", "streamfunction"):
            center_x, center_y, n_iter = _pressure_centroid_iter(
                sl_track, xx, yy, seed_x, seed_y,
                radius=centroid_radius, max_iter=10, tol_abs=tol_abs
            )
        else:
            center_x, center_y = seed_x, seed_y
            n_iter = 1

        _TRACK_STATE[track_key] = {
            "x": float(center_x), "y": float(center_y),
            "time_index": float(time_idx)
        }

        chosen_ix = _nearest_index_1d(xh, center_x)
        chosen_iy = _nearest_index_1d(yh, center_y)
        chosen_x = float(xh[chosen_ix])
        chosen_y = float(yh[chosen_iy])

        rx0 = max(0, chosen_ix - half)
        rx1 = min(nx - 1, chosen_ix + half)
        ry0 = max(0, chosen_iy - half)
        ry1 = min(ny - 1, chosen_iy + half)

        nb_x0 = max(0, chosen_ix - half)
        nb_x1 = min(nx - 1, chosen_ix + half)
        nb_y0 = max(0, chosen_iy - half)
        nb_y1 = min(ny - 1, chosen_iy + half)
        chosen_value = float(np.nanmean(sl[nb_y0:nb_y1 + 1, nb_x0:nb_x1 + 1]))

        result = {
            "ix": int(chosen_ix),
            "iy": int(chosen_iy),
            "x": chosen_x,
            "y": chosen_y,
            "smoothed_value": chosen_value,
            "time_index": int(time_idx),
            "time_value": float(time_value),
            "zh_index": int(zh_idx),
            "zh_value": zh_value,
            "central_window_indices": (int(rx0), int(rx1), int(ry0), int(ry1))
        }

        if verbose:
            print(
                f"seed=({seed_x:.3f},{seed_y:.3f}, p={seed_p:.3f}), "
                f"center=({center_x:.3f},{center_y:.3f}), iter={n_iter}"
            )
            print("结果:", result)

        return result
