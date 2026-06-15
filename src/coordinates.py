"""
坐标变换与网格工具 (Coordinate Transforms & Grid Utilities)

CM1 使用 Arakawa-C 交错网格，本模块提供去交错(destagger)、维度检查、
时间切片提取等通用坐标处理函数。

这些函数原本在 budget_full、budget_full_grouped_residual 和 3 个 SE pipeline
文件中各自重复定义，现已统一至此。
"""

from typing import List, Sequence, Tuple
import numpy as np
import xarray as xr


def destagger_axis(data: np.ndarray, axis: int) -> np.ndarray:
    """
    对交错网格沿指定轴做简单算术平均去交错。

    CM1 的 u 定义在 xf 面、v 定义在 yf 面、w 定义在 zf 面。
    去交错后数据维度减少 1。
    """
    sl0 = [slice(None)] * data.ndim
    sl1 = [slice(None)] * data.ndim
    sl0[axis] = slice(0, -1)
    sl1[axis] = slice(1, None)
    return 0.5 * (data[tuple(sl0)] + data[tuple(sl1)])


def destagger_to_scalar_grid(
    data: np.ndarray, dims_wo_time: Sequence[str]
) -> Tuple[np.ndarray, List[str]]:
    """
    将交错网格数据去交错到标量网格 (xh, yh, zh)。

    Parameters
    ----------
    data : np.ndarray
        输入数据（不含 time 维）。
    dims_wo_time : Sequence[str]
        维度名列表（不含 time），如 ['zf', 'yf', 'xf']。

    Returns
    -------
    out : np.ndarray
        去交错后的数据。
    out_dims : List[str]
        去交错后的维度名列表。
    """
    out = np.asarray(data, dtype=np.float64)
    out_dims = list(dims_wo_time)
    stag_to_scalar = {"xf": "xh", "yf": "yh", "zf": "zh"}
    for stag_dim in ("xf", "yf", "zf"):
        if stag_dim in out_dims:
            axis = out_dims.index(stag_dim)
            out = destagger_axis(out, axis)
            out_dims[axis] = stag_to_scalar[stag_dim]
    return out, out_dims


def require_dims(
    dims: Sequence[str], expected: Sequence[str], var_name: str
) -> None:
    """检查变量的维度是否与预期一致，不一致则抛出 ValueError。"""
    if tuple(dims) != tuple(expected):
        raise ValueError(
            f"变量 {var_name} 维度为 {tuple(dims)}，期望 {tuple(expected)}"
        )


def get_time_slice(
    var_in: xr.DataArray, t_idx: int
) -> Tuple[np.ndarray, List[str]]:
    """
    从 xarray DataArray 中提取单个时间切片。

    Returns
    -------
    data : np.ndarray
        不含 time 维的数据。
    dims : List[str]
        不含 'time' 的维度名列表。
    """
    dims = list(var_in.dims)
    indexers = {d: (t_idx if d == "time" else slice(None)) for d in dims}
    data = np.asarray(var_in.isel(indexers), dtype=np.float64)
    return data, [d for d in dims if d != "time"]


def ensure_2d_xy(arr: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """
    确保数组为 (ny, nx) 的二维数组，即索引顺序 [iy, ix]。

    自动处理 (nx, ny) ↔ (ny, nx) 的转置。
    """
    if arr.shape == (nx, ny):
        return arr.T
    if arr.shape == (ny, nx):
        return arr
    if arr.ndim >= 2 and arr.shape[-2] == ny and arr.shape[-1] == nx:
        return arr.reshape(arr.shape[-2], arr.shape[-1])
    if arr.ndim >= 2 and arr.shape[-2] == nx and arr.shape[-1] == ny:
        return arr.reshape(arr.shape[-2], arr.shape[-1]).T
    raise ValueError(
        f"无法识别切片形状 {arr.shape} 与网格尺寸 (nx,ny)=({nx},{ny}) 的对应关系。"
    )


def nearest_index_1d(arr: np.ndarray, value: float) -> int:
    """在一维数组中寻找最接近 value 的索引。"""
    return int(np.argmin(np.abs(np.asarray(arr, dtype=float) - float(value))))


def parse_csv_names(text: str) -> Tuple[str, ...]:
    """将逗号分隔字符串解析为 tuple。"""
    names = [x.strip() for x in text.split(",") if x.strip()]
    return tuple(names)


def safe_gradient(
    field_2d: np.ndarray, coords_1d: np.ndarray, axis: int
) -> np.ndarray:
    """
    安全计算二维场沿指定轴的梯度。

    处理单列/单行情况，避免 np.gradient 崩溃。
    """
    if field_2d.shape[axis] < 2:
        return np.zeros_like(field_2d, dtype=np.float64)
    return np.gradient(field_2d, coords_1d, axis=axis, edge_order=1)
