"""
IO 工具模块 (Input/Output Utilities)

提供：
  - netCDF 文件鲁棒打开（含 Windows 非 ASCII 路径 subst 映射）
  - 变量名自动匹配（候选列表 fallback）
  - 时间索引解析
  - IEEE 二进制读写（Fortran unformatted 格式）
  - NC 属性复制
"""

from __future__ import annotations

import os
import subprocess
import struct
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr


# ==============================================================================
# Windows 非 ASCII 路径处理 (subst)
# ==============================================================================

def is_ascii_path(path_str: str) -> bool:
    """判断路径是否全为 ASCII 字符。"""
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
    """返回 (drive_letter, mapped_root, created_now)。"""
    target_norm = str(dir_path).rstrip("\\/").lower()
    existing = _list_subst_mappings()
    for drv, tgt in existing.items():
        if tgt.lower() == target_norm:
            return drv, f"{drv}:/", False

    letter = _find_free_drive_letter()
    if letter is None:
        return None, None, False

    proc = subprocess.run(
        ["subst", f"{letter}:", str(dir_path)], capture_output=True, text=True
    )
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


# ==============================================================================
# netCDF 鲁棒打开
# ==============================================================================

def open_dataset_robust(
    input_file: str
) -> Tuple[xr.Dataset, Dict[str, str], Optional[str], bool]:
    """
    鲁棒打开 netCDF 文件。

    先尝试直接打开，若路径含非 ASCII 字符则使用 Windows subst 映射。

    Returns
    -------
    ds : xr.Dataset
    open_meta : dict
    subst_letter : Optional[str]
    subst_created_now : bool
    """
    p = Path(input_file).expanduser().resolve()
    path_text = str(p)
    open_meta: Dict[str, str] = {
        "input_path": path_text, "open_path": path_text, "engine": "default"
    }

    try:
        ds = xr.open_dataset(path_text, decode_cf=False)
        return ds, open_meta, None, False
    except Exception as e_direct:
        direct_err = f"{type(e_direct).__name__}: {e_direct}"

    if not is_ascii_path(path_text):
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

    raise RuntimeError(
        f"无法打开 NetCDF 文件: {path_text}\n- 直接打开失败: {direct_err}"
    )


# ==============================================================================
# 变量名匹配
# ==============================================================================

def first_existing_var(ds: xr.Dataset, names: Sequence[str]) -> Optional[str]:
    """返回第一个在数据集中存在的变量名。"""
    for name in names:
        if name in ds.variables:
            return name
    return None


def resolve_core_var_names(
    ds: xr.Dataset,
    u_name: str, v_name: str, w_name: str,
    prs_name: str, rho_name: str, theta_name: str, psfc_name: str,
    u_candidates: Sequence[str],
    v_candidates: Sequence[str],
    w_candidates: Sequence[str],
    prs_candidates: Sequence[str],
    rho_candidates: Sequence[str],
    theta_candidates: Sequence[str],
    psfc_candidates: Sequence[str],
) -> Dict[str, str]:
    """自动匹配核心变量名，打印匹配信息。"""
    mapping: Dict[str, str] = {}

    def pick(preferred: str, candidates: Sequence[str], label: str) -> str:
        if preferred in ds.variables:
            return preferred
        found = first_existing_var(ds, candidates)
        if found is None:
            raise KeyError(
                f"未找到变量 {label}。首选={preferred}, 候选={tuple(candidates)}"
            )
        print(f"[INFO] 变量 {label} 自动匹配为: {found}")
        return found

    mapping["u"] = pick(u_name, u_candidates, "u")
    mapping["v"] = pick(v_name, v_candidates, "v")
    mapping["w"] = pick(w_name, w_candidates, "w")
    mapping["prs"] = pick(prs_name, prs_candidates, "prs")
    mapping["rho"] = pick(rho_name, rho_candidates, "rho")
    mapping["theta"] = pick(theta_name, theta_candidates, "theta")

    if psfc_name in ds.variables:
        mapping["psfc"] = psfc_name
    else:
        psfc_found = first_existing_var(ds, psfc_candidates)
        mapping["psfc"] = psfc_found if psfc_found is not None else ""
        if psfc_found:
            print(f"[INFO] 变量 psfc 自动匹配为: {psfc_found}")
        else:
            print("[WARN] 未找到地面气压变量，将使用网格中心作为台风中心。")

    return mapping


# ==============================================================================
# 时间索引解析
# ==============================================================================

def resolve_time_index(
    time_vals: np.ndarray, time_index: int,
    target_time_seconds: Optional[float],
    target_time_hours: Optional[float],
) -> Tuple[int, float, str]:
    """
    解析目标时间对应的索引。

    Returns (time_index, time_value, method_description).
    """
    nt = int(len(time_vals))
    if nt <= 0:
        return 0, 0.0, "fallback_empty_time"

    if target_time_hours is not None:
        target_sec = float(target_time_hours) * 3600.0
        idx = int(np.nanargmin(np.abs(time_vals - target_sec)))
        return idx, float(time_vals[idx]), f"target_time_hours={target_time_hours}"

    if target_time_seconds is not None:
        target_sec = float(target_time_seconds)
        idx = int(np.nanargmin(np.abs(time_vals - target_sec)))
        return idx, float(time_vals[idx]), f"target_time_seconds={target_time_seconds}"

    if time_index < 0 or time_index >= nt:
        raise IndexError(f"time_index={time_index} 越界，time 维长度={nt}")
    return int(time_index), float(time_vals[time_index]), "time_index"


def resolve_time_indices_for_averaging(
    time_vals: np.ndarray,
    time_avg_start_hours: Optional[float],
    time_avg_end_hours: Optional[float],
    time_index: int,
    target_time_seconds: Optional[float],
    target_time_hours: Optional[float],
) -> Tuple[List[int], str]:
    """返回时间段内所有时间索引（用于时间平均模式）。"""
    if time_avg_start_hours is None or time_avg_end_hours is None:
        idx, _, method = resolve_time_index(
            time_vals, time_index, target_time_seconds, target_time_hours
        )
        return [idx], method

    t_start = float(time_avg_start_hours) * 3600.0
    t_end = float(time_avg_end_hours) * 3600.0
    mask = (time_vals >= t_start - 1e-6) & (time_vals <= t_end + 1e-6)
    indices = [int(i) for i in np.where(mask)[0]]
    if not indices:
        raise ValueError(
            f"时间段 [{time_avg_start_hours}h, {time_avg_end_hours}h] "
            f"内无时间点 (time 范围 [{time_vals.min():.0f}, {time_vals.max():.0f}]s)"
        )
    method = f"time_avg_{time_avg_start_hours}h_to_{time_avg_end_hours}h"
    return indices, method


# ==============================================================================
# IEEE 二进制读写 (Fortran unformatted)
# ==============================================================================

def read_ieee_matrix(filepath: str | Path, shape: Tuple[int, int],
                     dtype: str = ">f8") -> np.ndarray:
    """
    读取 Fortran unformatted IEEE 二进制文件。

    Fortran 记录格式：4 字节头(len) + 数据 + 4 字节尾(len)。
    """
    with open(filepath, "rb") as f:
        rec_head = f.read(4)
        if len(rec_head) < 4:
            raise ValueError(f"{filepath}: 文件太小，无法读取记录头。")
        (expected,) = struct.unpack(">i", rec_head)
        raw = f.read(expected)
        if len(raw) < expected:
            raise ValueError(f"{filepath}: 数据不足，期望 {expected} 字节。")
        rec_tail = f.read(4)
        if len(rec_tail) < 4:
            raise ValueError(f"{filepath}: 文件太小，无法读取记录尾。")

    data = np.frombuffer(raw, dtype=np.dtype(dtype))
    return data.reshape(shape, order="F")  # Fortran 列主序


def write_ieee_matrix(filepath: str | Path, data: np.ndarray,
                      dtype: str = ">f8") -> None:
    """写入 Fortran unformatted IEEE 二进制文件。"""
    raw = np.asarray(data, dtype=np.dtype(dtype)).tobytes(order="F")
    header = struct.pack(">i", len(raw))

    with open(filepath, "wb") as f:
        f.write(header)
        f.write(raw)
        f.write(header)


# ==============================================================================
# NC 属性复制
# ==============================================================================

def copy_nc_attrs(src_var, dst_var) -> None:
    """将源 NC 变量的所有属性复制到目标变量。"""
    for attr_name in src_var.ncattrs():
        try:
            dst_var.setncattr(attr_name, src_var.getncattr(attr_name))
        except Exception:
            pass
