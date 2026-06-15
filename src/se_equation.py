"""
Sawyer-Eliassen (SE) 方程求解模块

CM1 台风次级环流诊断的核心算法。

主要功能:
  - 方位角平均 (azimuthal_average_from_3d)
  - 热成风平衡位温反演 (invert_theta_from_thermal_wind)
  - SE 系数矩阵构建 (build_se_coefficients)
  - 椭圆性正则化 (regularize_inertial_stability_for_ellipticity)
  - SOR 迭代求解 (solve_se_sor)
  - 稀疏直接求解 (solve_se_sparse)
  - 流函数→风速转换 (psi_to_uw)
  - 蒸发冷却强迫项构造 (build_evap_cooling_Q)

对外接口: run_se_pipeline(cfg: PipelineConfig) -> None
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from .config import PipelineConfig, SourceMaskConfig
from .coordinates import (
    destagger_axis,
    destagger_to_scalar_grid,
    get_time_slice,
    nearest_index_1d,
    parse_csv_names,
    require_dims,
    safe_gradient as _safe_gradient,
)
from .io import (
    first_existing_var,
    open_dataset_robust,
    resolve_time_index,
    resolve_time_indices_for_averaging,
)

# 重力加速度
G = 9.806


# ==============================================================================
# 一维平滑工具
# ==============================================================================

def moving_average_1d(arr: np.ndarray, window: int) -> np.ndarray:
    """一维滑动平均。"""
    w = int(window)
    if w <= 1:
        return np.asarray(arr, dtype=np.float64)
    if w % 2 == 0:
        w += 1
    pad = w // 2
    arr_pad = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=np.float64) / w
    return np.convolve(arr_pad, kernel, mode="valid")


def smooth_2d_3point(field: np.ndarray) -> np.ndarray:
    """NCL式 3点滑动平均，z方向和r方向各做一遍。"""
    out = np.array(field, copy=True, dtype=np.float64)
    nz, nr = out.shape
    if nz >= 3:
        for i in range(1, nz - 1):
            out[i, :] = (field[i - 1, :] + field[i, :] + field[i + 1, :]) / 3.0
    if nr >= 3:
        tmp = np.array(out, copy=True)
        for j in range(1, nr - 1):
            out[:, j] = (tmp[:, j - 1] + tmp[:, j] + tmp[:, j + 1]) / 3.0
    return out


def repair_nan_2d(field: np.ndarray) -> np.ndarray:
    """用邻居填充 NaN。"""
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
        print(f"[INFO] repair_nan_2d: 修复了 {n_bad} 个非法值")
    return out


# ==============================================================================
# 方位角平均
# ==============================================================================

def compute_radial_bin_index(
    r2d: np.ndarray, r_bins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """计算每个网格点的径向 bin 索引。"""
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


def expand_azimuthal_mean_to_xy(
    mean_zr: np.ndarray,
    bin_index_1d: np.ndarray,
    valid_mask_1d: np.ndarray,
    ny: int,
    nx: int,
) -> np.ndarray:
    """将方位角平均场展开回 (z, y, x) 笛卡尔网格。"""
    nz, _ = mean_zr.shape
    out = np.full((nz, ny, nx), np.nan, dtype=np.float64)
    for k in range(nz):
        flat = out[k].ravel()
        flat[valid_mask_1d] = mean_zr[k, bin_index_1d[valid_mask_1d]]
    return out


# ==============================================================================
# 平衡位温反演 (热成风)
# ==============================================================================

def invert_theta_from_thermal_wind(
    ut_2d: np.ndarray,
    theta_2d_model: np.ndarray,
    r_m: np.ndarray,
    z_m: np.ndarray,
    f: float,
    theta_floor: float,
    smooth_window: int,
) -> np.ndarray:
    """
    以外边界位温作为积分边界，使用梯度风热成风关系反算平衡位温。
    """
    vt = np.asarray(ut_2d, dtype=np.float64)
    theta_model = np.asarray(theta_2d_model, dtype=np.float64)

    r_safe = np.maximum(r_m, 0.5 * np.min(np.diff(r_m)) if len(r_m) > 1 else 1.0)
    gradwind = vt**2 / r_safe[None, :] + f * vt
    dgradwind_dz = _safe_gradient(gradwind, z_m, axis=0)

    theta_outer = theta_model[:, -1]
    theta_outer = moving_average_1d(theta_outer, smooth_window)
    theta_outer = np.where(
        np.isfinite(theta_outer),
        theta_outer,
        np.nanmedian(theta_model, axis=1)
    )
    theta_outer = np.maximum(theta_outer, theta_floor)

    dtheta_dr = -(theta_outer[:, None] / G) * dgradwind_dz

    nz, nr = vt.shape
    theta_bal = np.full((nz, nr), np.nan, dtype=np.float64)
    theta_bal[:, -1] = theta_outer

    if nr > 1:
        dr = np.diff(r_m)
        for j in range(nr - 2, -1, -1):
            theta_bal[:, j] = theta_bal[:, j + 1] \
                - 0.5 * (dtheta_dr[:, j + 1] + dtheta_dr[:, j]) * dr[j]

    theta_bal = np.where(np.isfinite(theta_bal), theta_bal, theta_model)
    theta_bal = np.maximum(theta_bal, theta_floor)
    return theta_bal


# ==============================================================================
# SE 诊断场构建
# ==============================================================================

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
    """从方位角平均场构建 SE 系数所需的诊断场。"""
    vt = np.asarray(ut_2d, dtype=np.float64)
    theta = np.asarray(theta_bal_2d, dtype=np.float64)
    rho = np.asarray(rho_2d, dtype=np.float64)

    r_safe = np.maximum(r_m, 0.5 * np.min(np.diff(r_m)) if len(r_m) > 1 else 1.0)

    chi = 1.0 / np.maximum(theta, 1.0)
    C = _safe_gradient(vt, z_m, axis=0)
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


# ==============================================================================
# SE 系数矩阵构建
# ==============================================================================

def build_se_coefficients(
    fields: Dict[str, np.ndarray],
    r_m: np.ndarray,
    z_m: np.ndarray,
    baroclinic_scale: float = 0.4,
) -> Dict[str, np.ndarray]:
    """
    构建 SE 方程的 A/B/C/D/E/F 系数矩阵。

    方程形式: A ∂²ψ/∂r² + B ∂²ψ/∂r∂z + C ∂²ψ/∂z²
              + D ∂ψ/∂r + E ∂ψ/∂z = F
    """
    chi = fields["chi"]
    C_field = fields["C"]
    xi = fields["xi"]
    rho = np.maximum(fields["rho"], 1.0e-8)
    Q = fields["Q"]
    Fnu = fields["Fnu"]

    # NCL式 3点平滑
    chi = smooth_2d_3point(chi)
    rho = smooth_2d_3point(rho)

    chi_r = _safe_gradient(chi, r_m, axis=1)
    chi_z = _safe_gradient(chi, z_m, axis=0)

    K1 = -G * chi_z

    # K2: 使用梯度风项 ct = (vt/r + f)*vt
    ct_field = fields.get("ct", None)
    if ct_field is not None:
        K2 = -_safe_gradient(chi * ct_field, z_m, axis=0)
        K2 *= baroclinic_scale
    else:
        K2 = -_safe_gradient(chi * C_field, z_m, axis=0)

    # K3: I2 = chi * xi * (zeta + f) + ct * chi_r
    if ct_field is not None:
        K3 = chi * fields["inertial_stability"] + ct_field * chi_r
    else:
        K3 = chi * fields["inertial_stability"] + C_field * chi_r

    r_safe = np.maximum(r_m, 0.5 * np.min(np.diff(r_m)) if len(r_m) > 1 else 1.0)
    M = 1.0 / (rho * r_safe[None, :])

    A = K1 * M
    B_coef = 2.0 * K2 * M
    C_coef = K3 * M

    D = _safe_gradient(A, r_m, axis=1) + _safe_gradient(K2 * M, z_m, axis=0)
    E = _safe_gradient(K2 * M, r_m, axis=1) + _safe_gradient(C_coef, z_m, axis=0)

    thermal_flux = (chi**2) * Q
    momentum_flux = chi * xi * Fnu

    forcing_thermal = (
        G * _safe_gradient(thermal_flux, r_m, axis=1)
        + _safe_gradient(C_field * thermal_flux, z_m, axis=0)
    )
    forcing_momentum = -_safe_gradient(momentum_flux, z_m, axis=0)
    F_term = forcing_thermal + forcing_momentum

    discriminant = 4.0 * A * C_coef - B_coef**2

    return {
        "A": A, "B": B_coef, "C": C_coef, "D": D, "E": E,
        "F": F_term,
        "forcing_total": F_term,
        "forcing_thermal": forcing_thermal,
        "forcing_momentum": forcing_momentum,
        "discriminant": discriminant,
        "K1": K1, "K2": K2, "K3": K3,
    }


# ==============================================================================
# 椭圆性正则化
# ==============================================================================

def regularize_inertial_stability_for_ellipticity(
    fields: Dict[str, np.ndarray],
    r_m: np.ndarray,
    z_m: np.ndarray,
    margin: float,
    eps_ratio: float,
    max_iter: int,
    baroclinic_scale: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    调整 K1/K2/K3 系数确保方程全域为椭圆型。

    上层海绵层: 15km以上指数增强 K1/K2/K3, 维持椭圆性。
    """
    chi = smooth_2d_3point(fields["chi"])
    C_field = fields["C"]
    xi = fields["xi"]
    zeta = fields["zeta"]
    f_cor = 5e-5

    zeta_reg = np.maximum(zeta, 0.01 * f_cor)

    chi_r = _safe_gradient(chi, r_m, axis=1)
    chi_z = _safe_gradient(chi, z_m, axis=0)

    K1 = -G * chi_z

    # 上层海绵
    z_sponge_start = 15000.0
    z_sponge_scale = 500.0
    sponge = np.where(
        z_m > z_sponge_start,
        np.exp((z_m - z_sponge_start) / z_sponge_scale),
        1.0
    )
    sponge_2d = sponge[:, None]

    K1_reg = np.maximum(K1, 1.0e-10) * sponge_2d

    # K3
    ct_field = fields.get("ct", None)
    if ct_field is not None:
        I2 = chi * xi * (zeta_reg + f_cor) + ct_field * chi_r
    else:
        I2 = chi * xi * (zeta_reg + f_cor) + C_field * chi_r

    I2_max = float(np.nanmax(np.abs(I2)))
    small_pos = I2_max * 1e-3
    I2_reg = np.copy(I2)
    bad_I2 = I2_reg < 0
    I2_reg[bad_I2] = small_pos
    K3_reg = I2_reg

    # K2
    if ct_field is not None:
        K2_reg = -_safe_gradient(chi * ct_field, z_m, axis=0)
        K2_reg *= baroclinic_scale
    else:
        K2_reg = -_safe_gradient(chi * C_field, z_m, axis=0)

    K2_reg *= sponge_2d
    K3_reg *= sponge_2d

    bad_before = 0
    n_iters = 0
    n_bad = 0

    for k in range(max_iter):
        D_img = K1_reg * K3_reg - K2_reg**2
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


# ==============================================================================
# SOR 迭代求解器
# ==============================================================================

def solve_se_sor(
    A: np.ndarray, B: np.ndarray, C: np.ndarray,
    D: np.ndarray, E: np.ndarray, F: np.ndarray,
    dr: float, dz: float,
    max_iter: int, omega: float, tol: float,
    verbose_every: int,
) -> np.ndarray:
    """
    SOR (Successive Over-Relaxation) 求解 SE 椭圆方程。

    边界条件:
      - r=0: dψ/dr = 0 (轴对称)
      - r=Rmax: dψ/dr = 0 (远场)
      - z=0: ψ = 0 (地面)
      - z=Zmax: dψ/dz = 0 (自由滑移顶)
    """
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
            print(f"Retrying SOR with omega={current_omega} "
                  f"(retry {current_retry}/{max_retries})")

        for it in range(1, max_iter + 1):
            max_res = 0.0

            for i in range(1, nr - 1):
                js = slice(1, nz + 1)
                jj = slice(0, nz)

                if i == 1:
                    ip1, im1 = i + 1, 0
                elif i == nr - 2:
                    ip1, im1 = nr - 1, i - 1
                else:
                    ip1, im1 = i + 1, i - 1

                if i == nr - 2:
                    p_xx = (P[im1, js] - P[i, js]) / dr2
                    p_x = (P[i, js] - P[im1, js]) / dr2_2
                else:
                    p_xx = (P[ip1, js] + P[im1, js] - 2.0 * P[i, js]) / dr2
                    p_x = (P[ip1, js] - P[im1, js]) / dr2_2

                p_xy = (P[ip1, 2:nz+2] - P[ip1, 0:nz]
                        - P[im1, 2:nz+2] + P[im1, 0:nz]) / drdz4

                p_yy = (P[i, 2:nz+2] + P[i, 0:nz] - 2.0 * P[i, js]) / dz2
                p_y = (P[i, 2:nz+2] - P[i, 0:nz]) / dz2_2

                residual = (
                    A[i, jj] * p_xx + B[i, jj] * p_xy
                    + C[i, jj] * p_yy + D[i, jj] * p_x
                    + E[i, jj] * p_y - F[i, jj]
                )

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
                print(f"SOR failed at iter={it}: non-finite residual")
                break

            # 边界条件
            P[0, :] = P[1, :]
            P[-1, :] = P[-2, :]
            P[:, 0] = 0.0
            P[:, -1] = P[:, -2]

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
                )

        break

    if (not converged) and max_iter > 0:
        print("[WARN] SOR未满足残差阈值，输出为未完全收敛解。")

    return P


# ==============================================================================
# 稀疏直接求解器
# ==============================================================================

def solve_se_sparse(
    A: np.ndarray, B: np.ndarray, C: np.ndarray,
    D: np.ndarray, E: np.ndarray, Fin: np.ndarray,
    dr: float, dz: float,
) -> np.ndarray:
    """scipy 稀疏直接求解: 无条件收敛。"""
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
            k = i * nz + j

            # p_xx
            if i == 0:
                c0, c1 = -2.0/dr2, 2.0/dr2
                rows.extend([k, k]); cols.extend([k, k+nz])
                vals.extend([c0*A[i,j], c1*A[i,j]])
            elif i == nr-1:
                c0, c1 = -2.0/dr2, 2.0/dr2
                rows.extend([k, k]); cols.extend([k, k-nz])
                vals.extend([c0*A[i,j], c1*A[i,j]])
            else:
                rows.extend([k, k, k])
                cols.extend([k, k-nz, k+nz])
                vals.extend([-2.0*A[i,j]/dr2, A[i,j]/dr2, A[i,j]/dr2])

            # p_yy
            if j == 0:
                rows.extend([k, k]); cols.extend([k, k+1])
                vals.extend([-2.0*C[i,j]/dz2, C[i,j]/dz2])
            elif j == nz-1:
                rows.extend([k, k]); cols.extend([k, k-1])
                vals.extend([-C[i,j]/dz2, C[i,j]/dz2])
            else:
                rows.extend([k, k, k])
                cols.extend([k, k-1, k+1])
                vals.extend([-2.0*C[i,j]/dz2, C[i,j]/dz2, C[i,j]/dz2])

            # p_xy (B term)
            def _add_xy(ri, rj, sign):
                if 0 <= ri < nr and 0 <= rj < nz:
                    rows.append(k); cols.append(ri*nz+rj)
                    vals.append(sign * B[i,j] / drdz4)

            if i == 0:
                _add_xy(i+1, j+1 if j+1<nz else nz-2, 1.0)
                _add_xy(i+1, j-1, -1.0)
            elif i == nr-1:
                _add_xy(i-1, j+1 if j+1<nz else nz-2, -1.0)
                _add_xy(i-1, j-1, 1.0)
            else:
                _add_xy(i+1, j+1 if j+1<nz else nz-2, 1.0)
                _add_xy(i+1, j-1, -1.0)
                _add_xy(i-1, j+1 if j+1<nz else nz-2, -1.0)
                _add_xy(i-1, j-1, 1.0)

            # p_x (D term)
            if i == 0:
                pass
            elif i == nr-1:
                rows.extend([k, k]); cols.extend([k, k-nz])
                vals.extend([D[i,j]/dr2_2, -D[i,j]/dr2_2])
            else:
                rows.extend([k, k]); cols.extend([k+nz, k-nz])
                vals.extend([D[i,j]/dr2_2, -D[i,j]/dr2_2])

            # p_y (E term)
            if j == 0:
                rows.append(k); cols.append(k+1)
                vals.append(E[i,j]/dz2_2)
            elif j == nz-1:
                pass
            else:
                rows.extend([k, k]); cols.extend([k+1, k-1])
                vals.extend([E[i,j]/dz2_2, -E[i,j]/dz2_2])

            rhs[k] = Fin[i, j]

    M = csr_matrix((vals, (rows, cols)), shape=(N, N))
    p_flat = spsolve(M, rhs)
    P = np.zeros((nr, nz + 2), dtype=np.float64)
    P[:, 1:-1] = p_flat.reshape(nr, nz)
    P[:, 0] = 0.0
    P[:, -1] = P[:, -2]
    return P


# ==============================================================================
# 流函数 → 风速转换
# ==============================================================================

def psi_to_uw(
    psi: np.ndarray, rho_ext: np.ndarray,
    r_m: np.ndarray, dr: float, dz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从流函数 ψ 计算次级环流风速 (U_se, W_se)。

    u = -(1/rρ) ∂ψ/∂z
    w = (1/rρ) ∂ψ/∂r
    """
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


def to_solver_layout_zr_to_rz(field_zr: np.ndarray) -> np.ndarray:
    """(nz, nr) → (nr, nz) 转置为求解器布局。"""
    return np.asarray(field_zr, dtype=np.float64).T


def rho_ext_from_rho_zr(rho_zr: np.ndarray) -> np.ndarray:
    """构建含 ghost cells 的 rho 扩展数组。"""
    rho_rz = to_solver_layout_zr_to_rz(rho_zr)
    nr, nz = rho_rz.shape
    ext = np.zeros((nr, nz + 2), dtype=np.float64)
    ext[:, 1:-1] = rho_rz
    ext[:, 0] = rho_rz[:, 0]
    ext[:, -1] = rho_rz[:, -1]
    return ext


# ==============================================================================
# 源项遮罩
# ==============================================================================

def build_box_mask(
    r_km: np.ndarray, z_km: np.ndarray, box: Dict[str, float]
) -> np.ndarray:
    """根据边界框构建布尔遮罩。"""
    r_min = float(box.get("r_min_km", -np.inf))
    r_max = float(box.get("r_max_km", np.inf))
    z_min = float(box.get("z_min_km", -np.inf))
    z_max = float(box.get("z_max_km", np.inf))
    rr = r_km[None, :]
    zz = z_km[:, None]
    return (rr >= r_min) & (rr <= r_max) & (zz >= z_min) & (zz <= z_max)


def apply_source_mask(
    forcing_thermal: np.ndarray,
    forcing_momentum: np.ndarray,
    r_km: np.ndarray,
    z_km: np.ndarray,
    mask_cfg: SourceMaskConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """对强迫项施加遮罩和缩放。"""
    ft = np.array(forcing_thermal, copy=True, dtype=np.float64)
    fm = np.array(forcing_momentum, copy=True, dtype=np.float64)

    ft *= mask_cfg.thermal_scale
    fm *= mask_cfg.momentum_scale

    for box in mask_cfg.thermal_zero_boxes:
        mask = build_box_mask(r_km, z_km, box)
        ft[mask] = 0.0

    for box in mask_cfg.momentum_zero_boxes:
        mask = build_box_mask(r_km, z_km, box)
        fm[mask] = 0.0

    return ft, fm


# ==============================================================================
# 蒸发冷却强迫项
# ==============================================================================

def build_evap_cooling_Q(
    r_km: np.ndarray,
    z_km: np.ndarray,
    q0: float = -2.0e-4,
    r_center: float = 145.0,
    z_center: float = 15.0,
    r_half: float = 105.0,
    z_half: float = 2.5,
    dipole: bool = False,
    dipole_q_factor: float = 1.0,
    dipole_sigma_z: float = 1.5,
) -> np.ndarray:
    """
    构造层状云降水蒸发冷却的 Q 场 (K/s)。

    默认模式: 二维高斯冷却。
    偶极子模式: 加热-冷却偶极子。
    """
    rr = r_km[None, :]
    zz = z_km[:, None]

    if dipole:
        # 偶极子: 上层冷却 + 下层加热
        q_field = np.zeros((len(z_km), len(r_km)), dtype=np.float64)

        sigma_r = r_half / np.sqrt(2.0 * np.log(2.0))
        sigma_z_cool = z_half / np.sqrt(2.0 * np.log(2.0))
        sigma_z_heat = dipole_sigma_z / np.sqrt(2.0 * np.log(2.0))

        gauss_cool = np.exp(
            -((rr - r_center) / sigma_r)**2 / 2.0
            - ((zz - z_center) / sigma_z_cool)**2 / 2.0
        )
        gauss_heat = np.exp(
            -((rr - r_center) / sigma_r)**2 / 2.0
            - ((zz - (z_center - z_half)) / sigma_z_heat)**2 / 2.0
        )

        q_heat_peak = abs(q0) * dipole_q_factor * (sigma_z_heat / sigma_z_cool)
        q_field = q0 * gauss_cool + q_heat_peak * gauss_heat
    else:
        # 纯冷却高斯
        sigma_r = r_half / np.sqrt(2.0 * np.log(2.0))
        sigma_z = z_half / np.sqrt(2.0 * np.log(2.0))

        q_field = q0 * np.exp(
            -((rr - r_center) / sigma_r)**2 / 2.0
            - ((zz - z_center) / sigma_z)**2 / 2.0
        )

    return q_field


# ==============================================================================
# IEEE 二进制写入
# ==============================================================================

def write_fortran_unformatted_real32(path: Path, arr: np.ndarray) -> None:
    """写入 Fortran unformatted IEEE 二进制文件 (float32)。"""
    data = np.asarray(arr, dtype=np.float32, order="F")
    payload = data.tobytes(order="F")
    nbytes = len(payload)
    with open(path, "wb") as f:
        f.write(struct.pack("<i", nbytes))
        f.write(payload)
        f.write(struct.pack("<i", nbytes))
