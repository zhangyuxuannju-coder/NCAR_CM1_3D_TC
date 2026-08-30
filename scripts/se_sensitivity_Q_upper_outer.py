#!/usr/bin/env python3
"""
SE 敏感性试验：上层外区非绝热加热 Q 缩放 → 次级环流响应
=================================================================
修改区域: z ∈ [10, 17.5] km,  r > 50 km
平滑方法: tanh (双曲正切) 锥形过渡 + 轻量 Gaussian 滤波
SOR 参数: ω=1.5, tol=1.5e-9 (手册推荐)

试验组:
  CTRL   — 基准 (factor=1.0)
  Q_m50  — Q 减弱 50% (factor=0.5)
  Q_m20  — Q 减弱 20% (factor=0.8)
  Q_p20  — Q 增强 20% (factor=1.2)
  Q_p50  — Q 增强 50% (factor=1.5)

输出:
  output/figures/se_Q_sensitivity/
    ├── se_wind_comparison.png      — U_se / W_se 三列对比
    ├── se_dU_difference.png        — ΔU_se 差值图
    ├── se_outflow_profile.png      — 出流层 (15.0 km) U_se 廓线
    ├── se_inflow_profile.png       — 入流层 (11.3 km) U_se 廓线
    └── se_sensitivity_results.npz  — 各试验结果数据
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter

from src.se_equation import (
    solve_se_sor,
    psi_to_uw,
    to_solver_layout_zr_to_rz,
    rho_ext_from_rho_zr,
    _safe_gradient,
)

# ══════════════════════════════════════════════════════════════════════
# 配置 — 可按需修改
# ══════════════════════════════════════════════════════════════════════
PRODUCTS_FILE = (
    "/data1/home/zhangyx/project/refactor/output/se_pipeline/thompson/"
    "se_pipeline_products.npz"
)
OUTPUT_DIR = Path("output/figures/se_Q_sensitivity_r50_200")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 目标区域 ──
Z_MIN, Z_MAX = 10.0, 17.5   # km
R_MIN, R_MAX = 50.0, 200.0  # km

# ── 平滑参数 ──
Z_TAPER = 0.5     # km,  垂直过渡半宽度 (tanh参数, 2*delta≈2.5dz, 保证中心区>0.95)
R_TAPER = 8.0     # km,  径向过渡半宽度 (tanh参数, 2*delta≈1.3dr)
GAUSS_SIGMA = 0.0  # 不额外做 Gaussian 滤波 (tanh C∞光滑已足够)

# ── SOR 参数 (手册推荐) ──
SOR_OMEGA = 1.5
SOR_TOL   = 1.5e-9
SOR_MAXITER = 60000

# ── 出流/入流层高度 ──
Z_OUTFLOW = 15.0   # km
Z_INFLOW  = 11.3   # km

# ── 试验组 ──
EXPERIMENTS = {
    "CTRL":  1.0,
    "Q_m50": 0.5,
    "Q_m20": 0.8,
    "Q_p20": 1.2,
    "Q_p50": 1.5,
}

G = 9.80665


# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════

def smooth_2d_3point(field: np.ndarray) -> np.ndarray:
    """NCL 风格三点移动平均: 先在 z 方向, 再在 r 方向各做一次 0.25-0.5-0.25 平滑"""
    out = np.array(field, copy=True, dtype=np.float64)
    nz, nr = out.shape
    # z 方向
    for i in range(nr):
        col = out[:, i].copy()
        for k in range(1, nz - 1):
            out[k, i] = 0.25 * col[k - 1] + 0.5 * col[k] + 0.25 * col[k + 1]
    # r 方向
    for k in range(nz):
        row = out[k, :].copy()
        for i in range(1, nr - 1):
            out[k, i] = 0.25 * row[i - 1] + 0.5 * row[i] + 0.25 * row[i + 1]
    return out


def repair_nan_2d(field: np.ndarray) -> np.ndarray:
    """与 refactor 一致的 NaN 修复: 优先用左邻, 其次用上邻, 否则置零"""
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
    if n_bad:
        print(f"  [repair_nan] 修复 {n_bad} 个非法值")
    return out


def build_tanh_taper(
    r_m: np.ndarray, z_m: np.ndarray,
    z_min: float, z_max: float, r_min: float, r_max: float,
    delta_z: float, delta_r: float,
) -> np.ndarray:
    """
    构建平滑二维锥形权重 (taper mask), 使用 tanh 函数。

    平滑原理:
      - 垂直方向: 在 z=z_min 和 z=z_max 处各有一个 tanh 过渡
        w_lower(z) = 0.5 * (1 + tanh((z - z_min) / delta_z))
        w_upper(z) = 0.5 * (1 + tanh((z_max - z) / delta_z))
      - 径向方向: 在 r=r_min 和 r=r_max 处各有一个 tanh 过渡
        w_radial_lower(r) = 0.5 * (1 + tanh((r - r_min) / delta_r))
        w_radial_upper(r) = 0.5 * (1 + tanh((r_max - r) / delta_r))
      - 总权重 = w_lower * w_upper * w_radial_lower * w_radial_upper

    tanh 函数特性:
      - tanh(0) = 0        → 权重 = 0.5 在边界上
      - tanh(1) ≈ 0.762   → 权重从 0.12 到 0.88 跨越约 2*delta
      - tanh(2) ≈ 0.964   → 权重从 0.02 到 0.98 跨越约 4*delta
      - tanh 是 C∞ 光滑函数, 保证所有阶导数连续

    修改 Q 时: Q_new = Q_orig * (1 + (factor - 1) * taper)
      - taper ≈ 0 区域: Q_new ≈ Q_orig (未修改)
      - taper ≈ 1 区域: Q_new ≈ Q_orig * factor (完全缩放)
      - 过渡区域: 光滑插值, 避免导数突变导致的虚假波纹

    为什么这样能避免虚假波纹:
      SE 方程中 Q 通过其梯度 dQ/dr, dQ/dz 进入强迫项。若修改区域
      边界处 Q 不光滑, 其导数会出现尖峰, 导致 SOR 解出现 Gibbs 现象。
      tanh 函数保证 taper 的所有阶导数连续, 因此 Q_new 的梯度在边界
      处也是光滑的——没有虚假源项 → 没有虚假波纹。
    """
    Rg, Zg = np.meshgrid(r_m, z_m)  # Rg: (nz, nr), Zg: (nz, nr)

    z_lower = 0.5 * (1.0 + np.tanh((Zg - z_min) / delta_z))
    z_upper = 0.5 * (1.0 + np.tanh((z_max - Zg) / delta_z))
    r_lower = 0.5 * (1.0 + np.tanh((Rg - r_min) / delta_r))
    r_upper = 0.5 * (1.0 + np.tanh((r_max - Rg) / delta_r))

    taper = z_lower * z_upper * r_lower * r_upper
    return np.clip(taper, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("SE 敏感性试验: 上层外区 Q 缩放 → 次级环流响应")
    print("=" * 65)

    # ── 1. 加载基准 SE 产品 ──────────────────────────────────────
    print("\n[1/5] 加载基准 SE 产品...")
    data = np.load(PRODUCTS_FILE)
    r_km = data["r_km"].astype(np.float64)
    z_km = data["z_km"].astype(np.float64)
    nz, nr = len(z_km), len(r_km)

    r_m = r_km * 1000.0
    z_m = z_km * 1000.0
    dr = float(np.mean(np.diff(r_m)))
    dz = float(np.mean(np.diff(z_m)))

    Q_orig     = data["Q"].astype(np.float64)
    ut_2d      = data["ut"].astype(np.float64)
    rho_2d     = data["rho"].astype(np.float64)
    theta_bal  = data["theta_bal"].astype(np.float64)

    # 已正则化的 SE 系数
    A_s = data["A"].astype(np.float64)
    B_s = data["B"].astype(np.float64)
    C_s = data["C"].astype(np.float64)
    D_s = data["D"].astype(np.float64)
    E_s = data["E"].astype(np.float64)
    forcing_momentum = data["forcing_momentum"].astype(np.float64)

    print(f"  网格: nz={nz}, nr={nr}, dr≈{dr/1000:.1f}km, dz≈{dz:.0f}m")
    print(f"  Q 原始范围: [{Q_orig.min():.2e}, {Q_orig.max():.2e}] K/s")
    print(f"  出流层 iz≈{np.argmin(np.abs(z_km - Z_OUTFLOW))} "
          f"(z={z_km[np.argmin(np.abs(z_km - Z_OUTFLOW))]:.3f} km)")
    print(f"  入流层 iz≈{np.argmin(np.abs(z_km - Z_INFLOW))} "
          f"(z={z_km[np.argmin(np.abs(z_km - Z_INFLOW))]:.3f} km)")

    # ── 2. 预计算与 Q 无关的量 ──────────────────────────────────
    print("\n[2/5] 预计算系数 (与 Q 无关部分)...")

    # chi = 1/theta, 应用 3 点平滑 (与 refactor 一致)
    chi_raw = 1.0 / np.maximum(theta_bal, 1.0)
    chi = smooth_2d_3point(chi_raw)
    rho_s = smooth_2d_3point(rho_2d)

    # C_field = d(vt)/dz
    C_field = _safe_gradient(ut_2d, z_m, axis=0)

    # 转为求解器布局 (与 refactor 一致，提前做一次)
    A_rz = to_solver_layout_zr_to_rz(A_s)
    B_rz = to_solver_layout_zr_to_rz(B_s)
    C_rz = to_solver_layout_zr_to_rz(C_s)
    D_rz = to_solver_layout_zr_to_rz(D_s)
    E_rz = to_solver_layout_zr_to_rz(E_s)
    rho_ext = rho_ext_from_rho_zr(rho_s)

    # ── 3. 构建 taper ──────────────────────────────────────────
    print("\n[3/5] 构建平滑 taper...")
    taper = build_tanh_taper(
        r_m, z_m,
        z_min=Z_MIN * 1000.0, z_max=Z_MAX * 1000.0,
        r_min=R_MIN * 1000.0, r_max=R_MAX * 1000.0,
        delta_z=Z_TAPER * 1000.0,
        delta_r=R_TAPER * 1000.0,
    )
    inner = (taper > 0.95)
    trans = (taper >= 0.05) & (taper <= 0.95)
    print(f"  taper 内部 (>0.95): {inner.sum()} 点")
    print(f"  taper 过渡 (0.05-0.95): {trans.sum()} 点")
    print(f"  taper 外部 (<0.05): {(taper < 0.05).sum()} 点")
    Q_inner = Q_orig[inner]
    if len(Q_inner) > 0:
        print(f"  目标区 Q 均值: {Q_inner.mean():.2e} K/s")

    # ── 4. 各试验: 修改 Q → 重算 F → SOR 求解 ─────────────────
    print(f"\n[4/5] 运行 {len(EXPERIMENTS)} 组试验 (SOR ω={SOR_OMEGA}, tol={SOR_TOL})...")
    results = {}

    for label, factor in EXPERIMENTS.items():
        print(f"\n  {'─'*50}")
        print(f"  处理: {label} (factor={factor})")

        # 4a. 修改 Q
        if factor == 1.0:
            Q_mod = Q_orig.copy()
        else:
            # 核心公式: Q_new = Q_orig × [1 + (factor-1) × taper]
            Q_mod = Q_orig * (1.0 + (factor - 1.0) * taper)
            # 轻量 Gaussian 滤波抑制格点尺度噪声
            # sigma=1 格点 → 只影响 ~2Δx 尺度, 不影响物理结构
            Q_mod = gaussian_filter(Q_mod, sigma=GAUSS_SIGMA)

        # 4b. 重算热力强迫项 (与 refactor build_se_coefficients 一致)
        #   thermal_flux = χ² · Q
        #   forcing_thermal = G · ∂(thermal_flux)/∂r  +  ∂(C · thermal_flux)/∂z
        thermal_flux = (chi ** 2) * Q_mod
        forcing_thermal = (
            G * _safe_gradient(thermal_flux, r_m, axis=1)
            + _safe_gradient(C_field * thermal_flux, z_m, axis=0)
        )
        F_new = forcing_thermal + forcing_momentum
        F_new = repair_nan_2d(F_new)

        # 4c. SOR 求解 (使用与 refactor 一致的参数)
        F_rz = to_solver_layout_zr_to_rz(F_new)
        print(f"    SOR 求解中 (max_iter={SOR_MAXITER})...")
        psi_rz = solve_se_sor(
            A_rz, B_rz, C_rz, D_rz, E_rz, F_rz,
            dr, dz,
            max_iter=SOR_MAXITER,
            omega=SOR_OMEGA,
            tol=SOR_TOL,
            verbose_every=0,
        )

        # 4d. psi → U_se, W_se
        U_rz, W_rz = psi_to_uw(psi_rz, rho_ext, r_m, dr, dz)
        U_se = U_rz[:, 1:-1].T  # (nz, nr)
        W_se = W_rz[:, 1:-1].T

        results[label] = {
            "Q": Q_mod, "U_se": U_se, "W_se": W_se,
            "psi_rz": psi_rz,
        }
        print(f"    U_se: [{U_se.min():.3f}, {U_se.max():.3f}] m/s")
        print(f"    W_se: [{W_se.min():.4f}, {W_se.max():.4f}] m/s")

    # ── 验证 CTRL 与 refactor 原始解的一致性 ─────────────────
    print(f"\n  {'─'*50}")
    U_ref = data["U_se"][:, 1:-1].T
    U_ctrl = results["CTRL"]["U_se"]
    diff_ctrl = np.abs(U_ref - U_ctrl)
    print(f"  |U_ctrl - U_ref|: max={diff_ctrl.max():.3e}, mean={diff_ctrl.mean():.3e}")
    if diff_ctrl.max() > 0.1:
        print("  ⚠ CTRL 与 refactor 差异较大，请检查！")

    # ── 5. 保存数据 ──────────────────────────────────────────────
    print("\n[5/5] 保存结果与绘图...")
    npz_out = OUTPUT_DIR / "se_sensitivity_results.npz"
    save_dict = {
        "r_km": r_km, "z_km": z_km,
        "ut": ut_2d, "rho": rho_2d,
        "Z_OUTFLOW": Z_OUTFLOW, "Z_INFLOW": Z_INFLOW,
    }
    for label, res in results.items():
        for key in ["Q", "U_se", "W_se"]:
            save_dict[f"{label}_{key}"] = res[key]
    np.savez_compressed(npz_out, **save_dict)
    print(f"  [保存] {npz_out}")

    # ══════════════════════════════════════════════════════════════
    # 绘图
    # ══════════════════════════════════════════════════════════════
    R, Z = np.meshgrid(r_km, z_km)
    iz_out = int(np.argmin(np.abs(z_km - Z_OUTFLOW)))
    iz_in  = int(np.argmin(np.abs(z_km - Z_INFLOW)))
    ir50   = int(np.argmin(np.abs(r_km - R_MIN)))

    colors  = {"CTRL": "black", "Q_m50": "blue", "Q_m20": "royalblue",
               "Q_p20": "tomato", "Q_p50": "red"}
    styles  = {"CTRL": "-", "Q_m50": "--", "Q_m20": "-.",
               "Q_p20": "-.", "Q_p50": "--"}
    lws     = {"CTRL": 2.2, "Q_m50": 1.5, "Q_m20": 1.5,
               "Q_p20": 1.5, "Q_p50": 1.5}

    # ── 图 1: U_se / W_se 面板 (5行×2列) ────────────────────────
    print("  绘图 1: 风场对比...")
    labels_ordered = ["CTRL", "Q_m50", "Q_m20", "Q_p20", "Q_p50"]
    fig, axes = plt.subplots(5, 2, figsize=(20, 28), dpi=150)

    # 确定全局色标范围 (排除 CTRL 避免被其他试验的极端值影响)
    all_U = [results[l]["U_se"] for l in labels_ordered]
    all_W = [results[l]["W_se"] for l in labels_ordered]
    # CTRL 也是合理范围的一部分
    U_lim = float(np.nanpercentile(np.abs(np.stack(all_U)), 98))
    W_lim = float(np.nanpercentile(np.abs(np.stack(all_W)), 98))
    U_lim = max(U_lim, 5.0)
    W_lim = max(W_lim, 1.0)

    for row, label in enumerate(labels_ordered):
        res = results[label]
        for col, (field, clim, vname, unit) in enumerate([
            (res["U_se"], U_lim, "U_se", "m/s"),
            (res["W_se"], W_lim, "W_se", "m/s"),
        ]):
            ax = axes[row, col]
            f = np.clip(field, -clim, clim)
            lv = np.linspace(-clim, clim, 31)
            nm = TwoSlopeNorm(vcenter=0, vmin=-clim, vmax=clim)
            im = ax.contourf(R, Z, f, levels=lv, cmap="RdBu_r", norm=nm, extend="both")
            ax.contour(R, Z, f, levels=lv[::4], colors='k', alpha=0.3, linewidths=0.4)
            # 叠加切向风等值线
            ax.contour(R, Z, ut_2d, levels=np.arange(0, 101, 20),
                       colors='green', alpha=0.45, linewidths=0.6, linestyles='dashed')
            # 标注修改区域
            ax.fill_between([R_MIN, R_MAX], Z_MIN, Z_MAX,
                            alpha=0.06, color='yellow')
            ax.axhline(y=Z_MIN, color='orange', alpha=0.4, linestyle='--', linewidth=0.6)
            ax.axhline(y=Z_MAX, color='orange', alpha=0.4, linestyle='--', linewidth=0.6)
            ax.axvline(x=R_MIN, color='orange', alpha=0.4, linestyle='--', linewidth=0.6)
            ax.axvline(x=R_MAX, color='orange', alpha=0.4, linestyle='--', linewidth=0.6)

            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            title = f"{label} — {vname} ({unit})"
            if label != "CTRL":
                pct = int((EXPERIMENTS[label] - 1) * 100)
                title += f"  [Q {'+' if pct>0 else ''}{pct}%]"
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_ylabel("Height (km)")
            ax.set_ylim(0, 20)
            ax.set_xlim(0, 300)
            ax.grid(True, alpha=0.2, linestyle='--')
            if row == 4:
                ax.set_xlabel("Radius (km)")

    fig.suptitle(
        f"SE Secondary Circulation — Q Sensitivity (z={Z_MIN}–{Z_MAX} km, r={R_MIN}–{R_MAX} km)",
        fontsize=15, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    p1 = OUTPUT_DIR / "se_wind_comparison.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [保存] {p1}")

    # ── 图 2: ΔU_se 差值图 ──────────────────────────────────────
    print("  绘图 2: ΔU_se 差值图...")
    deltas = [l for l in labels_ordered if l != "CTRL"]
    fig, axes = plt.subplots(2, 2, figsize=(20, 13), dpi=150)
    for idx, label in enumerate(deltas):
        ax = axes[idx // 2, idx % 2]
        dU = results[label]["U_se"] - results["CTRL"]["U_se"]
        dv = float(max(abs(np.nanmin(dU)), abs(np.nanmax(dU)), 0.5))
        lv = np.linspace(-dv, dv, 31)
        nm = TwoSlopeNorm(vcenter=0, vmin=-dv, vmax=dv)
        im = ax.contourf(R, Z, dU, levels=lv, cmap="RdBu_r", norm=nm, extend="both")
        ax.contour(R, Z, dU, levels=lv[::4], colors='k', alpha=0.3, linewidths=0.4)
        ax.fill_between([R_MIN, R_MAX], Z_MIN, Z_MAX,
                        alpha=0.06, color='yellow')
        ax.axhline(y=Z_MIN, color='orange', alpha=0.5, linestyle='--', linewidth=0.8)
        ax.axhline(y=Z_MAX, color='orange', alpha=0.5, linestyle='--', linewidth=0.8)
        ax.axvline(x=R_MIN, color='orange', alpha=0.5, linestyle='--', linewidth=0.8)
        ax.axvline(x=R_MAX, color='orange', alpha=0.5, linestyle='--', linewidth=0.8)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        pct = int((EXPERIMENTS[label] - 1) * 100)
        ax.set_title(f"ΔU_se: Q {'+' if pct>0 else ''}{pct}% − CTRL", fontsize=12, fontweight="bold")
        ax.set_xlabel("Radius (km)")
        ax.set_ylabel("Height (km)")
        ax.set_ylim(0, 20)
        ax.set_xlim(0, 300)
        ax.grid(True, alpha=0.2, linestyle='--')
    fig.suptitle("U_se Response to Upper-Outer Q Modification", fontsize=14, fontweight="bold")
    fig.tight_layout()
    p2 = OUTPUT_DIR / "se_dU_difference.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [保存] {p2}")

    # ── 图 3: 出流层廓线 (Z_OUTFLOW km), 仅 r>50km, 仅正值 ────
    print(f"  绘图 3: 出流层廓线 (z≈{z_km[iz_out]:.2f} km, r>50km, U_se>0)...")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    mask_r = r_km >= R_MIN
    for label in labels_ordered:
        ax.plot(r_km[mask_r], results[label]["U_se"][iz_out, :][mask_r],
                color=colors[label], linestyle=styles[label],
                linewidth=lws[label], label=label,
                marker='o', markersize=3.5, markevery=2)
    ax.axhline(y=0, color='gray', alpha=0.5, linestyle='-')
    ax.set_xlabel("Radius (km)", fontsize=12)
    ax.set_ylabel(f"U_se at z≈{z_km[iz_out]:.2f} km (m/s)", fontsize=12)
    ax.set_title(
        f"Outflow Layer — $U_{{se}}$ at z≈{z_km[iz_out]:.2f} km "
        f"(r = {R_MIN}–{R_MAX} km, positive only)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, ncol=5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(R_MIN, R_MAX + 5)
    # 仅显示正值范围
    y_pos = np.concatenate([results[l]["U_se"][iz_out, :][mask_r] for l in labels_ordered])
    y_pos = y_pos[y_pos > 0]
    y_max = float(np.nanmax(y_pos)) * 1.15 if len(y_pos) > 0 else 15.0
    ax.set_ylim(0, y_max)
    fig.tight_layout()
    p3 = OUTPUT_DIR / "se_outflow_profile.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [保存] {p3}")

    # ── 图 4: 入流层廓线 (Z_INFLOW km), 仅 r>50km, 仅负值 0~-3 ─
    print(f"  绘图 4: 入流层廓线 (z≈{z_km[iz_in]:.2f} km, r>50km, U_se<0)...")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    mask_r = r_km >= R_MIN
    for label in labels_ordered:
        ax.plot(r_km[mask_r], results[label]["U_se"][iz_in, :][mask_r],
                color=colors[label], linestyle=styles[label],
                linewidth=lws[label], label=label,
                marker='o', markersize=3.5, markevery=2)
    ax.axhline(y=0, color='gray', alpha=0.5, linestyle='-')
    ax.set_xlabel("Radius (km)", fontsize=12)
    ax.set_ylabel(f"U_se at z≈{z_km[iz_in]:.2f} km (m/s)", fontsize=12)
    ax.set_title(
        f"Inflow Layer — $U_{{se}}$ at z≈{z_km[iz_in]:.2f} km "
        f"(r = {R_MIN}–{R_MAX} km, 0 ~ −3 m/s)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, ncol=5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(R_MIN, R_MAX + 5)
    ax.set_ylim(-3.0, 0.0)
    fig.tight_layout()
    p4 = OUTPUT_DIR / "se_inflow_profile.png"
    fig.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [保存] {p4}")

    # ── 图 5: 相对变化趋势 — Q 变化 vs ΔU_se% ──────────────────
    print("  绘图 5: 相对变化趋势 (Q factor → ΔU_se%)...")
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

    mod_labels = ["Q_m50", "Q_m20", "Q_p20", "Q_p50"]
    # 加入 CTRL 点让折线穿过 (1.0, 0%)
    factors_arr = np.array([0.5, 0.8, 1.0, 1.2, 1.5])
    r_targets = [100, 150, 200]
    # 为每个半径选色
    r_colors = {100: "#2196F3", 150: "#4CAF50", 200: "#FF9800"}  # 蓝、绿、橙

    u_out_ctrl_prof = results["CTRL"]["U_se"][iz_out, :]
    u_in_ctrl_prof  = results["CTRL"]["U_se"][iz_in, :]

    for rc in r_targets:
        ir = int(np.argmin(np.abs(r_km - rc)))

        # 出流层: 虚线 (含 CTRL 0%)
        d_out = []
        for label in mod_labels:
            u_mod = results[label]["U_se"][iz_out, ir]
            u_ref = u_out_ctrl_prof[ir]
            dp = (u_mod - u_ref) / abs(u_ref) * 100
            d_out.append(dp)
        # 在 factor=1.0 处插入 0%
        d_out.insert(2, 0.0)
        ax.plot(factors_arr, d_out, 'o--', color=r_colors[rc],
                linewidth=2.0, markersize=8,
                label=f"Outflow r={rc} km")

        # 入流层: 实线 (含 CTRL 0%)
        d_in = []
        for label in mod_labels:
            u_mod = results[label]["U_se"][iz_in, ir]
            u_ref = u_in_ctrl_prof[ir]
            dp = (u_mod - u_ref) / abs(u_ref) * 100
            d_in.append(dp)
        d_in.insert(2, 0.0)
        ax.plot(factors_arr, d_in, 's-', color=r_colors[rc],
                linewidth=2.0, markersize=8,
                label=f"Inflow  r={rc} km")

    # CTRL 参考点 (0%)
    ax.scatter([1.0, 1.0], [0.0, 0.0], c='black', s=80, zorder=5, marker='D')
    ax.annotate('CTRL', (1.0, 0.0), textcoords="offset points",
                xytext=(8, -12), fontsize=10, fontweight='bold', color='black')

    ax.axhline(y=0, color='grey', alpha=0.4, linestyle='-', linewidth=0.8)
    ax.axvline(x=1.0, color='grey', alpha=0.3, linestyle=':', linewidth=0.8)

    ax.set_xlabel("Q Scaling Factor", fontsize=13)
    ax.set_ylabel("ΔU_se relative to CTRL (%)", fontsize=13)
    ax.set_title(
        f"SE Sensitivity: Q Scaling → U_se Response\n"
        f"Outflow z≈{z_km[iz_out]:.2f} km (dashed)  |  "
        f"Inflow z≈{z_km[iz_in]:.2f} km (solid)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.35, 1.65)
    fig.tight_layout()
    p5 = OUTPUT_DIR / "se_sensitivity_trend.png"
    fig.savefig(p5, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [保存] {p5}")

    # ── 图 6: CTRL Q 热力场 (10–17.5 km, 50–200 km) ──────────
    print("  绘图 6: CTRL Q 热力场 (z=10–17.5 km, r=50–200 km)...")

    z_mask = (z_km >= Z_MIN) & (z_km <= Z_MAX)
    r_mask = (r_km >= R_MIN) & (r_km <= R_MAX)
    z_plot = z_km[z_mask]
    r_plot = r_km[r_mask]
    R_plot, Z_plot = np.meshgrid(r_plot, z_plot)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    Q_plot = results["CTRL"]["Q"][z_mask, :][:, r_mask]

    vmin = float(np.nanmin(Q_plot))
    vmax = float(np.nanmax(Q_plot))
    lp = np.linspace(0, vmax, 20)[1:] if vmax > 0 else []
    ln = np.linspace(vmin, 0, 20)[:-1] if vmin < 0 else []
    lv = np.unique(np.concatenate([ln, lp]))
    if len(lv) < 3:
        lv = np.linspace(vmin, vmax, 31)

    if vmin < 0 and vmax > 0:
        nrm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
    else:
        nrm = None

    im = ax.contourf(R_plot, Z_plot, Q_plot, levels=lv,
                     cmap="RdBu_r", norm=nrm, extend="both")
    ax.contour(R_plot, Z_plot, Q_plot, levels=lv[::3],
               colors='k', alpha=0.35, linewidths=0.4)

    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Q (K/s)", fontsize=11)

    ax.set_xlabel("Radius (km)", fontsize=12)
    ax.set_ylabel("Height (km)", fontsize=12)
    ax.set_title(
        f"CTRL Diabatic Heating Q  |  z = {Z_MIN}–{Z_MAX} km, r = {R_MIN}–{R_MAX} km",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(R_MIN, R_MAX)
    ax.set_ylim(Z_MIN, Z_MAX)
    ax.set_box_aspect(1.0)
    ax.grid(True, alpha=0.2, linestyle='--')
    fig.tight_layout()
    p6 = OUTPUT_DIR / "se_Q_field.png"
    fig.savefig(p6, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [保存] {p6}")

    # ── 汇总 ─────────────────────────────────────────────────────
    print(f"\n  {'='*55}")
    print(f"  出流层 (z≈{z_km[iz_out]:.2f} km) U_se 汇总:")
    print(f"  {'试验':<10} {'r=50km':>10} {'r=100km':>10} {'r=150km':>10} {'r=200km':>10}")
    for label in labels_ordered:
        u_prof = results[label]["U_se"][iz_out, :]
        print(f"  {label:<10} "
              f"{u_prof[np.argmin(np.abs(r_km-50))]:>10.4f} "
              f"{u_prof[np.argmin(np.abs(r_km-100))]:>10.4f} "
              f"{u_prof[np.argmin(np.abs(r_km-150))]:>10.4f} "
              f"{u_prof[np.argmin(np.abs(r_km-200))]:>10.4f}")

    print(f"\n  入流层 (z≈{z_km[iz_in]:.2f} km) U_se 汇总:")
    print(f"  {'试验':<10} {'r=50km':>10} {'r=100km':>10} {'r=150km':>10} {'r=200km':>10}")
    for label in labels_ordered:
        u_prof = results[label]["U_se"][iz_in, :]
        print(f"  {label:<10} "
              f"{u_prof[np.argmin(np.abs(r_km-50))]:>10.4f} "
              f"{u_prof[np.argmin(np.abs(r_km-100))]:>10.4f} "
              f"{u_prof[np.argmin(np.abs(r_km-150))]:>10.4f} "
              f"{u_prof[np.argmin(np.abs(r_km-200))]:>10.4f}")

    print(f"\n{'='*65}")
    print(f"完成！所有输出: {OUTPUT_DIR.resolve()}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
