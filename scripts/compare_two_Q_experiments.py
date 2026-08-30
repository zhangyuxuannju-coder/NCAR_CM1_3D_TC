#!/usr/bin/env python3
"""
横向对比两组 SE Q 敏感性试验: r=50–300 km vs r=50–200 km
"""

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
EXP1_DIR = Path("output/figures/se_Q_sensitivity")
EXP2_DIR = Path("output/figures/se_Q_sensitivity_r50_200")
OUT_DIR  = Path("output/figures/se_Q_cross_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXP1_LABEL = "r = 50–300 km"
EXP2_LABEL = "r = 50–200 km"

Z_OUTFLOW, Z_INFLOW = 15.0, 11.3
MOD_LABELS = ["Q_m50", "Q_m20", "Q_p20", "Q_p50"]
FACTORS    = [0.5, 0.8, 1.2, 1.5]

# ═══════════════════════════════════════════════════════════════
# 1. 加载数据
# ═══════════════════════════════════════════════════════════════
d1 = np.load(EXP1_DIR / "se_sensitivity_results.npz")
d2 = np.load(EXP2_DIR / "se_sensitivity_results.npz")
r_km = d1["r_km"]; z_km = d1["z_km"]
iz_out = int(np.argmin(np.abs(z_km - Z_OUTFLOW)))
iz_in  = int(np.argmin(np.abs(z_km - Z_INFLOW)))
R, Z = np.meshgrid(r_km, z_km)

def get(res_dict, exp, label, field):
    return res_dict[f"{label}_{field}"]

# ═══════════════════════════════════════════════════════════════
# 图 1: ΔU_se 对比面板 — 两组试验并排 (2行×4列)
# ═══════════════════════════════════════════════════════════════
print("绘图 1: ΔU_se 两组试验对比...")
fig, axes = plt.subplots(2, 4, figsize=(26, 12), dpi=150)

for col, label in enumerate(MOD_LABELS):
    dU1 = get(d1, "EXP1", label, "U_se") - d1["CTRL_U_se"]
    dU2 = get(d2, "EXP2", label, "U_se") - d2["CTRL_U_se"]

    # 统一色标
    dv = float(max(abs(np.nanmin([dU1, dU2])), abs(np.nanmax([dU1, dU2])), 0.3))
    lv = np.linspace(-dv, dv, 31)
    nrm = TwoSlopeNorm(vcenter=0, vmin=-dv, vmax=dv)

    for row, (dU, exp_label, exp_dir) in enumerate([
        (dU1, EXP1_LABEL, EXP1_DIR),
        (dU2, EXP2_LABEL, EXP2_DIR),
    ]):
        ax = axes[row, col]
        im = ax.contourf(R, Z, dU, levels=lv, cmap="RdBu_r", norm=nrm, extend="both")
        ax.contour(R, Z, dU, levels=lv[::4], colors='k', alpha=0.3, linewidths=0.4)
        pct = int((FACTORS[col] - 1) * 100)
        ax.set_title(f"{exp_label}\nQ {'+' if pct>0 else ''}{pct}%", fontsize=11, fontweight="bold")
        ax.set_xlabel("Radius (km)"); ax.set_ylabel("Height (km)")
        ax.set_ylim(0, 20); ax.set_xlim(0, 300)
        ax.grid(True, alpha=0.2, linestyle='--')
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

fig.suptitle("ΔU_se Comparison: r=50–300 km vs r=50–200 km", fontsize=14, fontweight="bold")
fig.tight_layout()
p1 = OUT_DIR / "cross_dU_comparison.png"
fig.savefig(p1, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"  [保存] {p1}")

# ═══════════════════════════════════════════════════════════════
# 图 2: 出流层 & 入流层 两组试验廓线叠加
# ═══════════════════════════════════════════════════════════════
print("绘图 2: 廓线对比...")
colors_q = {"Q_m50": "blue", "Q_m20": "royalblue", "CTRL": "black",
            "Q_p20": "tomato", "Q_p50": "red"}
styles_exp = {EXP1_LABEL: "--", EXP2_LABEL: "-"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), dpi=150)

for ax, iz, zt, ylabel, title in [
    (ax1, iz_out, f"{z_km[iz_out]:.2f}", "U_se (m/s)", "Outflow Layer"),
    (ax2, iz_in,  f"{z_km[iz_in]:.2f}", "U_se (m/s)", "Inflow Layer"),
]:
    for exp_label, d in [(EXP1_LABEL, d1), (EXP2_LABEL, d2)]:
        for ql in ["CTRL"] + MOD_LABELS:
            u = d[f"{ql}_U_se"][iz, :]
            ls = styles_exp[exp_label]
            lw = 2.2 if ql == "CTRL" else 1.3
            alpha = 1.0 if ql == "CTRL" else 0.7
            ax.plot(r_km, u, linestyle=ls, color=colors_q[ql],
                    linewidth=lw, alpha=alpha,
                    label=f"{exp_label} {ql}" if exp_label == EXP1_LABEL else "")

    ax.axhline(y=0, color='gray', alpha=0.4, linestyle='-')
    ax.axvline(x=50, color='gray', alpha=0.3, linestyle=':')
    ax.axvline(x=200, color='gray', alpha=0.3, linestyle=':')
    ax.set_xlabel("Radius (km)", fontsize=12)
    ax.set_ylabel(f"{ylabel} at z≈{zt} km", fontsize=12)
    ax.set_title(f"{title} — z≈{zt} km", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 300); ax.grid(True, alpha=0.3)

# 手动构建图例
from matplotlib.lines import Line2D
leg_q = [Line2D([0],[0], color=colors_q[l], linewidth=2, label=l) for l in ["CTRL","Q_m50","Q_p50"]]
leg_e = [Line2D([0],[0], color='gray', linestyle=ls, linewidth=2, label=lbl)
         for ls, lbl in [("--", EXP1_LABEL), ("-", EXP2_LABEL)]]
ax1.legend(handles=leg_q + leg_e, fontsize=8, ncol=2, loc='upper right')

fig.suptitle("U_se Profiles: r=50–300 km (dashed) vs r=50–200 km (solid)", fontsize=14, fontweight="bold")
fig.tight_layout()
p2 = OUT_DIR / "cross_profiles.png"
fig.savefig(p2, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"  [保存] {p2}")

# ═══════════════════════════════════════════════════════════════
# 图 3: 敏感性趋势对比 (两组叠加)
# ═══════════════════════════════════════════════════════════════
print("绘图 3: 敏感性趋势对比...")
fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
r_targets = [100, 150, 200]
r_colors = {100: "#2196F3", 150: "#4CAF50", 200: "#FF9800"}
exp_markers = {EXP1_LABEL: "o", EXP2_LABEL: "s"}

factors_with_ctrl = [0.5, 0.8, 1.0, 1.2, 1.5]  # 含 CTRL 0% 点
u_out_ctrl_1 = d1["CTRL_U_se"][iz_out, :]; u_in_ctrl_1 = d1["CTRL_U_se"][iz_in, :]
u_out_ctrl_2 = d2["CTRL_U_se"][iz_out, :]; u_in_ctrl_2 = d2["CTRL_U_se"][iz_in, :]

for rc in r_targets:
    ir = int(np.argmin(np.abs(r_km - rc)))
    for exp_label, d, uo_ctrl, ui_ctrl, marker in [
        (EXP1_LABEL, d1, u_out_ctrl_1, u_in_ctrl_1, "o"),
        (EXP2_LABEL, d2, u_out_ctrl_2, u_in_ctrl_2, "s"),
    ]:
        # 出流
        d_out = []
        for label in MOD_LABELS:
            u_mod = d[f"{label}_U_se"][iz_out, ir]
            dp = (u_mod - uo_ctrl[ir]) / abs(uo_ctrl[ir]) * 100
            d_out.append(dp)
        d_out.insert(2, 0.0)  # CTRL 点
        ax.plot(factors_with_ctrl, d_out, marker=marker, linestyle='--', color=r_colors[rc],
                linewidth=1.5, markersize=7, alpha=0.7)

        # 入流
        d_in = []
        for label in MOD_LABELS:
            u_mod = d[f"{label}_U_se"][iz_in, ir]
            dp = (u_mod - ui_ctrl[ir]) / abs(ui_ctrl[ir]) * 100
            d_in.append(dp)
        d_in.insert(2, 0.0)  # CTRL 点
        ax.plot(factors_with_ctrl, d_in, marker=marker, linestyle='-', color=r_colors[rc],
                linewidth=2.0, markersize=7, alpha=0.7)

# CTRL 参考点
ax.scatter([1.0], [0.0], c='black', s=100, zorder=5, marker='D')
ax.annotate('CTRL', (1.0, 0.0), textcoords="offset points", xytext=(8,-12), fontsize=10, fontweight='bold')
ax.axhline(y=0, color='grey', alpha=0.3); ax.axvline(x=1.0, color='grey', alpha=0.2, linestyle=':')

# 图例
leg_r = [Line2D([0],[0], color=r_colors[rc], linewidth=2, label=f"r={rc} km") for rc in r_targets]
leg_e = [Line2D([0],[0], color='gray', marker=m, linestyle=ls, linewidth=2, markersize=7, label=lbl)
         for m, ls, lbl in [("o", "--", EXP1_LABEL), ("s", "-", EXP2_LABEL)]]
ax.legend(handles=leg_r + leg_e, fontsize=9, ncol=2, loc='upper left')
ax.set_xlabel("Q Scaling Factor", fontsize=13); ax.set_ylabel("ΔU_se (%)", fontsize=13)
ax.set_title("Sensitivity Trend Comparison\nDashed=Outflow, Solid=Inflow", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3); ax.set_xlim(0.35, 1.65)
fig.tight_layout()
p3 = OUT_DIR / "cross_trend_comparison.png"
fig.savefig(p3, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"  [保存] {p3}")

# ═══════════════════════════════════════════════════════════════
# 图 4: 两组试验的 Q 场差异 (Q field from exp1 minus exp2, at same factor)
# ═══════════════════════════════════════════════════════════════
print("绘图 4: Q 场差异...")
fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=150)
z_mask = (z_km >= 10) & (z_km <= 17.5)
r_mask = (r_km >= 50) & (r_km <= 200)
z_plot = z_km[z_mask]; r_plot = r_km[r_mask]
Rp, Zp = np.meshgrid(r_plot, z_plot)

for ax, label, title in zip(axes, ["Q_m50", "Q_p50"],
                            ["Q×0.5: EXP1−EXP2", "Q×1.5: EXP1−EXP2"]):
    Q1 = d1[f"{label}_Q"][z_mask,:][:,r_mask]
    Q2 = d2[f"{label}_Q"][z_mask,:][:,r_mask]
    dQ = Q1 - Q2
    dv = float(max(abs(np.nanmin(dQ)), abs(np.nanmax(dQ)), 1e-8))
    lv = np.linspace(-dv, dv, 31)
    nrm = TwoSlopeNorm(vcenter=0, vmin=-dv, vmax=dv)
    im = ax.contourf(Rp, Zp, dQ, levels=lv, cmap="RdBu_r", norm=nrm, extend="both")
    ax.contour(Rp, Zp, dQ, levels=lv[::4], colors='k', alpha=0.3, linewidths=0.4)
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="ΔQ (K/s)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Radius (km)"); ax.set_ylabel("Height (km)")
    ax.set_xlim(50, 200); ax.set_ylim(10, 17.5); ax.set_box_aspect(1.0)
    ax.grid(True, alpha=0.2, linestyle='--')

fig.suptitle("Q Field Difference: EXP1 (r≤300km) − EXP2 (r≤200km)", fontsize=14, fontweight="bold")
fig.tight_layout()
p4 = OUT_DIR / "cross_Q_difference.png"
fig.savefig(p4, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"  [保存] {p4}")

# ═══════════════════════════════════════════════════════════════
# 汇总表格
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*85)
print("横向对比汇总: r=50–300 km (EXP1) vs r=50–200 km (EXP2)")
print("="*85)

for title, iz, zt in [("出流层", iz_out, z_km[iz_out]), ("入流层", iz_in, z_km[iz_in])]:
    print(f"\n--- {title} (z≈{zt:.2f} km) ---")
    print(f"{'Q因子':<10} {'r=100 Δ%':>28} {'r=150 Δ%':>14} {'r=200 Δ%':>14}")
    print(f"{'':10} {'EXP1':>12} {'EXP2':>12} {'EXP1':>12} {'EXP2':>12} {'EXP1':>12} {'EXP2':>12}")
    print("-"*85)

    u_ctrl_1 = d1["CTRL_U_se"][iz, :]; u_ctrl_2 = d2["CTRL_U_se"][iz, :]
    for label in MOD_LABELS:
        print(f"{label:<10}", end="")
        for rc in [100, 150, 200]:
            ir = int(np.argmin(np.abs(r_km - rc)))
            dp1 = (d1[f"{label}_U_se"][iz,ir] - u_ctrl_1[ir]) / abs(u_ctrl_1[ir]) * 100
            dp2 = (d2[f"{label}_U_se"][iz,ir] - u_ctrl_2[ir]) / abs(u_ctrl_2[ir]) * 100
            print(f"  {dp1:+8.2f}% {dp2:+8.2f}%", end="")
        print()

print(f"\n完成! 输出: {OUT_DIR.resolve()}")
