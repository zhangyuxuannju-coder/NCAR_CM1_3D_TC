#!/usr/bin/env python3
"""
绘制非绝热加热场 Q 的 R-Z 截面图 (72h)。

从 SE 流水线输出中读取方位角平均的 Q 场 (K/s)，
并以填色图+切向风等值线叠加的形式绘制。

用法:
    python scripts/plot_Q_cross_section.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

# ── 数据路径 ──────────────────────────────────────────────────
PRODUCTS_FILE = "output/se_pipeline/72h/se_pipeline_products.nc"
OUTPUT_DIR   = Path("output/figures")

# ── 加载数据 ──────────────────────────────────────────────────
import xarray as xr

ds = xr.open_dataset(str(PRODUCTS_FILE))

Q   = ds["Q"].values          # (zh, radius)
ut  = ds["ut"].values         # 切向风, 用于叠加等值线
zh  = ds["zh"].values         # km
r   = ds["radius"].values     # km

ds.close()

R, Z = np.meshgrid(r, zh)

print(f"Q  shape: {Q.shape}")
print(f"Q  range: [{Q.min():.4e}, {Q.max():.4e}] K/s")
print(f"zh range: [{zh.min():.3f}, {zh.max():.1f}] km")
print(f"r  range: [{r.min():.1f}, {r.max():.1f}] km")

# ── 绘图 ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

# ---- 填色: 非绝热加热 Q ----
# 使用对称色阶 (Q 同时包含加热和冷却)
vmax_q = float(np.nanpercentile(np.abs(Q), 99))
levels_q = np.linspace(-vmax_q, vmax_q, 41)
norm_q = TwoSlopeNorm(vcenter=0, vmin=-vmax_q, vmax=vmax_q)

im = ax.contourf(R, Z, Q, levels=levels_q, cmap="RdBu_r",
                 norm=norm_q, extend="both")

# 叠加等值线 (细黑线, 辅助辨识结构)
ax.contour(R, Z, Q, levels=levels_q[::4],
           colors='k', alpha=0.35, linewidths=0.5)

# ---- 叠加切向风等值线 (绿色虚线) ----
ut_levels = np.arange(0, 101, 10)
ax.contour(R, Z, ut, levels=ut_levels,
           colors='green', alpha=0.55, linewidths=0.7,
           linestyles='dashed')
# 标注切向风
cs_ut = ax.contour(R, Z, ut, levels=[20, 40, 60, 80],
                   colors='green', alpha=0.8, linewidths=1.0)
ax.clabel(cs_ut, inline=True, fontsize=8, fmt='%d m/s')

# ---- colorbar ----
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cbar.set_label("Diabatic Heating Q (K/s)", fontsize=12)

# ---- 标签与样式 ----
ax.set_xlabel("Radius (km)", fontsize=12)
ax.set_ylabel("Height (km)", fontsize=12)
ax.set_title("Diabatic Heating Q — 72 h", fontsize=14, fontweight="bold")
ax.set_ylim(0, 20)
ax.set_xlim(0, 300)
ax.grid(True, alpha=0.2, linestyle='--')

# 标注图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='green', lw=1.0, linestyle='dashed',
           label='Tangential wind $V_t$ (m/s)'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
          framealpha=0.85)

fig.tight_layout()

# ── 保存 ──────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
out_png = OUTPUT_DIR / "Q_diabatic_heating_72h.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"[完成] 图片已保存: {out_png}")
