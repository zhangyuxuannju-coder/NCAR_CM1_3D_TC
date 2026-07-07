#!/usr/bin/env python3
"""
绘制径向风场 ur 的 R-Z 截面图 (72h)。

从 SE 流水线输出中读取方位角平均的径向风 ur (m/s)，
以填色图+切向风等值线叠加的形式绘制。

用法:
    python scripts/plot_radial_wind.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import xarray as xr

# ── 数据路径 ──────────────────────────────────────────────────
PRODUCTS_FILE = "output/se_pipeline/72h/se_pipeline_products.nc"
OUTPUT_DIR   = Path("output/figures")

# ── 加载数据 ──────────────────────────────────────────────────
ds = xr.open_dataset(str(PRODUCTS_FILE))

ur  = ds["ur"].values         # (zh, radius) 径向风
ut  = ds["ut"].values         # 切向风, 用于叠加等值线
zh  = ds["zh"].values         # km
r   = ds["radius"].values     # km

ds.close()

R, Z = np.meshgrid(r, zh)

print(f"ur shape: {ur.shape}")
print(f"ur range: [{ur.min():.4f}, {ur.max():.4f}] m/s")
print(f"zh range: [{zh.min():.3f}, {zh.max():.1f}] km")
print(f"r  range: [{r.min():.1f}, {r.max():.1f}] km")

# ── 绘图 ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

# ---- 填色: 径向风 ur ----
vmax_u = float(np.nanpercentile(np.abs(ur), 99))
levels_u = np.linspace(-vmax_u, vmax_u, 41)
norm_u = TwoSlopeNorm(vcenter=0, vmin=-vmax_u, vmax=vmax_u)

im = ax.contourf(R, Z, ur, levels=levels_u, cmap="RdBu_r",
                 norm=norm_u, extend="both")

ax.contour(R, Z, ur, levels=levels_u[::4],
           colors='k', alpha=0.35, linewidths=0.5)

# ---- 叠加切向风等值线 (绿色虚线) ----
ut_levels = np.arange(0, 101, 10)
ax.contour(R, Z, ut, levels=ut_levels,
           colors='green', alpha=0.55, linewidths=0.7,
           linestyles='dashed')
cs_ut = ax.contour(R, Z, ut, levels=[20, 40, 60, 80],
                   colors='green', alpha=0.8, linewidths=1.0)
ax.clabel(cs_ut, inline=True, fontsize=8, fmt='%d m/s')

# ---- colorbar ----
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cbar.set_label("Radial Wind $u_r$ (m/s)", fontsize=12)

# ---- 标签与样式 ----
ax.set_xlabel("Radius (km)", fontsize=12)
ax.set_ylabel("Height (km)", fontsize=12)
ax.set_title("Azimuthal-Mean Radial Wind $u_r$ — 72 h", fontsize=14, fontweight="bold")
ax.set_ylim(0, 20)
ax.set_xlim(0, 300)
ax.grid(True, alpha=0.2, linestyle='--')

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
out_png = OUTPUT_DIR / "ur_radial_wind_72h.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"[完成] 图片已保存: {out_png}")
