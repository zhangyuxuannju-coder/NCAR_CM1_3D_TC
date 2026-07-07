#!/usr/bin/env python3
"""
绘制非绝热加热场 Q 的上层 R-Z 截面图 (9–17.5 km, 72h)。

用法:
    python scripts/plot_Q_upper.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import xarray as xr

PRODUCTS_FILE = "output/se_pipeline/72h/se_pipeline_products.nc"
OUTPUT_DIR   = Path("output/figures")

ds = xr.open_dataset(str(PRODUCTS_FILE))
Q   = ds["Q"].values
ut  = ds["ut"].values
zh  = ds["zh"].values
r   = ds["radius"].values
ds.close()

R, Z = np.meshgrid(r, zh)

# ── 裁剪到 1–20 km, 50–300 km ─────────────────────────────
z_mask = (zh >= 1.0) & (zh <= 20.0)
r_mask = (r >= 50.0) & (r <= 300.0)
zh_crop = zh[z_mask]
r_crop  = r[r_mask]
Q_crop  = Q[z_mask, :][:, r_mask]
ut_crop = ut[z_mask, :][:, r_mask]
R_crop, Z_crop = np.meshgrid(r_crop, zh_crop)

print(f"zh cropped: {zh_crop[0]:.2f} ~ {zh_crop[-1]:.2f} km ({len(zh_crop)} levels)")
print(f"Q  range:   [{Q_crop.min():.4e}, {Q_crop.max():.4e}] K/s")

# ── 绘图 ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

vmax_q = float(np.nanpercentile(np.abs(Q_crop), 99))
levels_q = np.linspace(-vmax_q, vmax_q, 41)
norm_q = TwoSlopeNorm(vcenter=0, vmin=-vmax_q, vmax=vmax_q)

im = ax.contourf(R_crop, Z_crop, Q_crop, levels=levels_q, cmap="RdBu_r",
                 norm=norm_q, extend="both")
ax.contour(R_crop, Z_crop, Q_crop, levels=levels_q[::4],
           colors='k', alpha=0.35, linewidths=0.5)

# 切向风等值线
ut_levels = np.arange(0, 101, 10)
ax.contour(R_crop, Z_crop, ut_crop, levels=ut_levels,
           colors='green', alpha=0.55, linewidths=0.7, linestyles='dashed')
cs_ut = ax.contour(R_crop, Z_crop, ut_crop, levels=[20, 40, 60, 80],
                   colors='green', alpha=0.8, linewidths=1.0)
ax.clabel(cs_ut, inline=True, fontsize=8, fmt='%d m/s')

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cbar.set_label("Diabatic Heating Q (K/s)", fontsize=12)

ax.set_xlabel("Radius (km)", fontsize=12)
ax.set_ylabel("Height (km)", fontsize=12)
ax.set_title("Diabatic Heating Q (1–20 km, 50–300 km) — 72 h", fontsize=14, fontweight="bold")
ax.set_ylim(1, 20)
ax.set_xlim(50, 300)
ax.grid(True, alpha=0.2, linestyle='--')

from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([0], [0], color='green', lw=1.0, linestyle='dashed',
           label='$V_t$ (m/s)'),
], loc='lower right', fontsize=9, framealpha=0.85)

fig.tight_layout()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
out_png = OUTPUT_DIR / "Q_diabatic_heating_upper_72h.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[完成] 图片已保存: {out_png}")
