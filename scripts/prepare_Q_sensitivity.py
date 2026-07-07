#!/usr/bin/env python3
"""
准备 SE 敏感性试验：创建目标区域缩放后的 Q 场 .npy 文件。

目标区域: z = 11–16 km, r > 50 km
平滑过渡: sigmoid taper 避免导数突变

输出:
  - output/sensitivity_Q/Q_CTRL.npy     (原始 Q)
  - output/sensitivity_Q/Q_x1.5.npy     (增强 50%)
  - output/sensitivity_Q/Q_x0.5.npy     (减弱 50%)
"""

import numpy as np
import xarray as xr
from scipy.special import expit
from pathlib import Path

PRODUCTS_FILE = "output/se_pipeline/72h/se_pipeline_products.nc"
OUTPUT_DIR   = Path("output/sensitivity_Q")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 目标区域 & taper 参数
Z_MIN, Z_MAX = 11.0, 16.0   # km
R_MIN = 50.0                 # km
Z_TAPER = 1.0                # km
R_TAPER = 15.0               # km

FACTORS = {"CTRL": 1.0, "Q_x1.5": 1.5, "Q_x0.5": 0.5}

# ── 加载原始 Q ──────────────────────────────────────────────
ds = xr.open_dataset(PRODUCTS_FILE)
zh  = ds["zh"].values.astype(np.float64)
r   = ds["radius"].values.astype(np.float64)
Q0  = ds["Q"].values.astype(np.float64)
ds.close()

z_m = zh * 1000.0
r_m = r * 1000.0
print(f"Q 原始: min={Q0.min():.4e}, max={Q0.max():.4e}, shape={Q0.shape}")

# ── 构建平滑 taper (同之前) ─────────────────────────────────
Rg, Zg = np.meshgrid(r_m, z_m)  # Rg各行=r_m, Zg各列=z_m
z_lower = expit((Zg - Z_MIN*1000) / (Z_TAPER*1000 / 4.0))
z_upper = expit(-(Zg - Z_MAX*1000) / (Z_TAPER*1000 / 4.0))
z_t = z_lower * z_upper
r_t = expit((Rg - R_MIN*1000) / (R_TAPER*1000 / 4.0))
taper = z_t * r_t

print(f"taper: max={taper.max():.4f}, >0.9={np.sum(taper>0.9)}点, "
      f"目标区Q均值={Q0[taper>0.9].mean():.4e}")

# ── 生成并保存各试验 Q 场 ──────────────────────────────────
for label, factor in FACTORS.items():
    if factor == 1.0:
        Q_out = Q0.copy()
    else:
        Q_out = Q0 * (1.0 + (factor - 1.0) * taper)

    fname = OUTPUT_DIR / f"Q_{label}.npy"
    np.save(fname, Q_out)
    print(f"[保存] {fname}  |  Q范围: [{Q_out.min():.4e}, {Q_out.max():.4e}]")

print("\n准备完成，接下来运行流水线。")
