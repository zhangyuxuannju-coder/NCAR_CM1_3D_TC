#!/usr/bin/env python3
"""
绘制方位角平均雷达回波 (dbz) 垂直剖面图。
目的: 判断高层是否为层云降水 (dbz < 35 dBZ)。

使用方法与 refactor SE 诊断完全一致:
  - t = 72h
  - 从 psfc 最小值找台风中心
  - 交错网格去交错到标量网格
  - np.digitize + np.bincount 方位角平均
"""

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════
INPUT_FILE = "/data1/home/zhangyx/data/cm1out_thompson.nc"
OUTPUT_DIR = Path("output/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_HOURS = 72.0
DR_KM = 12.0       # 与 SE 诊断一致
MAX_R_KM = 300.0
MAX_Z_KM = 20.0
DBZ_STRATIFORM = 35.0  # 层云/对流分界

# ═══════════════════════════════════════════════════════════════
# 1. 加载数据
# ═══════════════════════════════════════════════════════════════
print("加载数据...")
ds = xr.open_dataset(INPUT_FILE)

xh = np.asarray(ds["xh"], dtype=np.float64)
yh = np.asarray(ds["yh"], dtype=np.float64)
zh = np.asarray(ds["zh"], dtype=np.float64)
time_vals = np.asarray(ds["time"], dtype=np.float64)

# 选时间
target_sec = TARGET_HOURS * 3600.0
time_idx = int(np.argmin(np.abs(time_vals - target_sec)))
print(f"  时间: index={time_idx}, t={time_vals[time_idx]/3600:.1f} h")

# 垂直裁剪
z_mask = zh <= MAX_Z_KM
zh_sel = zh[z_mask]
nz = len(zh_sel)

# ── 去交错 dbz 到标量网格 ──
# dbz 在 zh 维上是交错的 (59层 vs zh=60层), 需要去交错
dbz_full = np.asarray(ds["dbz"].isel(time=time_idx), dtype=np.float64)  # (zh_stag, yh, xh)
nz_stag, ny, nx = dbz_full.shape
print(f"  dbz 原始: shape={dbz_full.shape}, zh 标量层数={len(zh)}")

# 去交错: 对 zh 维做线性插值 (W点→标量点)
# CM1: zh 是标量高度, dbz 的 zh 维是 W 点 (交错)
# 方法: (dbz[k] + dbz[k+1]) / 2, 对应标量层 k=0..nz_stag-2
if dbz_full.shape[0] == len(zh) - 1:
    # W点→标量点: 用相邻平均
    dbz_scalar = np.zeros((len(zh), ny, nx), dtype=np.float64)
    dbz_scalar[0, :, :] = dbz_full[0, :, :]   # 底边界: 取 W 点值
    for k in range(1, len(zh) - 1):
        dbz_scalar[k, :, :] = 0.5 * (dbz_full[k-1, :, :] + dbz_full[k, :, :])
    dbz_scalar[-1, :, :] = dbz_full[-1, :, :]  # 顶边界
    print(f"  去交错: {dbz_full.shape[0]} → {len(zh)} (W→标量)")
elif dbz_full.shape[0] == len(zh):
    dbz_scalar = dbz_full
    print(f"  已为标量网格，无需去交错")
else:
    raise ValueError(f"dbz 垂直维度 {dbz_full.shape[0]}, zh={len(zh)}, 无法判断交错方式")

dbz_scalar = dbz_scalar[z_mask, :, :]  # 裁剪高度

# ── 找台风中心 (用 psfc 最低气压) ──
psfc = np.asarray(ds["psfc"].isel(time=time_idx), dtype=np.float64)
iy_min, ix_min = np.unravel_index(np.nanargmin(psfc), psfc.shape)
xc_km = float(xh[ix_min])
yc_km = float(yh[iy_min])
print(f"  台风中心: xc={xc_km:.1f} km, yc={yc_km:.1f} km")
print(f"  psfc_min = {psfc[iy_min, ix_min]/100:.1f} hPa")

# ═══════════════════════════════════════════════════════════════
# 2. 方位角平均 (与 refactor _azimuthal_average_by_radius 一致)
# ═══════════════════════════════════════════════════════════════
print("方位角平均...")

X, Y = np.meshgrid(xh, yh)
r2d = np.sqrt((X - xc_km)**2 + (Y - yc_km)**2)

r_bins = np.arange(0.0, MAX_R_KM + DR_KM, DR_KM)
r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
nr = len(r_centers)

# digitize: bin 1..nbins, 0=underflow, nbins=overflow
bin_idx = np.digitize(r2d.ravel(), r_bins) - 1
valid = (bin_idx >= 0) & (bin_idx < nr)

dbz_azim = np.full((nz, nr), np.nan, dtype=np.float64)
for k in range(nz):
    flat = dbz_scalar[k].ravel()
    use = valid & np.isfinite(flat)
    if not np.any(use):
        continue
    idx = bin_idx[use]
    vals = flat[use]
    cnt = np.bincount(idx, minlength=nr)
    sm  = np.bincount(idx, weights=vals, minlength=nr)
    with np.errstate(invalid="ignore", divide="ignore"):
        dbz_azim[k, :] = sm / cnt

# 填充 NaN (内圈可能为空)
dbz_filled = np.where(np.isfinite(dbz_azim), dbz_azim, np.nan)

print(f"  dbz 方位角平均: shape={dbz_filled.shape}")
print(f"  dbz 范围: [{np.nanmin(dbz_filled):.1f}, {np.nanmax(dbz_filled):.1f}] dBZ")
print(f"  有效点数: {np.sum(np.isfinite(dbz_filled))} / {dbz_filled.size}")

# ═══════════════════════════════════════════════════════════════
# 3. 绘图
# ═══════════════════════════════════════════════════════════════
print("绘图...")

R, Z = np.meshgrid(r_centers, zh_sel)

fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

# dbz 填充色
lv = np.arange(0, 61, 5)
im = ax.contourf(R, Z, dbz_filled, levels=lv, cmap="Spectral_r",
                 extend="both", vmin=0, vmax=60)

# 叠加 dbz = 35 等值线 (层云/对流分界)
cs = ax.contour(R, Z, dbz_filled, levels=[DBZ_STRATIFORM],
                colors='black', linewidths=2.0, linestyles='--')
ax.clabel(cs, fmt='%d dBZ', fontsize=10, colors='black')

# 填充 dbz < 35 的半透明标记 (层云区域)
# ax.contourf(R, Z, dbz_filled, levels=[0, DBZ_STRATIFORM],
#             colors='none', hatches=['//'], alpha=0.0)

cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
cbar.set_label("dBZ", fontsize=12)

# 计算并标注层云占比
mask_stratiform = (dbz_filled >= 0) & (dbz_filled < DBZ_STRATIFORM)
mask_convective = dbz_filled >= DBZ_STRATIFORM
# 仅统计 z > 10 km 的高层
high_mask = zh_sel[:, None] > 10.0
stratiform_high = np.sum(mask_stratiform & high_mask)
convective_high = np.sum(mask_convective & high_mask)
total_high = stratiform_high + convective_high
if total_high > 0:
    pct_s = stratiform_high / total_high * 100
    print(f"  高层 (z>10km): 层云={stratiform_high}点 ({pct_s:.1f}%), "
          f"对流={convective_high}点 ({100-pct_s:.1f}%)")

ax.set_xlabel("Radius (km)", fontsize=12)
ax.set_ylabel("Height (km)", fontsize=12)
ax.set_title(
    f"Azimuthal-Mean Radar Reflectivity — Thompson t={TARGET_HOURS:.0f}h\n"
    f"Dashed line: {DBZ_STRATIFORM:.0f} dBZ (stratiform/convective boundary)",
    fontsize=13, fontweight="bold",
)
ax.set_xlim(0, MAX_R_KM)
ax.set_ylim(0, MAX_Z_KM)
ax.set_box_aspect(0.75)
ax.grid(True, alpha=0.2, linestyle="--")

fig.tight_layout()
out_path = OUTPUT_DIR / "dbz_azimuthal_mean.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"[保存] {out_path}")
print("完成。")

ds.close()
