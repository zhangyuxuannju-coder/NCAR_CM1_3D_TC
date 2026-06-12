"""
Dipole sensitivity analysis:
  z < 12.5 km: original ptb_mp
  z > 12.5 km: dipole heating/cooling (WEAK/MODERATE/STRONG)
Compare: CTRL vs three dipole experiments
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent / "exp_dipole"
CTRL_PATH = Path(__file__).resolve().parent.parent.parent / "exp_results/exp_ctrl"

d_ctrl = np.load(CTRL_PATH / "se_pipeline_products.npz")
r = d_ctrl["r_km"]
z = d_ctrl["z_km"]
ctrl = {"U_se": d_ctrl["U_se"][:, 1:-1].T, "Q": d_ctrl["Q"]}

exps = [
    ("weak",    "DIPOLE_WEAK (|Qpk|~1.4e-4)"),
    ("moderate","DIPOLE_MODERATE (|Qpk|~4.2e-4)"),
    ("strong",  "DIPOLE_STRONG (|Qpk|~8.3e-4)"),
]
q0_vals = [-5e-4, -5e-4, -5e-4]
q_factors = [1.0, 3.0, 6.0]
peak_vals = [1.38e-4, 4.15e-4, 8.30e-4]

data = {}
for folder, label in exps:
    d = np.load(BASE / folder / "se_pipeline_products.npz")
    data[folder] = {"label": label, "U_se": d["U_se"][:, 1:-1].T, "Q": d["Q"]}

# ====== 1. Q comparison ======
fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)
q_list = [("CTRL", ctrl["Q"])] + [(data[f]["label"], data[f]["Q"]) for f, _ in exps]
for ax, (ttl, qq) in zip(axes, q_list):
    vmin, vmax = float(np.nanmin(qq)), float(np.nanmax(qq))
    lp = np.linspace(0, vmax, 16)[1:] if vmax > 0 else []
    ln = np.linspace(vmin, 0, 16)[:-1] if vmin < 0 else []
    lv = np.unique(np.concatenate([ln, lp]))
    if len(lv) < 2: lv = np.linspace(vmin, vmax, 31)
    nrm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax) if (vmin<0 and vmax>0) else None
    im = ax.contourf(r, z, qq, levels=lv, cmap="RdBu_r", norm=nrm, extend="both")
    ax.contour(r, z, qq, levels=lv[::2], colors='k', alpha=0.4, linewidths=0.5)
    ax.axhline(y=12.5, color='green', linestyle='--', linewidth=1.5)
    ax.set_title(ttl, fontsize=11, fontweight='bold')
    ax.set_xlabel("Radius (km)"); ax.set_ylabel("Height (km)")
    ax.set_ylim(0, 20); ax.set_box_aspect(3/4)
    plt.colorbar(im, ax=ax, fraction=0.046, label="Q (K/s)")
fig.savefig(BASE / "dipole_Q.png", dpi=180); plt.close(fig)
print("Saved: dipole_Q.png")

# ====== 2. U_se panels + delta ======
fig, axes = plt.subplots(2, 3, figsize=(22, 13), constrained_layout=True)
zm = z >= 8.0
ua = [ctrl["U_se"][zm, :]] + [data[f]["U_se"][zm, :] for f, _ in exps]
vm = max(abs(np.nanmin(np.concatenate(ua))), abs(np.nanmax(np.concatenate(ua))), 0.5)

u_list = ["CTRL"] + [data[f]["label"] for f, _ in exps]
u_data = [ctrl["U_se"]] + [data[f]["U_se"] for f, _ in exps]
for ax, ttl, ff in zip(axes[0], u_list, u_data):
    lv = np.linspace(-vm, vm, 31)
    im = ax.contourf(r, z, ff, levels=lv, cmap="RdBu_r", extend="both")
    ax.contour(r, z, ff, levels=lv[::3], colors='k', alpha=0.4, linewidths=0.5)
    ax.axhline(y=12.5, color='green', linestyle='--', linewidth=1.5)
    ax.set_title(ttl, fontsize=11, fontweight='bold')
    ax.set_xlabel("Radius (km)"); ax.set_ylabel("Height (km)")
    ax.set_ylim(0, 20); ax.set_box_aspect(3/4)
    plt.colorbar(im, ax=ax, fraction=0.046, label="U_se (m/s)")

for ax, (folder, label), pk in zip(axes[1], exps, peak_vals):
    diff = data[folder]["U_se"] - ctrl["U_se"]
    vd = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)), 0.05)
    ld = np.linspace(-vd, vd, 31)
    nrm = TwoSlopeNorm(vcenter=0, vmin=-vd, vmax=vd)
    im = ax.contourf(r, z, diff, levels=ld, cmap="RdBu_r", norm=nrm, extend="both")
    ax.contour(r, z, diff, levels=ld[::3], colors='k', alpha=0.4, linewidths=0.5)
    ax.axhline(y=12.5, color='green', linestyle='--', linewidth=1.5)
    ax.fill_between([40, 250], 10.0, 20.0, alpha=0.06, color='yellow')
    ax.set_title(f"Delta U_se: {label.split('(')[1].rstrip(')')} - CTRL", fontsize=11, fontweight='bold')
    ax.set_xlabel("Radius (km)"); ax.set_ylabel("Height (km)")
    ax.set_ylim(0, 20); ax.set_box_aspect(3/4)
    plt.colorbar(im, ax=ax, fraction=0.046, label="Delta U_se (m/s)")
fig.savefig(BASE / "dipole_U_se.png", dpi=180); plt.close(fig)
print("Saved: dipole_U_se.png")

# ====== 3. Profiles z~11 & z~15 km ======
fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)
colors = ['black', 'blue', 'orange', 'red']
lss = ['-', '--', '-.', ':']
for idx, (zt, ttl) in enumerate([(11.0, "Inflow layer (~11 km)"), (15.0, "Outflow layer (~15 km)")]):
    ax = axes[idx]; iz = np.argmin(np.abs(z - zt))
    ax.plot(r, ctrl["U_se"][iz, :], '-', color='black', label='CTRL', linewidth=2.0)
    for (folder, label), c, ls in zip(exps, colors[1:], lss[1:]):
        ax.plot(r, data[folder]["U_se"][iz, :], ls, color=c, label=label, linewidth=1.5)
    ax.axhline(y=0, color='grey', linestyle=':')
    ax.set_xlabel("Radius (km)", fontsize=12)
    ax.set_ylabel(f"U_se at z~{z[iz]:.2f} km (m/s)", fontsize=12)
    ax.set_title(ttl, fontsize=14, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25); ax.set_xlim(0, 300)
fig.savefig(BASE / "dipole_profiles.png", dpi=180); plt.close(fig)
print("Saved: dipole_profiles.png")

# ====== 4. Pure artificial dipole Q field ======
rc, zc = 145.0, 15.0
sr, sz = 105.0, 1.5  # 偶极子垂直半轴
R, Z = np.meshgrid(r, z)

fig, axes = plt.subplots(1, 3, figsize=(22, 6.5), constrained_layout=True)

for ax, (folder, label), q0, qf, pk in zip(axes, exps, q0_vals, q_factors, peak_vals):
    # 修正后的偶极子公式: Q = -Q0 * q_factor * (z-zc)/sz² * exp(-dist²)
    # 负号使 Q0<0 时: 上部(z>zc)→加热(正), 下部(z<zc)→冷却(负)
    dist_sq = ((R - rc) / sr)**2 + ((Z - zc) / sz)**2
    q_dipole = -q0 * qf * (Z - zc) / (sz**2) * np.exp(-dist_sq)
    
    vd = max(abs(q_dipole.min()), abs(q_dipole.max()), 1e-10)
    lv = np.linspace(-vd, vd, 31)
    nrm = TwoSlopeNorm(vcenter=0, vmin=-vd, vmax=vd)
    im = ax.contourf(r, z, q_dipole, levels=lv, cmap="RdBu_r", norm=nrm, extend="both")
    ax.contour(r, z, q_dipole, levels=lv[::3], colors='k', alpha=0.4, linewidths=0.5)
    ax.axhline(y=zc, color='grey', linestyle=':', alpha=0.6)
    ax.set_title(f"Q0={q0*qf:.1e}  |Qpk|={pk:.1e} K/s\nsigma_z={sz:.1f} km", fontsize=13, fontweight='bold')
    ax.set_xlabel("Radius (km)"); ax.set_ylabel("Height (km)")
    ax.set_ylim(10, 20); ax.set_box_aspect(3/4)
    plt.colorbar(im, ax=ax, fraction=0.046, label="Q_dipole (K/s)")

fig.savefig(BASE / "dipole_artificial_Q.png", dpi=180); plt.close(fig)
print("Saved: dipole_artificial_Q.png")

# ====== 4. Summary ======
iz11 = np.argmin(np.abs(z - 11.0))
iz15 = np.argmin(np.abs(z - 15.0))
ir150 = np.argmin(np.abs(r - 150))
print(f"\n=== Dipole Sensitivity Summary (sigma_z=1.5km, corrected sign) ===")
print(f"{'Exp':<20} {'|Qpk| (K/s)':<16} {'U_se@150km(z11)':<18} {'U_se@150km(z15)':<18}")
print(f"{'CTRL':<20} {'N/A':<16} {ctrl['U_se'][iz11,ir150]:<18.4f} {ctrl['U_se'][iz15,ir150]:<18.4f}")
for (folder, label), pk in zip(exps, peak_vals):
    u11 = data[folder]["U_se"][iz11, ir150]
    u15 = data[folder]["U_se"][iz15, ir150]
    print(f"{folder.upper():<20} {pk:<16.2e} {u11:<18.4f} {u15:<18.4f}")
print("Done.")
