"""
Controlled experiment comparison: CTRL vs EXP_NOEVAP vs EXP_EVAP_ONLY
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent / "exp_results"

def load_all():
    data = {}
    for name in ["exp_ctrl", "exp_noevap", "exp_evap_only"]:
        d = np.load(BASE / name / "se_pipeline_products.npz")
        data[name] = {
            "r_km": d["r_km"],
            "z_km": d["z_km"],
            "U_se": d["U_se"][:, 1:-1].T,   # (nz, nr)
            "W_se": d["W_se"][:, 1:-1].T,
            "psi": d["psi"][:, 1:-1].T,
            "Q": d["Q"],
            "forcing_total": d["forcing_total"],
            "forcing_thermal": d["forcing_thermal"],
        }
    return data

data = load_all()
r = data["exp_ctrl"]["r_km"]
z = data["exp_ctrl"]["z_km"]

# ====== 1. Panel figure: 3x U_se (top) + 2x diff + summary (bottom) ======
fig, axes = plt.subplots(2, 3, figsize=(22, 13), constrained_layout=True)
plt.rcParams.update({'font.size': 11})

# Unified colorbar range using z>=8km (outflow layer) max abs
z_mask = z >= 8.0
u_all = []
for name in ["exp_ctrl", "exp_noevap", "exp_evap_only"]:
    u_all.append(data[name]["U_se"][z_mask, :])
vmax_abs = max(abs(np.nanmin(np.concatenate(u_all))), abs(np.nanmax(np.concatenate(u_all))), 0.5)

titles_upper = ["CTRL (full ptb_mp)", "EXP_NOEVAP (zero Q in evap region)", "EXP_EVAP_ONLY (artificial evap cooling only)"]
labels = ["exp_ctrl", "exp_noevap", "exp_evap_only"]

for j, (ax, name, ttl) in enumerate(zip(axes[0], labels, titles_upper)):
    ff = data[name]["U_se"]
    levels = np.linspace(-vmax_abs, vmax_abs, 31)
    im = ax.contourf(r, z, ff, levels=levels, cmap="RdBu_r", extend="both")
    ax.contour(r, z, ff, levels=levels[::3], colors='k', alpha=0.4, linewidths=0.5)
    # Overlay psi contours
    psi = data[name]["psi"]
    psilevels = np.linspace(np.nanmin(psi)*0.9, np.nanmax(psi)*1.1, 8)
    ax.contour(r, z, psi, levels=psilevels, colors='k', linewidths=0.8, linestyles='--')
    ax.set_title(ttl, fontweight='bold')
    ax.set_xlabel("Radius (km)")
    ax.set_ylabel("Height (km)")
    ax.set_ylim(0, 20)
    ax.set_box_aspect(3/4)
    plt.colorbar(im, ax=ax, fraction=0.046, label="U_se (m/s)")

# Difference maps
diff_labels = [
    ("exp_noevap", "exp_ctrl", "Delta U_se: NOEVAP - CTRL\n(effect of removing evap cooling)"),
    ("exp_evap_only", "exp_ctrl", "Delta U_se: EVAP_ONLY - CTRL\n(isolated evap cooling response)"),
]

for j, (label_a, label_b, ttl) in enumerate(diff_labels):
    ax = axes[1, j]
    diff = data[label_a]["U_se"] - data[label_b]["U_se"]
    vmax_diff = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)), 0.1)
    levels_diff = np.linspace(-vmax_diff, vmax_diff, 31)
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax_diff, vmax=vmax_diff)
    im = ax.contourf(r, z, diff, levels=levels_diff, cmap="RdBu_r", norm=norm, extend="both")
    ax.contour(r, z, diff, levels=levels_diff[::3], colors='k', alpha=0.4, linewidths=0.5)
    ax.axhline(y=10.0, color='grey', linestyle=':', alpha=0.6)
    ax.axhline(y=20.0, color='grey', linestyle=':', alpha=0.6)
    ax.fill_between([40, 250], 10.0, 20.0, alpha=0.08, color='yellow')
    ax.set_title(ttl, fontweight='bold')
    ax.set_xlabel("Radius (km)")
    ax.set_ylabel("Height (km)")
    ax.set_ylim(0, 20)
    ax.set_box_aspect(3/4)
    plt.colorbar(im, ax=ax, fraction=0.046, label="Delta U_se (m/s)")

# Third column: text summary
ax_text = axes[1, 2]
ax_text.axis('off')
summary_text = (
    "Experiment Summary\n\n"
    f"CTRL: full ptb_mp\n"
    f"  Converged: 2266 iter\n\n"
    f"EXP_NOEVAP: zero Q in\n"
    f"  r=40-250km, z>10km\n"
    f"  Converged: 2266 iter\n\n"
    f"EXP_EVAP_ONLY: artificial\n"
    f"  evap cooling only\n"
    f"  Q0=-2e-4 K/s, elliptical\n"
    f"  Converged: 2270 iter\n\n"
    "========================\n"
    "Key Questions:\n"
    "(1) Does removing evap\n"
    "    cooling weaken the\n"
    "    outflow-layer inflow?\n"
    "    (see Delta NOEVAP)\n"
    "(2) Can pure evap cooling\n"
    "    independently drive\n"
    "    secondary circulation?\n"
    "    (see Delta EVAP_ONLY)\n"
    "========================\n"
    "Yellow box = evap region"
)
ax_text.text(0.05, 0.95, summary_text, transform=ax_text.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.savefig(BASE / "comparison_U_se_panel.png", dpi=180)
plt.close(fig)
print("Saved: comparison_U_se_panel.png")


# ====== 2. Q field comparison ======
fig, axes = plt.subplots(1, 3, figsize=(22, 7), constrained_layout=True)
q_titles = ["CTRL: ptb_mp Q", "EXP_NOEVAP: Q (evap region zeroed)", "EXP_EVAP_ONLY: artificial evap cooling Q"]

for ax, name, ttl in zip(axes, labels, q_titles):
    qq = data[name]["Q"]
    vmin, vmax = float(np.nanmin(qq)), float(np.nanmax(qq))
    levels_pos = np.linspace(0, vmax, 16)[1:] if vmax > 0 else []
    levels_neg = np.linspace(vmin, 0, 16)[:-1] if vmin < 0 else []
    levels = np.unique(np.concatenate([levels_neg, levels_pos]))
    norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax) if (vmin<0 and vmax>0) else None
    im = ax.contourf(r, z, qq, levels=levels, cmap="RdBu_r", norm=norm, extend="both")
    ax.contour(r, z, qq, levels=levels[::2], colors='k', alpha=0.4, linewidths=0.5)
    ax.set_title(ttl)
    ax.set_xlabel("Radius (km)")
    ax.set_ylabel("Height (km)")
    ax.set_ylim(0, 20)
    ax.set_box_aspect(3/4)
    plt.colorbar(im, ax=ax, fraction=0.046, label="Q (K/s)")

fig.savefig(BASE / "comparison_Q_panel.png", dpi=180)
plt.close(fig)
print("Saved: comparison_Q_panel.png")


# ====== 3. Radial profile: U_se at z~15 km (outflow layer) ======
z_outflow = 15.0
iz_out = np.argmin(np.abs(z - z_outflow))

fig, ax = plt.subplots(figsize=(10, 5))
for name, label, ls in [("exp_ctrl", "CTRL", "-"),
                          ("exp_noevap", "EXP_NOEVAP", "--"),
                          ("exp_evap_only", "EXP_EVAP_ONLY", ":")]:
    u_prof = data[name]["U_se"][iz_out, :]
    ax.plot(r, u_prof, ls, label=label, linewidth=1.5)

ax.axvspan(40, 250, alpha=0.1, color='yellow', label='Evap cooling region')
ax.axhline(y=0, color='grey', linestyle=':')
ax.set_xlabel("Radius (km)")
ax.set_ylabel(f"U_se at z~{z[iz_out]:.1f} km (m/s)")
ax.set_title("Outflow-layer radial wind profile")
ax.legend()
ax.grid(True, alpha=0.3)
fig.savefig(BASE / "comparison_U_se_profile_z15km.png", dpi=160)
plt.close(fig)
print("Saved: comparison_U_se_profile_z15km.png")


# ====== 4. Radial profile: U_se at z~11 km (inflow layer below outflow) ======
z_inflow = 11.0
iz_in = np.argmin(np.abs(z - z_inflow))

fig, ax = plt.subplots(figsize=(10, 5))
for name, label, ls in [("exp_ctrl", "CTRL", "-"),
                          ("exp_noevap", "EXP_NOEVAP", "--"),
                          ("exp_evap_only", "EXP_EVAP_ONLY", ":")]:
    u_prof = data[name]["U_se"][iz_in, :]
    ax.plot(r, u_prof, ls, label=label, linewidth=1.5)

ax.axvspan(40, 250, alpha=0.1, color='yellow', label='Evap cooling region')
ax.axhline(y=0, color='grey', linestyle=':')
ax.set_xlabel("Radius (km)")
ax.set_ylabel(f"U_se at z~{z[iz_in]:.1f} km (m/s)")
ax.set_title("Inflow-layer radial wind profile (below outflow)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.savefig(BASE / "comparison_U_se_profile_z11km.png", dpi=160)
plt.close(fig)
print("Saved: comparison_U_se_profile_z11km.png")

print("\n=== Comparison analysis complete ===")
print(f"All figures saved in: {BASE}/")
