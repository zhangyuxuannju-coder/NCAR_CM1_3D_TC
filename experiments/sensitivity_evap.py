"""
Evap cooling sensitivity analysis: WEAK vs MODERATE vs STRONG
Focus: U_se at z~11 km (inflow layer below outflow)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent / "exp_results"

# Load all four experiments
data = {}
labels_map = [
    ("exp_ctrl",        "CTRL (full ptb_mp)"),
    ("exp_evap_weak",   "EVAP_WEAK  (Q0=-5e-5)"),
    ("exp_evap_only",   "EVAP_MODERATE (Q0=-2e-4)"),
    ("exp_evap_strong", "EVAP_STRONG (Q0=-5e-4)"),
]

for folder, label in labels_map:
    d = np.load(BASE / folder / "se_pipeline_products.npz")
    data[folder] = {
        "label": label,
        "r_km": d["r_km"],
        "z_km": d["z_km"],
        "U_se": d["U_se"][:, 1:-1].T,
        "W_se": d["W_se"][:, 1:-1].T,
        "psi": d["psi"][:, 1:-1].T,
        "Q": d["Q"],
    }

r = data["exp_ctrl"]["r_km"]
z = data["exp_ctrl"]["z_km"]

# ====== 1. Inflow-layer profile: U_se at z~11 km ======
z_target = 11.0
iz = np.argmin(np.abs(z - z_target))

fig, ax = plt.subplots(figsize=(12, 5.5))
colors = ['black', 'blue', 'orange', 'red']
linestyles = ['-', '--', '-.', ':']

for (folder, label), c, ls in zip(labels_map, colors, linestyles):
    u_prof = data[folder]["U_se"][iz, :]
    lw = 2.0 if folder == "exp_ctrl" else 1.5
    ax.plot(r, u_prof, ls, color=c, label=label, linewidth=lw)

ax.axvspan(40, 250, alpha=0.08, color='yellow')
ax.axhline(y=0, color='grey', linestyle=':')
ax.set_xlabel("Radius (km)", fontsize=13)
ax.set_ylabel(f"U_se at z~{z[iz]:.2f} km (m/s)", fontsize=13)
ax.set_title("Inflow-layer Radial Wind: Evap Cooling Sensitivity", fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.25)
ax.set_xlim(0, 300)
fig.tight_layout()
fig.savefig(BASE / "sensitivity_U_se_profile_z11km.png", dpi=180)
plt.close(fig)
print("Saved: sensitivity_U_se_profile_z11km.png")


# ====== 2. Sensitivity curve: U_se at specific radii vs Q0 ======
q0_vals = np.array([-5e-5, -2e-4, -5e-4])
labels_q0 = ["-5e-5", "-2e-4", "-5e-4"]
folders_evap = ["exp_evap_weak", "exp_evap_only", "exp_evap_strong"]
radii_check = [60, 100, 150, 200]  # km

fig, ax = plt.subplots(figsize=(8, 5))
for r_target in radii_check:
    ir = np.argmin(np.abs(r - r_target))
    u_vals = [data[f]["U_se"][iz, ir] for f in folders_evap]
    ax.plot(q0_vals * 1e4, u_vals, 'o-', label=f"r={r_target} km", linewidth=1.5)

ax.axhline(y=0, color='grey', linestyle=':')
ax.set_xlabel("Evap Cooling Q0 (x1e-4 K/s)", fontsize=13)
ax.set_ylabel(f"U_se at z~{z[iz]:.2f} km (m/s)", fontsize=13)
ax.set_title("U_se Response to Evap Cooling Intensity", fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(BASE / "sensitivity_U_se_vs_Q0.png", dpi=180)
plt.close(fig)
print("Saved: sensitivity_U_se_vs_Q0.png")


# ====== 3. Delta U_se panels: EVAP - CTRL for WEAK/MODERATE/STRONG ======
fig, axes = plt.subplots(1, 3, figsize=(22, 7), constrained_layout=True)

for ax, (folder, label), q0_text in zip(
    axes,
    [labels_map[1], labels_map[2], labels_map[3]],
    ["Q0=-5e-5", "Q0=-2e-4", "Q0=-5e-4"]
):
    diff = data[folder]["U_se"] - data["exp_ctrl"]["U_se"]
    vmax_diff = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)), 0.1)
    levels_diff = np.linspace(-vmax_diff, vmax_diff, 31)
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax_diff, vmax=vmax_diff)
    im = ax.contourf(r, z, diff, levels=levels_diff, cmap="RdBu_r", norm=norm, extend="both")
    ax.contour(r, z, diff, levels=levels_diff[::3], colors='k', alpha=0.4, linewidths=0.5)
    ax.axhline(y=10.0, color='grey', linestyle=':', alpha=0.5)
    ax.axhline(y=20.0, color='grey', linestyle=':', alpha=0.5)
    ax.fill_between([40, 250], 10.0, 20.0, alpha=0.06, color='yellow')
    ax.set_title(f"Delta U_se: {q0_text} - CTRL", fontsize=14, fontweight='bold')
    ax.set_xlabel("Radius (km)")
    ax.set_ylabel("Height (km)")
    ax.set_ylim(0, 20)
    ax.set_box_aspect(3/4)
    plt.colorbar(im, ax=ax, fraction=0.046, label="Delta U_se (m/s)")

fig.savefig(BASE / "sensitivity_delta_U_se_panel.png", dpi=180)
plt.close(fig)
print("Saved: sensitivity_delta_U_se_panel.png")


# ====== 4. Summary table print ======
print("\n=== Sensitivity Summary (z~{:.2f} km) ===".format(z[iz]))
print(f"{'Exp':<20} {'Q0 (K/s)':<14} {'U_se_100km':<14} {'U_se_150km':<14} {'U_se_200km':<14}")
u_ctrl_100 = data["exp_ctrl"]["U_se"][iz, np.argmin(np.abs(r-100))]
u_ctrl_150 = data["exp_ctrl"]["U_se"][iz, np.argmin(np.abs(r-150))]
u_ctrl_200 = data["exp_ctrl"]["U_se"][iz, np.argmin(np.abs(r-200))]
print(f"{'CTRL':<20} {'N/A':<14} {u_ctrl_100:<14.4f} {u_ctrl_150:<14.4f} {u_ctrl_200:<14.4f}")
for folder, label, q0 in zip(folders_evap, ["WEAK", "MODERATE", "STRONG"], q0_vals):
    u_100 = data[folder]["U_se"][iz, np.argmin(np.abs(r-100))]
    u_150 = data[folder]["U_se"][iz, np.argmin(np.abs(r-150))]
    u_200 = data[folder]["U_se"][iz, np.argmin(np.abs(r-200))]
    print(f"{label:<20} {q0:<14.1e} {u_100:<14.4f} {u_150:<14.4f} {u_200:<14.4f}")

print("\n=== Sensitivity analysis complete ===")
