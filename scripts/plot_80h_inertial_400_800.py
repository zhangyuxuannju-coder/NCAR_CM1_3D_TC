"""Plot the existing axis-regularized full-radius K3 response over 400-800 km."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path.cwd()))
from scripts import plot_inertial_operator_evolution_matched as old

ROOT = Path("output/se_axis_regularized_80h_full_inertial_total_new01")
OUT = Path("output/se_axis_regularized_80h_inertial_400_800_new01")
OUT.mkdir(exist_ok=False)

with np.load(ROOT / "window_080.npz") as data:
    m = {key: data[key] for key in data.files}
r, z = m["r_km"], m["z_km"]
dr = np.arange(400.0, 800.001, 2.0)
dz = np.arange(z[0], 18.001, 0.1)
R, Z = np.meshgrid(dr, dz)
points = np.c_[Z.ravel(), R.ravel()]
def display(field):
    return RegularGridInterpolator((z, r), field, method="linear",
                                   bounds_error=True)(points).reshape(Z.shape)

rhs = display(m["inertial_rhs"])
u = display(m["inertial_u"])
w = display(m["inertial_w"])
jet_u = display(m["jet_u"])
rhs_limit = max(float(np.max(np.abs(rhs))), 1e-30)
u_limit = max(float(np.max(np.abs(u))), 1e-12)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharex=True, sharey=True,
                         constrained_layout=True)
im = axes[0].contourf(dr, dz, rhs, levels=np.linspace(-rhs_limit, rhs_limit, 31),
                      cmap="RdBu_r", extend="both")
old.contour_radial_wind(axes[0], dr, dz, jet_u)
fig.colorbar(im, ax=axes[0], label="Equivalent RHS (K$^{-1}$ s$^{-3}$)")
axes[0].set_title("Full-radius inertial-operator equivalent RHS")

im = axes[1].contourf(dr, dz, u, levels=np.linspace(-u_limit, u_limit, 31),
                      cmap="RdBu_r", extend="both")
old.contour_radial_wind(axes[1], dr, dz, jet_u)
iz = np.arange(0, len(dz), 20); ir = np.arange(0, len(dr), 15)
axes[1].quiver(dr[ir], dz[iz], u[np.ix_(iz, ir)], w[np.ix_(iz, ir)],
               color="0.3", width=.002)
fig.colorbar(im, ax=axes[1], label="SE radial-wind response (m s$^{-1}$; outward +)")
axes[1].set_title("Radial/vertical circulation response")

for ax in axes:
    ax.set(xlim=(400, 800), ylim=(0, 18), xlabel="Radius (km)")
axes[0].set_ylabel("Height (km)")
fig.suptitle("80 h +/-3 h | 400-800 km zoom | black: JET SE radial wind\n"
             "inner-axis psi=0; full-radius regularized delta K3; regional auto scales")
fig.savefig(OUT / "080_inertial_operator_400_800km.png", dpi=200)
plt.close(fig)

summary = {
    "source": str(ROOT / "window_080.npz"),
    "new_SE_solves": 0,
    "radius_km": [400, 800], "height_km": [float(z[0]), 18],
    "display_interpolation": "linear 2 km radial x 0.1 km vertical; no extrapolation",
    "rhs_colour_limit": rhs_limit, "response_colour_limit_ms": u_limit,
    "colour_note": "regional maximum-absolute scales for structure visibility; do not compare amplitudes to the 0-500 km shared-scale panel from colour alone",
    "background": "JET SE solved radial-wind contours",
    "definition": "full-radius regularized delta K3 equivalent forcing and CTRL-operator balanced response; no 100-km mask",
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
