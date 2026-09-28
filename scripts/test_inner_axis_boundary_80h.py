"""80 h +/-3 h sensitivity test for an inner-node psi=0 axis approximation."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

source = Path("scripts/run_complete_se_attribution.py").read_text()
prefix = source[:source.index("names=['thermal'")].replace(
    "OUT=Path(sys.argv[1]);OUT.mkdir(exist_ok=False)", "OUT=None"
)
exec(compile(prefix, "existing_complete_helpers", "exec"), globals())

OUT = Path("output/se_inner_axis_dirichlet_80h_new03")
OUT.mkdir(exist_ok=False)
BASE = Path("output/se_boundary_fixed_baseline_new01")

original_sparse = old.solve.__globals__["solve_se_sparse"]
def axis_sparse(*args, **kwargs):
    kwargs["inner_axis_dirichlet"] = True
    return original_sparse(*args, **kwargs)
old.solve.__globals__["solve_se_sparse"] = axis_sparse

records = []
checks = []
for hour in range(77, 84):
    archived = dict(np.load(BASE / f"hourly_{hour:03d}.npz"))
    r = archived["r_km"] * 1000.0
    z = archived["z_km"] * 1000.0
    solved = {"r_km": archived["r_km"], "z_km": archived["z_km"],
              "actual_jet_ur": archived["jet_ur"]}
    for label, cache_label in [("ctrl", "CTRL"), ("jet", "U30")]:
        s = state(cache_label, hour)
        theta, _ = invert_balanced_theta(s["ut"], s["theta"], r, z, args.f,
                                          outer_smooth_window=1)
        basic = build_basic_state(s["ut"], theta, s["rho"], r, z, args.f)
        _, operator, _ = old.build_ctrl_operator(basic, r, z, args)
        rhs = archived[label + "_thermal_rhs"] + archived[label + "_momentum_rhs"]
        result = old.solve(operator, rhs, s["rho"], r, z)
        solved[label + "_psi"] = result["psi"]
        solved[label + "_u"] = result["u"]
        solved[label + "_w"] = result["w"]
        m = matrix(operator, r, z, s["rho"])
        rhs_solver = rhs.T.copy()
        rhs_solver[0, :] = 0.0  # the substituted inner Dirichlet matrix rows
        residual = rel(m @ result["psi"].T.ravel() - rhs_solver.ravel(), rhs_solver.ravel())
        top_inner = (archived["z_km"] >= 16)[:, None] & (archived["r_km"] <= 100)[None, :]
        checks.append({
            "hour": hour, "case": label,
            "relative_residual": residual,
            "old_top_inner_max_abs_u": float(np.max(np.abs(archived[label + "_u"][top_inner]))),
            "axis_top_inner_max_abs_u": float(np.max(np.abs(result["u"][top_inner]))),
            "axis_first_column_max_abs_psi": float(np.max(np.abs(result["psi"][:, 0]))),
            "axis_first_column_max_abs_u": float(np.max(np.abs(result["u"][:, 0]))),
        })
    np.savez_compressed(OUT / f"hourly_{hour:03d}.npz", **solved)
    records.append(solved)

axis = {k: np.mean([record[k] for record in records], axis=0) for k in records[0]}
old_mean = dict(np.load(BASE / "window_080.npz"))
np.savez_compressed(OUT / "window_080.npz", **axis,
                    old_jet_psi=old_mean["jet_psi"], old_jet_u=old_mean["jet_u"])

with (OUT / "checks.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=checks[0].keys())
    writer.writeheader(); writer.writerows(checks)

r, z = axis["r_km"], axis["z_km"]
rr, zz = np.meshgrid(r, z)
view = (rr <= 500) & (zz <= 18)
dr = np.arange(r[0], 500.001, 2.0)
dz = np.arange(z[0], 18.001, 0.1)
R, Z = np.meshgrid(dr, dz)
points = np.c_[Z.ravel(), R.ravel()]
def display(field):
    return RegularGridInterpolator((z, r), field, method="linear", bounds_error=True)(points).reshape(Z.shape)

fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True, sharey=True)
rows = [
    ([old_mean["jet_psi"], axis["jet_psi"], axis["jet_psi"] - old_mean["jet_psi"]],
     "psi (10^9 kg/s; no 2pi)", 1e9, "contour"),
    ([old_mean["jet_u"], axis["jet_u"], axis["jet_u"] - old_mean["jet_u"]],
     "ur (m/s; outward +)", 1.0, "filled"),
]
titles = ["Old inner Neumann", "Inner-node psi=0 sensitivity", "Axis sensitivity - old"]
for row, (fields, label, unit, kind) in enumerate(rows):
    limit = max(float(np.max(np.abs(field[view]))) / unit for field in fields[:2])
    levels = np.linspace(-max(limit, 1e-10), max(limit, 1e-10), 31)
    for col, (field, title) in enumerate(zip(fields, titles)):
        plotted = display(field / unit)
        if kind == "contour":
            im = axes[row, col].contour(dr, dz, plotted, levels=levels, cmap="turbo", linewidths=.85)
            axes[row, col].clabel(im, im.levels[::5], fontsize=7, fmt="%.2g")
        else:
            im = axes[row, col].contourf(dr, dz, plotted, levels=levels, cmap="RdBu_r", extend="both")
            actual = display(axis["actual_jet_ur"])
            axes[row, col].contour(dr, dz, actual, levels=[-2, 2, 5, 10], colors="k",
                                   linewidths=.65, linestyles=["--", "-", "-", "-"])
        fig.colorbar(im, ax=axes[row, col], shrink=.8, label=label)
        axes[row, col].set(xlim=(0, 500), ylim=(0, 18), title=title,
                           xlabel="Radius (km)", ylabel="Height (km)")
fig.suptitle("JET 80 h +/-3 h: inner-axis boundary sensitivity\n"
             "Same RHS/operator/regularization; SE solved to 1200 km; display interpolation only")
fig.tight_layout()
fig.savefig(OUT / "jet_080_inner_axis_boundary_comparison.png", dpi=170)
plt.close(fig)

summary = {
    "source": str(BASE), "centre_h": 80, "window": "+/-3 h; solve hourly then average",
    "change": "first stored radial node r=6 km constrained to psi=0; outer/top Neumann and bottom ghost unchanged",
    "status": "sensitivity approximation because the first stored node is r=dr/2, not the true r=0 axis",
    "solve_radius_km": 1200, "native_dr_km": 12,
    "max_relative_residual": max(row["relative_residual"] for row in checks),
    "max_axis_first_column_abs_psi": max(row["axis_first_column_max_abs_psi"] for row in checks),
    "max_axis_first_column_abs_u": max(row["axis_first_column_max_abs_u"] for row in checks),
    "old_jet_top_inner_max_abs_u_window": float(np.max(np.abs(old_mean["jet_u"][view & (zz >= 16) & (rr <= 100)]))),
    "axis_jet_top_inner_max_abs_u_window": float(np.max(np.abs(axis["jet_u"][view & (zz >= 16) & (rr <= 100)]))),
    "core_default_unchanged": True,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
