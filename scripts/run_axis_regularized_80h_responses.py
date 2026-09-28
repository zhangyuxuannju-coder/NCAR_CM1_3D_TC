"""Re-solve the 80 h response panel with the validated inner-axis option."""
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

OUT = Path("output/se_axis_regularized_80h_responses_2x_extreme_new03")
OUT.mkdir(exist_ok=False)
BASE = Path("output/se_boundary_fixed_baseline_new01")
INERTIAL_RHS_SOURCE = Path("output/same_time_se_window3h_50_150_new01")

original_sparse = old.solve.__globals__["solve_se_sparse"]
def axis_sparse(*args, **kwargs):
    kwargs["inner_axis_dirichlet"] = True
    return original_sparse(*args, **kwargs)
old.solve.__globals__["solve_se_sparse"] = axis_sparse

records = []
checks = []
for hour in range(77, 84):
    archived = dict(np.load(BASE / f"hourly_{hour:03d}.npz"))
    inertial_archived = dict(np.load(INERTIAL_RHS_SOURCE / f"hourly_{hour:03d}.npz"))
    states = [state("CTRL", hour), state("U30", hour)]
    r = archived["r_km"] * 1000.0
    z = archived["z_km"] * 1000.0
    operators = []
    for s in states:
        theta, _ = invert_balanced_theta(s["ut"], s["theta"], r, z, args.f,
                                          outer_smooth_window=1)
        basic = build_basic_state(s["ut"], theta, s["rho"], r, z, args.f)
        _, operator, _ = old.build_ctrl_operator(basic, r, z, args)
        operators.append(operator)

    rec = {"r_km": archived["r_km"], "z_km": archived["z_km"],
           "jet_ur": archived["jet_ur"], "delta_ur": archived["delta_ur"]}
    # Full CTRL/JET solutions and every response use the same axis boundary.
    for label, operator, s in zip(["ctrl", "jet"], operators, states):
        rhs = archived[label + "_thermal_rhs"] + archived[label + "_momentum_rhs"]
        solved = old.solve(operator, rhs, s["rho"], r, z)
        for key in ["psi", "u", "w"]:
            rec[label + "_" + key] = solved[key]

    mc = matrix(operators[0], r, z, states[0]["rho"])
    fac = spla.factorized(mc.tocsc())
    for label in ["thermal", "momentum", "inertial"]:
        rhs = (inertial_archived if label == "inertial" else archived)[label + "_rhs"]
        spla.spsolve = lambda m, b, **kwargs: fac(b)
        try:
            solved = old.solve(operators[0], rhs, states[0]["rho"], r, z)
        finally:
            spla.spsolve = original
        rec[label + "_rhs"] = rhs
        for key in ["psi", "u", "w"]:
            rec[label + "_" + key] = solved[key]
        solver_rhs = rhs.T.copy(); solver_rhs[0, :] = 0.0
        checks.append({
            "hour": hour, "component": label,
            "relative_residual": rel(mc @ solved["psi"].T.ravel() - solver_rhs.ravel(),
                                     solver_rhs.ravel()),
            "first_column_max_abs_psi": float(np.max(np.abs(solved["psi"][:, 0]))),
            "first_column_max_abs_u": float(np.max(np.abs(solved["u"][:, 0]))),
        })
    np.savez_compressed(OUT / f"hourly_{hour:03d}.npz", **rec)
    records.append(rec)

mean = {key: np.mean([record[key] for record in records], axis=0) for key in records[0]}
np.savez_compressed(OUT / "window_080.npz", **mean, hours=np.arange(77, 84))
with (OUT / "checks.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=checks[0].keys())
    writer.writeheader(); writer.writerows(checks)

r, z = mean["r_km"], mean["z_km"]
rr, zz = np.meshgrid(r, z)
native_view = (rr <= 500) & (zz <= 18)
dr = np.arange(r[0], 500.001, 2.0)
dz = np.arange(z[0], 18.001, 0.1)
R, Z = np.meshgrid(dr, dz)
points = np.c_[Z.ravel(), R.ravel()]
def display(field):
    return RegularGridInterpolator((z, r), field, method="linear", bounds_error=True)(points).reshape(Z.shape)

# Requested rule: twice the maximum absolute interpolated JET-CTRL radial-wind difference.
interpolated_wind_extreme = float(np.max(np.abs(display(mean["delta_ur"]))))
response_limit = 2.0 * max(interpolated_wind_extreme, 1e-6)

fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
for col, label in enumerate(["thermal", "momentum", "inertial"]):
    rhs = mean[label + "_rhs"]
    rhs_plot = display(rhs)
    rhs_limit = max(float(np.max(np.abs(rhs_plot))), 1e-30)
    im = axes[0, col].contourf(dr, dz, rhs_plot,
                               levels=np.linspace(-rhs_limit, rhs_limit, 31),
                               cmap="RdBu_r", extend="both")
    old.contour_radial_wind(axes[0, col], dr, dz, display(mean["jet_ur"]))
    axes[0, col].set_title(label + " difference/equivalent RHS (K^-1 s^-3)")
    fig.colorbar(im, ax=axes[0, col], shrink=.75)

    u = display(mean[label + "_u"])
    w = display(mean[label + "_w"])
    im = axes[1, col].contourf(dr, dz, u,
                               levels=np.linspace(-response_limit, response_limit, 31),
                               cmap="RdBu_r", extend="both")
    old.contour_radial_wind(axes[1, col], dr, dz, display(mean["jet_ur"]))
    iz = np.arange(0, len(dz), 20); ir = np.arange(0, len(dr), 18)
    axes[1, col].quiver(dr[ir], dz[iz], u[np.ix_(iz, ir)], w[np.ix_(iz, ir)],
                        color="0.35", width=.002)
    axes[1, col].set_title(label + " response ur (m/s), arrows ur/w")
    fig.colorbar(im, ax=axes[1, col], shrink=.75)
    if label == "inertial":
        axes[0, col].axvline(100, color="purple", ls="--", lw=.8)
        axes[1, col].axvline(100, color="purple", ls="--", lw=.8)
for ax in axes.ravel():
    ax.set(xlim=(0, 500), ylim=(0, 18), xlabel="Radius (km)")
axes[0, 0].set_ylabel("Height (km)"); axes[1, 0].set_ylabel("Height (km)")
fig.suptitle("80 h +/-3 h response mean | black: jet_ur | full solve 1200 km, view 500 km\n"
             "inner-axis psi=0 sensitivity; response colour range = +/-2 max|interpolated JET-CTRL ur|")
fig.tight_layout()
fig.savefig(OUT / "080_responses_jet_ur_axis_regularized_2x_extreme.png", dpi=200)
plt.close(fig)

summary = {
    "source_actual_and_rhs": str(BASE),
    "inertial_rhs_source": str(INERTIAL_RHS_SOURCE) + " (RHS only; archived psi/u/w not reused)",
    "window": "80 h +/-3 h; solve each hour then average",
    "boundary": "inner_axis_dirichlet=True for CTRL, JET, thermal, momentum, inertial; outer/top Neumann; bottom ghost zero",
    "solve_radius_km": 1200, "view_radius_km": 500,
    "display_interpolation": "linear 2 km radial x 0.1 km vertical; no extrapolation",
    "response_colour_rule": "+/- 2 times max absolute interpolated JET-CTRL actual radial-wind difference in the displayed domain",
    "interpolated_wind_extreme_ms": interpolated_wind_extreme,
    "response_colour_limit_ms": response_limit,
    "rhs_colour_rule": "unchanged original intent: per-field maximum absolute interpolated RHS in displayed domain",
    "max_component_relative_residual": max(row["relative_residual"] for row in checks),
    "max_first_column_abs_psi": max(row["first_column_max_abs_psi"] for row in checks),
    "max_first_column_abs_u": max(row["first_column_max_abs_u"] for row in checks),
    "interpretation": "regularized balanced SE projection; not actual wind difference or unique causality",
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
