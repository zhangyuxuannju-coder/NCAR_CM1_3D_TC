"""80 h full-radius K3 response and complete JET-minus-CTRL SE response."""
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

OUT = Path("output/se_axis_regularized_80h_full_inertial_total_new01")
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
    states = [state("CTRL", hour), state("U30", hour)]
    r = archived["r_km"] * 1000.0
    z = archived["z_km"] * 1000.0
    operators = []; basics = []; k_regularized = []
    for s in states:
        theta, _ = invert_balanced_theta(s["ut"], s["theta"], r, z, args.f,
                                          outer_smooth_window=1)
        basic = build_basic_state(s["ut"], theta, s["rho"], r, z, args.f)
        _, operator, _ = old.build_ctrl_operator(basic, r, z, args)
        operators.append(operator); basics.append(basic)
        k_regularized.append(regularize_ellipticity(
            basic["K1_raw"], basic["K2_raw"], basic["K3_raw"],
            eps_ratio=args.eps_ratio, margin=0.0)[:3])

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
    zero = np.zeros_like(states[0]["rho"])
    delta_k3 = k_regularized[1][2] - k_regularized[0][2]
    inertial_matrix = matrix(
        assemble_operator(basics[0], zero, zero, delta_k3, r, z),
        r, z, states[0]["rho"])
    ctrl_vector = rec["ctrl_psi"].T.ravel()
    full_inertial_rhs = (-(inertial_matrix @ ctrl_vector)).reshape(rec["ctrl_psi"].T.shape).T
    for label in ["thermal", "momentum", "inertial"]:
        rhs = full_inertial_rhs if label == "inertial" else archived[label + "_rhs"]
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
    for key in ["psi", "u", "w"]:
        rec["total_" + key] = rec["jet_" + key] - rec["ctrl_" + key]
    rec["total_rhs"] = (mc @ rec["total_psi"].T.ravel()).reshape(rec["total_psi"].T.shape).T
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

# Restore the prior response colour range: symmetric native-grid P5/P95 of actual JET-CTRL ur.
q5, q95 = np.percentile(mean["delta_ur"][native_view], [5, 95])
response_limit = max(abs(float(q5)), abs(float(q95)), 1e-6)

fig, axes = plt.subplots(2, 4, figsize=(21, 9), sharex=True, sharey=True)
labels = ["thermal", "momentum", "inertial", "total"]
for col, label in enumerate(labels):
    rhs = mean[label + "_rhs"]
    rhs_plot = display(rhs)
    rhs_limit = max(float(np.max(np.abs(rhs_plot))), 1e-30)
    im = axes[0, col].contourf(dr, dz, rhs_plot,
                               levels=np.linspace(-rhs_limit, rhs_limit, 31),
                               cmap="RdBu_r", extend="both")
    old.contour_radial_wind(axes[0, col], dr, dz, display(mean["jet_u"]))
    axes[0, col].set_title(("L_CTRL(psi_JET-psi_CTRL) equivalent RHS" if label == "total"
                            else label + " difference/equivalent RHS") + " (K^-1 s^-3)")
    fig.colorbar(im, ax=axes[0, col], shrink=.75)

    u = display(mean[label + "_u"])
    w = display(mean[label + "_w"])
    im = axes[1, col].contourf(dr, dz, u,
                               levels=np.linspace(-response_limit, response_limit, 31),
                               cmap="RdBu_r", extend="both")
    old.contour_radial_wind(axes[1, col], dr, dz, display(mean["jet_u"]))
    iz = np.arange(0, len(dz), 20); ir = np.arange(0, len(dr), 18)
    axes[1, col].quiver(dr[ir], dz[iz], u[np.ix_(iz, ir)], w[np.ix_(iz, ir)],
                        color="0.35", width=.002)
    axes[1, col].set_title(("complete JET SE - CTRL SE" if label == "total" else label + " response") +
                           " ur (m/s), arrows ur/w")
    fig.colorbar(im, ax=axes[1, col], shrink=.75)
for ax in axes.ravel():
    ax.set(xlim=(0, 500), ylim=(0, 18), xlabel="Radius (km)")
axes[0, 0].set_ylabel("Height (km)"); axes[1, 0].set_ylabel("Height (km)")
fig.suptitle("80 h +/-3 h response mean | black: JET SE solved ur | full solve 1200 km, view 500 km\n"
             "inner-axis psi=0; full-radius K3 operator response (no 100-km forcing mask)")
fig.tight_layout()
fig.savefig(OUT / "080_full_radius_inertial_and_total_with_jet_se_ur.png", dpi=190)
plt.close(fig)

summary = {
    "source_actual_and_rhs": str(BASE),
    "window": "80 h +/-3 h; solve each hour then average",
    "boundary": "inner_axis_dirichlet=True for CTRL, JET, thermal, momentum, inertial; outer/top Neumann; bottom ghost zero",
    "solve_radius_km": 1200, "view_radius_km": 500,
    "display_interpolation": "linear 2 km radial x 0.1 km vertical; no extrapolation",
    "response_colour_rule": "restored prior rule: symmetric max(abs(P5),abs(P95)) of native actual JET-CTRL ur in displayed domain",
    "actual_delta_ur_p5_p95_ms": [float(q5), float(q95)],
    "response_colour_limit_ms": response_limit,
    "rhs_colour_rule": "per-field maximum absolute interpolated RHS in displayed domain",
    "inertial_definition": "full-radius regularized delta K3 operator equivalent RHS, -deltaL_K3 psi_CTRL; no smoothing and no 100-km mask",
    "total_definition": "bottom: direct complete JET SE minus CTRL SE radial/vertical wind; top: L_CTRL applied to the complete streamfunction difference",
    "background_contours": "JET SE solved radial wind, not CM1 actual wind",
    "max_component_relative_residual": max(row["relative_residual"] for row in checks),
    "max_first_column_abs_psi": max(row["first_column_max_abs_psi"] for row in checks),
    "max_first_column_abs_u": max(row["first_column_max_abs_u"] for row in checks),
    "interpretation": "regularized balanced SE projection; not actual wind difference or unique causality",
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
