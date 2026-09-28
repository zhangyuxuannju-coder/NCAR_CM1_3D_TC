#!/usr/bin/env python3
"""Project state-derived JET--CTRL eddy torque through a nonuniform-grid SE operator.

This is deliberately narrower than a full CM1 tendency budget.  It uses only
the explicit three-dimensional eddy flux convergence cached from state fields.
The CM1 ``ub_*``/``vb_*`` fields are not used because their hourly snapshots
are not interval means and have not passed the P2 closure gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.se_bui import build_basic_state, build_forcing, invert_balanced_theta, regularize_ellipticity
from src.se_nonuniform import solve_flux_form_dirichlet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/thompson_240h_se_attribution.json")
    parser.add_argument("--hours", type=float, nargs="+", default=None)
    parser.add_argument("--eps-ratios", type=float, nargs="+", default=None)
    return parser.parse_args()


def tag(hour: float) -> str:
    return f"t{hour:05.1f}h".replace(".", "p")


def load_cache(cache_dir: Path, case: str, hour: float) -> dict[str, np.ndarray]:
    path = cache_dir / f"{case}_{tag(hour)}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing state cache: {path}")
    with np.load(path) as product:
        required = ("r_m", "z_m", "vt", "theta", "rho", "F_lambda_eddy")
        missing = [name for name in required if name not in product]
        if missing:
            raise KeyError(f"{path}: missing {missing}")
        return {name: np.asarray(product[name], dtype=np.float64) for name in required}


def operator_from_state(state: dict[str, np.ndarray], fcor: float, eps_ratio: float) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    r_m, z_m = state["r_m"], state["z_m"]
    theta_bal, thermal_wind = invert_balanced_theta(state["vt"], state["theta"], r_m, z_m, fcor)
    basic = build_basic_state(state["vt"], theta_bal, state["rho"], r_m, z_m, fcor)
    k1, k2, k3, regularization = regularize_ellipticity(
        basic["K1_raw"], basic["K2_raw"], basic["K3_raw"], eps_ratio=eps_ratio
    )
    metric = 1.0 / (np.maximum(state["rho"], 1.0e-10) * np.maximum(r_m[None, :], 0.5 * np.min(np.diff(r_m))))
    return {
        "basic": basic,
        "a": k1 * metric,
        "b": k2 * metric,
        "c": k3 * metric,
        "theta_bal": theta_bal,
    }, {"thermal_wind": thermal_wind, "regularization": regularization}


def velocity_from_psi(psi_zr: np.ndarray, rho_zr: np.ndarray, r_m: np.ndarray, z_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mass-streamfunction conversion on the same nonuniform coordinates."""
    denom = np.maximum(rho_zr * np.maximum(r_m[None, :], 0.5 * np.min(np.diff(r_m))), 1.0e-10)
    ur = -np.gradient(psi_zr, z_m, axis=0, edge_order=2) / denom
    w = np.gradient(psi_zr, r_m, axis=1, edge_order=2) / denom
    # Boundary derivatives reflect the stated finite-domain closure, not an
    # interior circulation response.  Exclude them from quantitative metrics.
    ur[[0, -1], :] = np.nan; ur[:, [0, -1]] = np.nan
    w[[0, -1], :] = np.nan; w[:, [0, -1]] = np.nan
    return ur, w


def metrics(field: np.ndarray, r_m: np.ndarray, z_m: np.ndarray) -> dict[str, dict[str, float]]:
    rr, zz = np.meshgrid(r_m / 1000.0, z_m / 1000.0)
    masks = {
        "interior": np.ones(field.shape, dtype=bool),
        "inner_outflow": (rr >= 50.0) & (rr <= 350.0) & (zz >= 10.0) & (zz <= 16.0),
        "outer_jet_layer": (rr >= 750.0) & (rr <= 1100.0) & (zz >= 10.0) & (zz <= 17.0),
    }
    out = {}
    for name, mask in masks.items():
        values = field[mask & np.isfinite(field)]
        out[name] = {
            "mean": float(np.mean(values)) if values.size else float("nan"),
            "rms": float(np.sqrt(np.mean(values**2))) if values.size else float("nan"),
            "max_abs": float(np.max(np.abs(values))) if values.size else float("nan"),
            "n": int(values.size),
        }
    return out


def plot(hour: float, r_m: np.ndarray, z_m: np.ndarray, f_env: np.ndarray, rhs_env: np.ndarray,
         source_psi: np.ndarray, source_ur: np.ndarray, source_w: np.ndarray, path: Path) -> None:
    fields = (f_env, rhs_env, source_psi, source_ur, source_w)
    titles = (
        r"$F_{\lambda,eddy}^{JET}-F_{\lambda,eddy}^{CTRL}$ (m s$^{-2}$)",
        r"fixed-CTRL eddy RHS (K$^{-1}$ s$^{-3}$)",
        r"fixed-CTRL $\Delta\psi$ (kg s$^{-1}$)",
        r"fixed-CTRL $\Delta u_r$ (m s$^{-1}$)",
        r"fixed-CTRL $\Delta w$ (m s$^{-1}$)",
    )
    fig, axes = plt.subplots(1, 5, figsize=(24, 4.8), constrained_layout=True, sharey=True)
    for ax, field, title in zip(axes, fields, titles):
        finite = np.abs(field[np.isfinite(field)])
        vmax = max(float(np.percentile(finite, 99.0)) if finite.size else 0.0, 1.0e-30)
        im = ax.contourf(r_m / 1000.0, z_m / 1000.0, field, levels=np.linspace(-vmax, vmax, 25), cmap="RdBu_r", extend="both")
        ax.set(title=title, xlabel="Radius (km)", xlim=(0, 1200), ylim=(0, 24))
        fig.colorbar(im, ax=ax, pad=0.02)
    axes[0].set_ylabel("Height (km)")
    fig.suptitle(f"Direct eddy-torque balanced projection, {hour:g} h", fontweight="bold")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_one(hour: float, ctrl: dict[str, np.ndarray], jet: dict[str, np.ndarray], fcor: float,
            eps_ratio: float, out_dir: Path, make_plot: bool) -> dict[str, object]:
    for coord in ("r_m", "z_m"):
        if not np.array_equal(ctrl[coord], jet[coord]):
            raise ValueError(f"CTRL/JET cache has non-identical {coord}")
    r_m, z_m = ctrl["r_m"], ctrl["z_m"]
    op_c, info_c = operator_from_state(ctrl, fcor, eps_ratio)
    op_j, info_j = operator_from_state(jet, fcor, eps_ratio)
    zero = np.zeros_like(ctrl["F_lambda_eddy"])
    rhs_cc = build_forcing(op_c["basic"], zero, ctrl["F_lambda_eddy"], r_m, z_m)["forcing_total"]
    rhs_cj = build_forcing(op_c["basic"], zero, jet["F_lambda_eddy"], r_m, z_m)["forcing_total"]
    rhs_jj = build_forcing(op_j["basic"], zero, jet["F_lambda_eddy"], r_m, z_m)["forcing_total"]
    rhs_env = build_forcing(op_c["basic"], zero, jet["F_lambda_eddy"] - ctrl["F_lambda_eddy"], r_m, z_m)["forcing_total"]
    solve_c = lambda rhs: solve_flux_form_dirichlet(op_c["a"], op_c["b"], op_c["c"], rhs, r_m, z_m)
    solve_j = lambda rhs: solve_flux_form_dirichlet(op_j["a"], op_j["b"], op_j["c"], rhs, r_m, z_m)
    cc, cj, jc, jj, env = solve_c(rhs_cc), solve_c(rhs_cj), solve_j(rhs_cc), solve_j(rhs_jj), solve_c(rhs_env)
    source = cj.psi - cc.psi
    operator = jc.psi - cc.psi
    interaction = jj.psi - jc.psi - cj.psi + cc.psi
    total = jj.psi - cc.psi
    decomposition_error = total - source - operator - interaction
    rhs_linearity_error = rhs_env - (rhs_cj - rhs_cc)
    source_ur, source_w = velocity_from_psi(source, ctrl["rho"], r_m, z_m)
    stem = f"direct_eddy_se_{tag(hour)}_eps{eps_ratio:.0e}"
    np.savez_compressed(
        out_dir / f"{stem}.npz", r_m=r_m, z_m=z_m, F_lambda_eddy_ctrl=ctrl["F_lambda_eddy"],
        F_lambda_eddy_jet=jet["F_lambda_eddy"], F_lambda_env=jet["F_lambda_eddy"] - ctrl["F_lambda_eddy"],
        rhs_CC=rhs_cc, rhs_CJ=rhs_cj, rhs_JJ=rhs_jj, rhs_env_ctrl=rhs_env,
        psi_CC=cc.psi, psi_CJ=cj.psi, psi_JC=jc.psi, psi_JJ=jj.psi,
        delta_psi_source=source, delta_psi_operator=operator, delta_psi_interaction=interaction,
        delta_psi_total=total, decomposition_error=decomposition_error,
        delta_ur_source_fixed_ctrl=source_ur, delta_w_source_fixed_ctrl=source_w,
        theta_bal_ctrl=op_c["theta_bal"], theta_bal_jet=op_j["theta_bal"],
    )
    if make_plot:
        plot(hour, r_m, z_m, jet["F_lambda_eddy"] - ctrl["F_lambda_eddy"], rhs_env, source, source_ur, source_w,
             out_dir / f"{stem}.png")
    solvers = {"CC": cc, "CJ": cj, "JC": jc, "JJ": jj, "env": env}
    return {
        "hour": hour, "eps_ratio": eps_ratio, "file": f"{stem}.npz",
        "figure": f"{stem}.png" if make_plot else "",
        "numerical_residuals": {name: {"relative": q.relative_residual, "absolute": q.absolute_residual} for name, q in solvers.items()},
        "rhs_linearity_relative_error": float(np.linalg.norm(rhs_linearity_error) / max(np.linalg.norm(rhs_env), 1.0e-30)),
        "four_way_identity_relative_error": float(np.linalg.norm(decomposition_error) / max(np.linalg.norm(total), 1.0e-30)),
        "metrics": {
            "F_lambda_env": metrics(jet["F_lambda_eddy"] - ctrl["F_lambda_eddy"], r_m, z_m),
            "delta_psi_source_fixed_CTRL": metrics(source, r_m, z_m),
            "delta_ur_source_fixed_CTRL": metrics(source_ur, r_m, z_m),
            "delta_w_source_fixed_CTRL": metrics(source_w, r_m, z_m),
        },
        "ctrl": info_c, "jet": info_j,
    }


def main() -> None:
    args = parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    hours = args.hours if args.hours is not None else config["windows"]["smoke_hours"]
    eps_ratios = args.eps_ratios if args.eps_ratios is not None else config["se"]["regularization_eps_ratios"]
    out_dir = ROOT / config["output_dir"] / "se" / "direct_eddy"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for hour in hours:
        ctrl = load_cache(ROOT / config["output_dir"] / "cache", "CTRL", hour)
        jet = load_cache(ROOT / config["output_dir"] / "cache", "JET", hour)
        for eps in eps_ratios:
            record = run_one(hour, ctrl, jet, float(config["se"]["f_s-1"]), float(eps), out_dir,
                             make_plot=bool(np.isclose(eps, 1.0e-4)))
            records.append(record)
            print(json.dumps({"hour": hour, "eps": eps, "file": record["file"]}), flush=True)
    summary = {
        "status": "BALANCED_PROJECTION_ONLY",
        "scope": "state-derived JET-CTRL eddy tangential-momentum flux convergence",
        "excludes": "raw CM1 ub_*/vb_* budget forcing; P2 closure did not pass",
        "operator": "Bui height-coordinate operator, nonuniform-grid flux form, homogeneous psi=0 on all finite boundaries",
        "strict_four_way": "L_C^-1 b_C, L_C^-1 b_J, L_J^-1 b_C, L_J^-1 b_J; b is assembled before operator exchange",
        "velocity_mapping": "source response maps psi through CTRL rho for a fixed-reference diagnostic",
        "records": records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
