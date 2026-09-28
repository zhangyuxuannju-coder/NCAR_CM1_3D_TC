#!/usr/bin/env python3
"""Calculate strength-matched JET--CTRL equivalent SE operator forcing.

The primary result is evaluated in precisely the non-uniform discrete operator
used by the SE solver:

    S_op,lead = -(L_J - L_C) psi_CC.

``psi_CC`` is the CTRL balanced response to the CTRL *direct eddy torque*
only.  This controlled reference is necessary because the archived CM1 budget
snapshots fail the state-tendency closure test.  The output is consequently an
operator-projection diagnostic, never a full CM1 forcing attribution.

For continuity with the earlier calculation, the script also saves S_A, S_I,
and S_B.  Their sum retains the historical coefficient-only definition.  The
difference from the fully discrete S_op contains the density-radius metric and
discretization remainder and must not be assigned a separate physical cause.
"""

from __future__ import annotations

import argparse
import csv
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
from src.se_nonuniform import assemble_flux_form_matrix, solve_flux_form_dirichlet


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matches", default="output/thompson_240h_se_attribution/matching/strength_matched_pairs.csv")
    p.add_argument("--output-dir", default="output/thompson_240h_se_attribution/se/matched_equivalent_operator")
    p.add_argument("--eps", type=float, nargs="+", default=[1e-5, 1e-4, 1e-3])
    p.add_argument("--f", type=float, default=6.2e-5)
    p.add_argument("--jet-r-min-km", type=float, default=600.0)
    p.add_argument("--jet-z-min-km", type=float, default=8.0)
    p.add_argument("--jet-z-max-km", type=float, default=18.0)
    return p.parse_args()


def tag(hour: float) -> str:
    return f"t{hour:05.1f}h".replace(".", "p")


def load(case: str, hour: float) -> dict[str, np.ndarray]:
    p = ROOT / "output/thompson_240h_se_attribution/cache" / f"{case}_{tag(hour)}.npz"
    with np.load(p) as a:
        fields = ("r_m", "z_m", "ur", "vt", "w", "theta", "rho", "F_lambda_eddy")
        return {k: np.asarray(a[k], dtype=float) for k in fields}


def zero_edge(x: np.ndarray) -> np.ndarray:
    out = np.array(x, dtype=float, copy=True)
    out[[0, -1], :] = 0.0
    out[:, [0, -1]] = 0.0
    return out


def gradient(x: np.ndarray, coord: np.ndarray, axis: int) -> np.ndarray:
    return np.gradient(np.asarray(x, float), coord, axis=axis, edge_order=2)


def operator(state: dict[str, np.ndarray], fcor: float, eps: float) -> dict[str, object]:
    r, z = state["r_m"], state["z_m"]
    theta, tw = invert_balanced_theta(state["vt"], state["theta"], r, z, fcor)
    basic = build_basic_state(state["vt"], theta, state["rho"], r, z, fcor)
    k1, k2, k3, reg = regularize_ellipticity(
        basic["K1_raw"], basic["K2_raw"], basic["K3_raw"], eps_ratio=eps,
    )
    r_safe = np.maximum(r, 0.5 * np.min(np.diff(r)))
    metric = 1.0 / np.maximum(state["rho"] * r_safe[None, :], 1e-10)
    matrix = assemble_flux_form_matrix(k1 * metric, k2 * metric, k3 * metric, r, z)
    return {"basic": basic, "k1": k1, "k2": k2, "k3": k3, "metric": metric,
            "matrix": matrix, "regularization": reg, "thermal_wind": tw}


def winds(psi: np.ndarray, rho: np.ndarray, r: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    den = np.maximum(rho * np.maximum(r[None, :], 0.5 * np.min(np.diff(r))), 1e-10)
    u = -gradient(psi, z, 0) / den
    w = gradient(psi, r, 1) / den
    return u, w


def old_components(kc: dict[str, object], kj: dict[str, object], u: np.ndarray, w: np.ndarray,
                   r: np.ndarray, z: np.ndarray) -> dict[str, np.ndarray]:
    # dB = -dK2 is deliberately retained to match the previous diagnostic.
    da = np.asarray(kj["k1"]) - np.asarray(kc["k1"])
    di = np.asarray(kj["k3"]) - np.asarray(kc["k3"])
    db = -(np.asarray(kj["k2"]) - np.asarray(kc["k2"]))
    sa = -gradient(da * w, r, 1)
    si = gradient(di * u, z, 0)
    sb = -gradient(db * u, r, 1) + gradient(db * w, z, 0)
    return {"S_A": zero_edge(sa), "S_I": zero_edge(si), "S_B": zero_edge(sb),
            "S_ABC": zero_edge(sa + si + sb)}


def norm(x: np.ndarray, mask: np.ndarray | None = None) -> float:
    a = np.asarray(x, float)
    if mask is not None:
        a = a[mask]
    a = a[np.isfinite(a)]
    return float(np.linalg.norm(a))


def rms(x: np.ndarray, mask: np.ndarray) -> float:
    a = np.asarray(x, float)[mask]
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a * a))) if a.size else float("nan")


def projection(x: np.ndarray, total: np.ndarray, mask: np.ndarray) -> float:
    use = mask & np.isfinite(x) & np.isfinite(total)
    return float(np.sum(x[use] * total[use]) / max(np.sum(total[use] ** 2), 1e-30))


def fig(path: Path, jh: float, ch: float, r: np.ndarray, z: np.ndarray, fields: dict[str, np.ndarray],
        du_lead: np.ndarray, dw_lead: np.ndarray, du_jet: np.ndarray, dw_jet: np.ndarray,
        jet_mask: np.ndarray) -> None:
    panels = [fields["S_A"], fields["S_I"], fields["S_B"], fields["S_ABC"], fields["S_op_leading"],
              fields["S_op_remainder"], np.where(jet_mask, fields["S_op_leading"], np.nan),
              du_lead, dw_lead, du_jet, dw_jet, fields["S_op_exact"]]
    titles = [r"legacy $S_A$", r"legacy $S_I$", r"legacy $S_B$", r"legacy $S_A+S_I+S_B$",
              r"discrete $S_{op,lead}$", "metric/discretization remainder", r"$S_{op,lead}$ in jet box",
              r"$\delta u_r$: full leading", r"$\delta w$: full leading", r"$\delta u_r$: jet-box source",
              r"$\delta w$: jet-box source", r"exact operator RHS"]
    fig, ax = plt.subplots(3, 4, figsize=(16, 11), constrained_layout=True, sharex=True, sharey=True)
    for a, x, title in zip(ax.flat, panels, titles):
        vals = np.abs(x[np.isfinite(x)])
        vmax = max(float(np.percentile(vals, 99)) if vals.size else 0.0, 1e-30)
        im = a.contourf(r / 1000, z / 1000, x, levels=np.linspace(-vmax, vmax, 25), cmap="RdBu_r", extend="both")
        a.set(title=title, xlim=(0, 1200), ylim=(0, 24), xlabel="Radius (km)", ylabel="Height (km)")
        a.axvline(600, color="k", lw=.55, ls="--")
        fig.colorbar(im, ax=a, pad=.015)
    fig.suptitle(f"Strength-matched equivalent operator: J{jh:g} h minus C{ch:g} h\n"
                 "CTRL direct-eddy SE reference; dashed line marks 600 km", fontweight="bold")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    a = args()
    out = ROOT / a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    with (ROOT / a.matches).open(encoding="utf-8") as h:
        pairs = list(csv.DictReader(h))
    records = []
    for pair in pairs:
        jh, ch = float(pair["jet_hour"]), float(pair["ctrl_hour"])
        ctrl, jet = load("CTRL", ch), load("JET", jh)
        r, z = ctrl["r_m"], ctrl["z_m"]
        if not (np.array_equal(r, jet["r_m"]) and np.array_equal(z, jet["z_m"])):
            raise ValueError("matched cache grids differ")
        rr, zz = np.meshgrid(r / 1000, z / 1000)
        jet_box = (rr >= a.jet_r_min_km) & (zz >= a.jet_z_min_km) & (zz <= a.jet_z_max_km)
        inner_low = (rr >= 20) & (rr <= 150) & (zz >= .5) & (zz <= 2)
        inner_ascent = (rr >= 20) & (rr <= 150) & (zz >= 2) & (zz <= 12)
        for eps in a.eps:
            kc, kj = operator(ctrl, a.f, eps), operator(jet, a.f, eps)
            # Controlled reference, avoiding known snapshot-budget nonclosure.
            rhs_cc = zero_edge(build_forcing(kc["basic"], np.zeros_like(ctrl["F_lambda_eddy"]),
                                               ctrl["F_lambda_eddy"], r, z)["forcing_total"])
            solve_c = lambda source: solve_flux_form_dirichlet(
                np.asarray(kc["k1"]) * np.asarray(kc["metric"]), np.asarray(kc["k2"]) * np.asarray(kc["metric"]),
                np.asarray(kc["k3"]) * np.asarray(kc["metric"]), zero_edge(source), r, z)
            psi_cc = solve_c(rhs_cc).psi
            psi_jc_result = solve_flux_form_dirichlet(
                np.asarray(kj["k1"]) * np.asarray(kj["metric"]), np.asarray(kj["k2"]) * np.asarray(kj["metric"]),
                np.asarray(kj["k3"]) * np.asarray(kj["metric"]), rhs_cc, r, z)
            psi_jc = psi_jc_result.psi
            dmatrix = kj["matrix"] - kc["matrix"]
            s_lead = zero_edge(-(dmatrix @ psi_cc.ravel()).reshape(psi_cc.shape))
            # Exact rearrangement: L_C(psi_JC-psi_CC) = -(L_J-L_C) psi_JC.
            dpsi_exact = psi_jc - psi_cc
            s_exact = zero_edge((kc["matrix"] @ dpsi_exact.ravel()).reshape(psi_cc.shape))
            s_second = s_exact - s_lead
            legacy = old_components(kc, kj, *winds(psi_cc, ctrl["rho"], r, z), r, z)
            fields = {**legacy, "S_op_leading": s_lead, "S_op_exact": s_exact,
                      "S_op_second_order": s_second, "S_op_remainder": s_lead - legacy["S_ABC"]}
            sol_lead = solve_c(s_lead).psi
            sol_jet = solve_c(s_lead * jet_box).psi
            du_lead, dw_lead = winds(sol_lead, ctrl["rho"], r, z)
            du_jet, dw_jet = winds(sol_jet, ctrl["rho"], r, z)
            stem = f"J{jh:05.1f}_C{ch:05.1f}_eps{eps:.0e}".replace(".", "p")
            np.savez_compressed(out / f"{stem}.npz", r_m=r, z_m=z, psi_CC=psi_cc, psi_JC=psi_jc,
                                rhs_CC=rhs_cc, jet_upper_mask=jet_box, delta_ur_leading=du_lead,
                                delta_w_leading=dw_lead, delta_ur_jet_upper=du_jet, delta_w_jet_upper=dw_jet,
                                **fields)
            response_error = norm(sol_lead - dpsi_exact) / max(norm(dpsi_exact), 1e-30)
            rec = {"jet_hour": jh, "ctrl_hour": ch, "eps_ratio": eps, "file": f"{stem}.npz",
                   "status": "BALANCED_OPERATOR_PROJECTION_DIRECT_EDDY_REFERENCE_ONLY",
                   "matrix_relative_residual_CC": solve_c(rhs_cc).relative_residual,
                   "matrix_relative_residual_JC": psi_jc_result.relative_residual,
                   "leading_vs_exact_response_relative_error": response_error,
                   "second_order_rhs_fraction_l2": norm(s_second) / max(norm(s_exact), 1e-30),
                   "legacy_ABC_fraction_of_discrete_leading_l2": norm(legacy["S_ABC"]) / max(norm(s_lead), 1e-30),
                   "metric_discretization_remainder_fraction_l2": norm(fields["S_op_remainder"]) / max(norm(s_lead), 1e-30),
                   "jet_upper_operator_rhs_fraction_l2": norm(s_lead, jet_box) / max(norm(s_lead), 1e-30),
                   "low_inflow_jet_projection_on_leading": projection(du_jet, du_lead, inner_low),
                   "inner_ascent_jet_projection_on_leading": projection(dw_jet, dw_lead, inner_ascent),
                   "low_inflow_leading_rms": rms(du_lead, inner_low), "low_inflow_jet_rms": rms(du_jet, inner_low),
                   "inner_ascent_leading_rms": rms(dw_lead, inner_ascent), "inner_ascent_jet_rms": rms(dw_jet, inner_ascent),
                   "regularization_CTRL": kc["regularization"], "regularization_JET": kj["regularization"]}
            records.append(rec)
            if np.isclose(eps, 1e-4):
                fig(out / f"{stem}.png", jh, ch, r, z, fields, du_lead, dw_lead, du_jet, dw_jet, jet_box)
            print(json.dumps({"JET": jh, "CTRL": ch, "eps": eps,
                              "jet_operator_rhs_fraction": rec["jet_upper_operator_rhs_fraction_l2"],
                              "legacy_fraction": rec["legacy_ABC_fraction_of_discrete_leading_l2"]}), flush=True)
    (out / "summary.json").write_text(json.dumps({
        "status": "BALANCED_OPERATOR_PROJECTION_DIRECT_EDDY_REFERENCE_ONLY",
        "primary_definition": "S_op_leading=-(L_JET-L_CTRL)psi_CC using the same nonuniform discrete L used by the solver",
        "legacy_definition": "S_A+S_I+S_B uses the historical DeltaK-only continuum form with the CTRL SE reference winds",
        "remainder_definition": "S_op_leading-(S_A+S_I+S_B); contains density-radius metric and discrete-form differences, not a separately attributable physical forcing",
        "jet_region_definition": "support of S_op_leading restricted to r>=600 km and 8<=z<=18 km; it is JET-run-induced rather than a pure imposed-jet effect",
        "causal_scope": "balanced diagnostic with direct-eddy CTRL reference only; raw CM1 tendency-budget snapshots did not close and are excluded",
        "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
