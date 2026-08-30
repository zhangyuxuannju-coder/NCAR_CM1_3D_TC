#!/usr/bin/env python3
"""Invert the previously diagnosed equivalent operator forcing by itself.

The solved experiment is

    L_CTRL,reg(psi_eq) = S_total = -delta(L) psi_CTRL,

with the traditional thermal and momentum RHS set to zero.  The forcing is
loaded verbatim from operator_perturbation_products.npz; it is not recomputed.
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src._se_pipeline_single import (
    PipelineConfig,
    _rho_ext_from_rho_zr,
    _to_solver_layout_zr_to_rz,
    azimuthal_average_from_3d,
    psi_to_uw,
    solve_se_sparse,
)
from src.se_bui import (
    assemble_operator,
    build_basic_state,
    invert_balanced_theta,
    regularize_ellipticity,
)


CMAP = LinearSegmentedColormap.from_list(
    "operator_only", ["#3C5488", "#8FAACC", "#F7F7F4", "#EFA287", "#B9363E"]
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ctrl", default="/data/zhangyx/DATA/cm1out_25N_nojet.nc")
    p.add_argument("--output-dir", default="output/operator_equivalent_forcing_only_se_25N")
    p.add_argument("--f", type=float, default=6.2e-5)
    p.add_argument("--max-r-km", type=float, default=1200.0)
    p.add_argument("--max-z-km", type=float, default=20.0)
    p.add_argument("--dr-km", type=float, default=12.0)
    p.add_argument("--eps-ratio", type=float, default=1.0e-5)
    p.add_argument("--elliptic-margin", type=float, default=0.0)
    p.add_argument("--baroclinic-scale", type=float, default=1.0)
    return p.parse_args()


def experiment_specs() -> list[dict]:
    base = Path("output")
    return [
        {"case": "JET15", "hour": 0.0, "folder": base / "operator_perturbation_forcing_25N_jet15/t000h"},
        {"case": "JET15", "hour": 5.0, "folder": base / "operator_perturbation_forcing_25N_jet15/t005h"},
        {"case": "JET30", "hour": 0.0, "folder": base / "operator_perturbation_forcing_25N_jet30/t000h"},
        {"case": "JET30", "hour": 5.0, "folder": base / "operator_perturbation_forcing_25N_jet30/t005h"},
    ]


def load_ctrl(hour: float, args: argparse.Namespace) -> dict[str, np.ndarray]:
    cfg = PipelineConfig(
        input_file=args.ctrl,
        output_dir=args.output_dir,
        target_time_hours=hour,
        max_r_km=args.max_r_km,
        dr_km=args.dr_km,
        max_z_km=args.max_z_km,
        coriolis_f=args.f,
        include_model_budget_terms=False,
        write_netcdf=False,
        write_ieee=False,
        plot_solution=False,
        sor_max_iter=80000,
        sor_omega=1.5,
        sor_tol=1.0e-14,
        sor_verbose_every=0,
    )
    return azimuthal_average_from_3d(cfg)


def build_ctrl_operator(avg: dict[str, np.ndarray], args: argparse.Namespace) -> dict:
    r_m = np.asarray(avg["r_km"], float) * 1000.0
    z_m = np.asarray(avg["z_km"], float) * 1000.0
    theta_bal, _ = invert_balanced_theta(avg["ut"], avg["theta"], r_m, z_m, args.f)
    basic = build_basic_state(
        avg["ut"], theta_bal, avg["rho"], r_m, z_m, args.f,
        baroclinic_scale=args.baroclinic_scale,
    )
    k1_raw = np.asarray(basic["K1_raw"], float)
    k2_raw = np.asarray(basic["K2_raw"], float)
    k3_raw = np.asarray(basic["K3_raw"], float)
    d_raw = k1_raw * k3_raw - k2_raw**2
    k1, k2, k3, reg = regularize_ellipticity(
        k1_raw, k2_raw, k3_raw,
        eps_ratio=args.eps_ratio,
        margin=args.elliptic_margin,
    )
    s1 = max(float(np.nanmax(np.abs(k1_raw))), 1.0e-30)
    s2 = max(float(np.nanmax(np.abs(k2_raw))), 1.0e-30)
    s3 = max(float(np.nanmax(np.abs(k3_raw))), 1.0e-30)
    changed = (
        (np.abs(k1 - k1_raw) > 1.0e-12 * s1)
        | (np.abs(k2 - k2_raw) > 1.0e-12 * s2)
        | (np.abs(k3 - k3_raw) > 1.0e-12 * s3)
    )
    raw_bad = (
        (~np.isfinite(d_raw)) | (~np.isfinite(k1_raw)) | (~np.isfinite(k3_raw))
        | (k1_raw <= 0.0) | (k3_raw <= 0.0) | (d_raw <= 0.0)
    )
    return {
        "basic": basic,
        "operator": assemble_operator(basic, k1, k2, k3, r_m, z_m),
        "changed": changed,
        "raw_bad": raw_bad,
        "regularization": reg,
    }


def solve_forcing(avg: dict, ctrl_op: dict, forcing: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_m = np.asarray(avg["r_km"], float) * 1000.0
    z_m = np.asarray(avg["z_km"], float) * 1000.0
    op = ctrl_op["operator"]
    arrays = {k: _to_solver_layout_zr_to_rz(op[k]) for k in ("A", "B", "C", "D", "E")}
    dr = float(np.mean(np.diff(r_m)))
    dz = float(np.mean(np.diff(z_m)))
    psi_ext = solve_se_sparse(
        A=arrays["A"], B=arrays["B"], C=arrays["C"], D=arrays["D"], E=arrays["E"],
        Fin=_to_solver_layout_zr_to_rz(forcing), dr=dr, dz=dz,
    )
    rho_ext = _rho_ext_from_rho_zr(ctrl_op["basic"]["rho"])
    u_ext, w_ext = psi_to_uw(psi_ext, rho_ext, r_m, dr, dz)
    return psi_ext[:, 1:-1].T, u_ext[:, 1:-1].T, w_ext[:, 1:-1].T


def weighted_mean(field: np.ndarray, weight: np.ndarray, mask: np.ndarray) -> float:
    good = mask & np.isfinite(field) & np.isfinite(weight)
    return float(np.sum(field[good] * weight[good]) / np.sum(weight[good]))


def weighted_rms(field: np.ndarray, weight: np.ndarray, mask: np.ndarray) -> float:
    good = mask & np.isfinite(field) & np.isfinite(weight)
    return float(np.sqrt(np.sum(field[good] ** 2 * weight[good]) / np.sum(weight[good])))


def robust_scale(fields: list[np.ndarray], masks: list[np.ndarray], percentile: float = 98.5) -> float:
    values = []
    for field, mask in zip(fields, masks):
        values.append(np.abs(field[mask & np.isfinite(field)]))
    values = np.concatenate(values) if values else np.array([1.0])
    return max(float(np.percentile(values, percentile)), 1.0e-30)


def collect_one(spec: dict, avg: dict, ctrl_op: dict, args: argparse.Namespace) -> dict:
    products = spec["folder"] / "operator_perturbation_products.npz"
    scale_json = spec["folder"] / "scale_analysis.json"
    with np.load(products) as data:
        r_force = np.asarray(data["r_km"], float)
        z_force = np.asarray(data["z_km"], float)
        forcing = np.asarray(data["S_total"], float)
    r_km = np.asarray(avg["r_km"], float)
    z_km = np.asarray(avg["z_km"], float)
    if r_force.shape != r_km.shape or z_force.shape != z_km.shape:
        raise ValueError(f"Grid shape mismatch for {spec['case']} {spec['hour']} h")
    if not np.allclose(r_force, r_km, atol=1.0e-9, rtol=0.0) or not np.allclose(z_force, z_km, atol=1.0e-9, rtol=0.0):
        raise ValueError(f"Grid coordinate mismatch for {spec['case']} {spec['hour']} h")
    psi, u, w = solve_forcing(avg, ctrl_op, forcing)
    rr, zz = np.meshgrid(r_km, z_km)
    weight = np.maximum(rr, 1.0)
    full = (rr >= 30) & (rr <= 1200) & (zz >= 0.5) & (zz <= 18)
    upper = (rr >= 50) & (rr <= 300) & (zz >= 10) & (zz <= 16)
    low = (rr >= 50) & (rr <= 300) & (zz >= 0.5) & (zz <= 3)
    ascent = (rr >= 20) & (rr <= 150) & (zz >= 2) & (zz <= 14)
    jet_annulus = (rr >= 650) & (rr <= 1150) & (zz >= 10) & (zz <= 16)
    approx = json.loads(scale_json.read_text(encoding="utf-8"))
    selected = float(avg["time_seconds_used"][0] / 3600.0)
    if abs(selected - spec["hour"]) > 1.0e-6:
        raise ValueError(f"Selected CTRL time {selected} differs from forcing time {spec['hour']}")
    if not (np.all(np.isfinite(psi)) and np.all(np.isfinite(u)) and np.all(np.isfinite(w))):
        raise FloatingPointError(f"Non-finite SE response for {spec['case']} {spec['hour']} h")
    return {
        **spec,
        "r_km": r_km,
        "z_km": z_km,
        "forcing": forcing,
        "psi": psi,
        "u": u,
        "w": w,
        "changed": ctrl_op["changed"],
        "raw_bad": ctrl_op["raw_bad"],
        "metrics": {
            "selected_time_hours": selected,
            "forcing_full_rms_K-1_s-3": weighted_rms(forcing, weight, full),
            "psi_full_rms_kg_s-1": weighted_rms(psi, weight, full),
            "u_full_rms_m_s-1": weighted_rms(u, weight, full),
            "w_full_rms_m_s-1": weighted_rms(w, weight, full),
            "upper_outflow_mean_m_s-1": weighted_mean(u, weight, upper),
            "low_inflow_strength_change_m_s-1": -weighted_mean(u, weight, low),
            "core_ascent_mean_m_s-1": weighted_mean(w, weight, ascent),
            "jet_annulus_u_mean_m_s-1": weighted_mean(u, weight, jet_annulus),
            "regularized_fraction": float(np.mean(ctrl_op["changed"])),
            "raw_nonelliptic_fraction": float(np.mean(ctrl_op["raw_bad"])),
            "second_order_proxy_ratio_full": approx["domains"]["full"]["second_order_proxy_ratio"],
            "second_order_proxy_ratio_inner": approx["domains"]["inner_outflow"]["second_order_proxy_ratio"],
            "smoothing_change_ratio_full": approx["domains"]["full"]["smoothing_change_ratio"],
            "density_correction_ratio_full": approx["domains"]["full"]["density_correction_ratio"],
        },
    }


def add_dots(ax: plt.Axes, rec: dict, xlim: tuple[float, float]) -> None:
    zi, ri = np.where(rec["changed"])
    keep = (rec["r_km"][ri] >= xlim[0]) & (rec["r_km"][ri] <= xlim[1])
    if np.any(keep):
        ax.scatter(rec["r_km"][ri[keep]], rec["z_km"][zi[keep]], s=2.0, c="0.15", alpha=0.28, linewidths=0)


def plot_comparison(records: list[dict], out: Path, xlim: tuple[float, float], title_suffix: str) -> None:
    fields = {
        "psi": [r["psi"] / 1.0e9 for r in records],
        "u": [r["u"] for r in records],
        "w": [1000.0 * r["w"] for r in records],
    }
    masks = []
    for rec in records:
        rr, zz = np.meshgrid(rec["r_km"], rec["z_km"])
        masks.append((rr >= max(20.0, xlim[0])) & (rr <= xlim[1]) & (zz >= 0.5) & (zz <= 18))
    scales = {key: robust_scale(value, masks) for key, value in fields.items()}
    labels = {
        "psi": r"$\psi_{eq}$ ($10^9$ kg s$^{-1}$)",
        "u": r"$u_{eq}$ (m s$^{-1}$)",
        "w": r"$w_{eq}$ (mm s$^{-1}$)",
    }
    fig, axes = plt.subplots(len(records), 3, figsize=(13.7, 3.05 * len(records)), sharex=True, sharey=True, constrained_layout=True)
    maps = []
    for i, rec in enumerate(records):
        for j, key in enumerate(("psi", "u", "w")):
            ax = axes[i, j]
            fld = fields[key][i]
            vmax = scales[key]
            m = ax.pcolormesh(
                rec["r_km"], rec["z_km"], fld, shading="auto", cmap=CMAP,
                norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax), rasterized=True,
            )
            maps.append(m)
            if np.nanmin(fld) < 0 < np.nanmax(fld):
                ax.contour(rec["r_km"], rec["z_km"], fld, levels=[0], colors="0.2", linewidths=0.5)
            add_dots(ax, rec, xlim)
            if xlim[1] > 900:
                ax.plot(888, 12, marker="*", ms=7, color="#7A1FA2", mec="white", mew=0.4)
            ax.set(xlim=xlim, ylim=(0, 18))
            ax.axhline(10, color="0.4", lw=0.45, ls=":")
            ax.axhline(16, color="0.4", lw=0.45, ls=":")
            if i == 0:
                ax.set_title(labels[key])
            if j == 0:
                ax.set_ylabel(f"{rec['case']} · {rec['hour']:.0f} h\nHeight (km)")
            if i == len(records) - 1:
                ax.set_xlabel("Radius (km)")
        met = rec["metrics"]
        axes[i, 2].text(
            0.98, 0.03,
            f"OUT={met['upper_outflow_mean_m_s-1']:+.3f} m s⁻¹\n"
            f"IN={met['low_inflow_strength_change_m_s-1']:+.3f} m s⁻¹\n"
            f"Wcore={1000*met['core_ascent_mean_m_s-1']:+.2f} mm s⁻¹",
            transform=axes[i, 2].transAxes, ha="right", va="bottom", fontsize=7.7,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.0},
        )
    for j, key in enumerate(("psi", "u", "w")):
        cb = fig.colorbar(maps[j], ax=axes[:, j], orientation="horizontal", shrink=0.80, pad=0.025)
        cb.set_label(f"Shared P98.5 scale: ±{scales[key]:.3g}")
    fig.suptitle(
        f"SE response to equivalent operator forcing only — {title_suffix}\n"
        r"$\mathcal{L}_{CTRL,reg}\psi_{eq}=S_{op}^{eq}$; traditional thermal/momentum forcing = 0; dots = regularized operator",
        fontweight="bold",
    )
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_record(rec: dict, out: Path) -> None:
    tag = f"{rec['case'].lower()}_t{rec['hour']:03.0f}h"
    np.savez_compressed(
        out / f"{tag}_operator_only_se.npz",
        r_km=rec["r_km"], z_km=rec["z_km"], S_operator_equivalent=rec["forcing"],
        psi_operator_equivalent=rec["psi"], u_operator_equivalent=rec["u"],
        w_operator_equivalent=rec["w"], regularized=rec["changed"], raw_nonelliptic=rec["raw_bad"],
    )
    (out / f"{tag}_metrics.json").write_text(json.dumps(rec["metrics"], indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ctrl_by_hour = {}
    op_by_hour = {}
    records = []
    for spec in experiment_specs():
        hour = spec["hour"]
        if hour not in ctrl_by_hour:
            ctrl_by_hour[hour] = load_ctrl(hour, args)
            op_by_hour[hour] = build_ctrl_operator(ctrl_by_hour[hour], args)
        rec = collect_one(spec, ctrl_by_hour[hour], op_by_hour[hour], args)
        records.append(rec)
        save_record(rec, out)
        print(
            f"[DONE] {rec['case']} {hour:.0f} h: "
            f"OUT={rec['metrics']['upper_outflow_mean_m_s-1']:+.4f}, "
            f"IN={rec['metrics']['low_inflow_strength_change_m_s-1']:+.4f}, "
            f"Wcore={1000*rec['metrics']['core_ascent_mean_m_s-1']:+.3f} mm/s"
        )
    plot_comparison(records, out / "operator_only_se_full_1200km.png", (0.0, 1200.0), "full jet–TC domain")
    plot_comparison(records, out / "operator_only_se_inner_350km.png", (0.0, 350.0), "inner 350 km")
    fields = ["case", "hour"] + list(records[0]["metrics"])
    with (out / "operator_only_se_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow({"case": rec["case"], "hour": rec["hour"], **rec["metrics"]})
    summary = {
        "equation": "L_CTRL_regularized(psi_eq) = S_total_previous_operator_equivalent",
        "traditional_rhs": "zero",
        "density_correction_included": False,
        "second_order_proxy_included": False,
        "configuration": {
            "ctrl": args.ctrl, "f_s-1": args.f, "max_r_km": args.max_r_km,
            "max_z_km": args.max_z_km, "dr_km": args.dr_km, "eps_ratio": args.eps_ratio,
        },
        "records": [{"case": r["case"], "hour": r["hour"], **r["metrics"]} for r in records],
    }
    (out / "operator_only_se_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
