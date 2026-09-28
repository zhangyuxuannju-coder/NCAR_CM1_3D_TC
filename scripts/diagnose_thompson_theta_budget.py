#!/usr/bin/env python3
"""TC-centred CM1 potential-temperature tendency budget."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from netCDF4 import Dataset
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.coordinates import destagger_to_scalar_grid
from src.tc_cylindrical import azimuthal_mean, regular_polar_geometry, sample_scalar_to_polar


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/thompson_theta_budget_72_78h.json")
    return p.parse_args()


def load_centres(path: Path, case_map: dict[str, str]) -> dict[tuple[str, float], tuple[float, float]]:
    reverse = {value: key for key, value in case_map.items()}
    out = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            case = reverse.get(row["case"])
            if case:
                out[(case, float(row["time_h"]))] = (float(row["center_x_km"]), float(row["center_y_km"]))
    return out


def index_of(time_h: np.ndarray, hour: float, case: str) -> int:
    i = int(np.argmin(np.abs(time_h - hour)))
    if not np.isclose(time_h[i], hour):
        raise ValueError(f"{case}: requested time {hour:g} h is absent")
    return i


def scalar(ds: Dataset, name: str, i: int) -> np.ndarray:
    var = ds.variables[name]
    value, dims = destagger_to_scalar_grid(np.asarray(var[i], dtype=np.float64), list(var.dimensions[1:]))
    if tuple(dims) != ("zh", "yh", "xh"):
        raise ValueError(f"{name}: got scalar dimensions {dims}")
    return value


def polar_mean(field: np.ndarray, geometry: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return azimuthal_mean(sample_scalar_to_polar(field, geometry))


def advection(theta: np.ndarray, u: np.ndarray, v: np.ndarray, w: np.ndarray,
              x_m: np.ndarray, y_m: np.ndarray, z_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = np.gradient(theta, x_m, axis=2, edge_order=1)
    dy = np.gradient(theta, y_m, axis=1, edge_order=1)
    dz = np.gradient(theta, z_m, axis=0, edge_order=1)
    return -u * dx - v * dy, -w * dz


def fd3(minus: np.ndarray, zero: np.ndarray, plus: np.ndarray,
        tm: float, t0: float, tp: float) -> tuple[np.ndarray, list[float]]:
    dtm, dtp = t0 - tm, tp - t0
    if dtm <= 0 or dtp <= 0:
        raise ValueError("finite-difference times must increase")
    weights = [-dtp / (dtm * (dtm + dtp)), (dtp - dtm) / (dtm * dtp),
               dtm / (dtp * (dtm + dtp))]
    return weights[0] * minus + weights[1] * zero + weights[2] * plus, weights


def map_sources(ds: Dataset, case: str, time_h: np.ndarray, hour: float,
                centre: tuple[float, float], radii_m: np.ndarray, z_keep: np.ndarray,
                x_m: np.ndarray, y_m: np.ndarray, z_m: np.ndarray, n_azimuth: int,
                source_terms: dict[str, list[str]]) -> tuple[dict[str, np.ndarray], float]:
    i = index_of(time_h, hour, case)
    geometry = regular_polar_geometry(x_m, y_m, centre[0] * 1000, centre[1] * 1000, radii_m, n_azimuth)
    theta, u, v, w = (scalar(ds, name, i)[z_keep] for name in ("th", "u", "v", "w"))
    hadv, vadv = advection(theta, u, v, w, x_m, y_m, z_m)
    fields = {}
    coverage = []
    for name, value in (("HADV", hadv), ("VADV", vadv)):
        fields[name], cov = polar_mean(value, geometry)
        coverage.append(cov)
    for group, names in source_terms.items():
        value = np.zeros_like(theta)
        for name in names:
            if name not in ds.variables:
                if group == "RDAMP_AUXILIARY":
                    continue
                raise KeyError(f"{case}: missing {name}")
            value += np.asarray(ds.variables[name][i], dtype=np.float64)[z_keep]
        fields[group], cov = polar_mean(value, geometry)
        coverage.append(cov)
    for name in ("ptb_hadv", "ptb_vadv", "ptb_hidiff", "ptb_vidiff", "ptb_hturb",
                 "ptb_vturb", "ptb_mp", "ptb_div", "ptb_rad", "ptb_rdamp"):
        if name in ds.variables:
            fields[f"CM1_{name[4:].upper()}"], cov = polar_mean(
                np.asarray(ds.variables[name][i], dtype=np.float64)[z_keep], geometry)
            coverage.append(cov)
    return fields, float(np.nanmin(np.stack(coverage)))


def map_fd(ds: Dataset, case: str, time_h: np.ndarray, hour: float,
           centres: dict[tuple[str, float], tuple[float, float]], radii_m: np.ndarray,
           z_keep: np.ndarray, x_m: np.ndarray, y_m: np.ndarray, z_m: np.ndarray,
           n_azimuth: int) -> tuple[np.ndarray, list[float]]:
    i = index_of(time_h, hour, case)
    if i == 0 or i + 1 >= time_h.size:
        raise ValueError(f"{case}: no centred neighbours at {hour:g} h")
    mapped = []
    for j in (i - 1, i, i + 1):
        sample_hour = float(time_h[j])
        theta = scalar(ds, "th", j)[z_keep]
        cx, cy = centres[(case, sample_hour)]
        geometry = regular_polar_geometry(x_m, y_m, cx * 1000, cy * 1000, radii_m, n_azimuth)
        mapped.append(polar_mean(theta, geometry)[0])
    return fd3(mapped[0], mapped[1], mapped[2], time_h[i - 1] * 3600,
               time_h[i] * 3600, time_h[i + 1] * 3600)


def time_mean(values: list[np.ndarray], time_s: np.ndarray) -> np.ndarray:
    return np.trapezoid(np.stack(values), x=time_s, axis=0) / (time_s[-1] - time_s[0])


def metrics(total: np.ndarray, budget: np.ndarray, residual: np.ndarray,
            r_m: np.ndarray, z_m: np.ndarray) -> tuple[dict, list[dict]]:
    weights = r_m[None, :] * np.gradient(r_m)[None, :] * np.gradient(z_m)[:, None]
    good = np.isfinite(total) & np.isfinite(budget) & np.isfinite(residual)
    weights = np.where(good, weights, 0.0)
    weights /= np.sum(weights)
    total_rms = np.sqrt(np.sum(weights * total ** 2))
    residual_rms = np.sqrt(np.sum(weights * residual ** 2))
    tm, bm = np.sum(weights * total), np.sum(weights * budget)
    covariance = np.sum(weights * (total - tm) * (budget - bm))
    corr_den = np.sqrt(np.sum(weights * (total - tm) ** 2) * np.sum(weights * (budget - bm) ** 2))
    summary = {
        "weighted_mean_unresolved_K_s": float(np.sum(weights * residual)),
        "weighted_mae_unresolved_K_s": float(np.sum(weights * np.abs(residual))),
        "weighted_rmse_unresolved_K_s": float(residual_rms),
        "max_abs_unresolved_K_s": float(np.nanmax(np.abs(residual))),
        "spatial_correlation_total_vs_budget": float(covariance / corr_den) if corr_den else None,
        "nrmse": float(residual_rms / max(total_rms, 1e-30)),
        "total_rms_K_s": float(total_rms),
    }
    by_height = []
    for k, height in enumerate(z_m / 1000):
        mask = np.isfinite(total[k]) & np.isfinite(budget[k])
        if np.any(mask):
            w = r_m[mask] * np.gradient(r_m)[mask]
            w /= np.sum(w)
            by_height.append({"height_km": float(height), "mae_K_s": float(np.sum(w * np.abs(residual[k, mask]))),
                              "total_rms_K_s": float(np.sqrt(np.sum(w * total[k, mask] ** 2)))})
    return summary, by_height


def limit(fields: list[np.ndarray], scale: float = 1.0) -> float:
    values = np.concatenate([np.abs(x[np.isfinite(x)]) for x in fields])
    return max(float(np.nanpercentile(values, 99)) * scale, 1e-12)


def plot_main(fields: dict[str, dict[str, np.ndarray]], r_km: np.ndarray, z_km: np.ndarray,
              path: Path, window_label: str) -> None:
    rows = [("TOTAL", "Total tendency (FD)"), ("ADV", "ADV = HADV + VADV"),
            ("TURB_DIFF", "TURB + DIFF"), ("MOIST", "MOIST = MP + moist divergence"),
            ("RADIATION", "RADIATION"), ("BUDGET_SUM", "Resolved budget sum"),
            ("UNRESOLVED", "UNRESOLVED / partial residual")]
    fig, axes = plt.subplots(len(rows), 3, figsize=(14, 24), sharex=True, sharey=True, constrained_layout=True)
    for row, (key, label) in enumerate(rows):
        main_lim = limit([fields["CTRL"][key], fields["JET30"][key]], 1.05)
        diff_lim = limit([fields["DELTA"][key]], 1.05)
        for col, case in enumerate(("CTRL", "JET30", "DELTA")):
            value = fields[case][key] * 1e3
            lim = (main_lim if case != "DELTA" else diff_lim) * 1e3
            image = axes[row, col].pcolormesh(r_km, z_km, value, cmap="RdBu_r",
                                              norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), shading="auto")
            try:
                axes[row, col].contour(r_km, z_km, value, levels=[0], colors="k", linewidths=0.35)
            except ValueError:
                pass
            axes[row, col].set(xlim=(0, 300), ylim=(0, 18))
            if row == 0:
                axes[row, col].set_title(("CTRL", "JET30", "JET30 − CTRL")[col])
            if col == 0:
                axes[row, col].set_ylabel(label + "\nHeight (km)")
            if row == len(rows) - 1:
                axes[row, col].set_xlabel("Radial distance (km)")
            fig.colorbar(image, ax=axes[row, col], fraction=0.046, pad=0.02, label=r"$10^{-3}$ K s$^{-1}$")
    fig.suptitle(f"Potential-temperature tendency budget, {window_label} mean", fontsize=16)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_closure(fields: dict[str, dict[str, np.ndarray]], r_km: np.ndarray, z_km: np.ndarray,
                 path: Path, window_label: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharex=True, sharey=True, constrained_layout=True)
    for ax, case in zip(axes, ("CTRL", "JET30", "DELTA")):
        value = fields[case]["UNRESOLVED"] * 1e3
        lim = limit([fields[case]["UNRESOLVED"]], 1.05) * 1e3
        image = ax.pcolormesh(r_km, z_km, value, cmap="RdBu_r",
                              norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), shading="auto")
        ax.contour(r_km, z_km, value, levels=[0], colors="k", linewidths=0.35)
        ax.set(title=case, xlabel="Radial distance (km)", ylabel="Height (km)", xlim=(0, 300), ylim=(0, 18))
        fig.colorbar(image, ax=ax, label=r"$10^{-3}$ K s$^{-1}")
    fig.suptitle(f"Partial potential-temperature budget closure, {window_label} mean")
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    cli = args()
    config = json.loads((ROOT / cli.config).read_text(encoding="utf-8"))
    out = ROOT / config["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    start, end = map(float, config["time_window_h"])
    hours = np.arange(start, end + 0.1, 1.0)
    centres = load_centres(ROOT / config["center_file"], config["center_case_map"])
    need = sorted({float(hour + offset) for hour in hours for offset in (-1, 0, 1)})
    for case in config["cases"]:
        missing = [hour for hour in need if (case, hour) not in centres]
        if missing:
            raise ValueError(f"{case}: missing centre records {missing}")
    radius_km = np.arange(config["radial_spacing_km"] / 2, config["radial_max_km"], config["radial_spacing_km"])
    radii_m = radius_km * 1000
    all_fields, coverage, fd_weights = {}, {}, {}
    z_km = None
    for case, input_path in config["cases"].items():
        mapped = {}
        fd_values = []
        with Dataset(input_path) as ds:
            time_h = np.asarray(ds.variables["time"][:], dtype=float) / 3600
            x_m = np.asarray(ds.variables["xh"][:], dtype=float) * 1000
            y_m = np.asarray(ds.variables["yh"][:], dtype=float) * 1000
            z_all = np.asarray(ds.variables["zh"][:], dtype=float) * 1000
            keep = z_all <= config["top_km"] * 1000
            z_m = z_all[keep]
            if z_km is None:
                z_km = z_m / 1000
            for hour in hours:
                source, cov = map_sources(ds, case, time_h, float(hour), centres[(case, float(hour))],
                                          radii_m, keep, x_m, y_m, z_m, config["n_azimuth"], config["source_terms"])
                for name, value in source.items():
                    mapped.setdefault(name, []).append(value)
                fd, weights = map_fd(ds, case, time_h, float(hour), centres, radii_m, keep,
                                     x_m, y_m, z_m, config["n_azimuth"])
                fd_values.append(fd)
                fd_weights[case] = weights
                coverage[case] = min(coverage.get(case, 1.0), cov)
        time_s = hours * 3600
        averaged = {name: time_mean(values, time_s) for name, values in mapped.items()}
        averaged["TOTAL"] = time_mean(fd_values, time_s)
        averaged["ADV"] = averaged["HADV"] + averaged["VADV"]
        averaged["TURB_DIFF"] = averaged["TURB"] + averaged["DIFF"]
        averaged["BUDGET_SUM"] = averaged["ADV"] + averaged["TURB_DIFF"] + averaged["MOIST"] + averaged["RADIATION"]
        averaged["UNRESOLVED"] = averaged["TOTAL"] - averaged["BUDGET_SUM"]
        all_fields[case] = averaged
    all_fields["DELTA"] = {key: all_fields["JET30"][key] - all_fields["CTRL"][key] for key in all_fields["CTRL"]}
    if z_km is None:
        raise ValueError("no height grid")
    names = sorted(all_fields["CTRL"])
    np.savez_compressed(out / "theta_budget_fields.npz", radius_km=radius_km, height_km=z_km,
                        **{f"{case}_{name}": all_fields[case][name]
                           for case in ("CTRL", "JET30", "DELTA") for name in names})
    mapping = {
        "th": {"definition": "total potential temperature state", "units": "K", "grid": "zh,yh,xh"},
        "u": {"definition": "x velocity, destaggered xf->xh", "units": "m s-1"},
        "v": {"definition": "y velocity, destaggered yf->yh", "units": "m s-1"},
        "w": {"definition": "vertical velocity, destaggered zf->zh", "units": "m s-1"},
        "HADV": {"definition": "-u*dtheta/dx-v*dtheta/dy on Cartesian scalar grid", "units": "K s-1"},
        "VADV": {"definition": "-w*dtheta/dz on Cartesian scalar grid", "units": "K s-1"},
        "TURB": {"variables": ["ptb_hturb", "ptb_vturb"], "units": "K s-1"},
        "DIFF": {"variables": ["ptb_hidiff", "ptb_vidiff"], "units": "K s-1"},
        "MOIST": {"variables": ["ptb_mp", "ptb_div"], "definition": "microphysics plus CM1 moist-divergence term", "units": "K s-1"},
        "RADIATION": {"variables": ["ptb_rad"], "units": "K s-1"},
        "DISS": {"variables": [], "definition": "no validated dissipative-heating field; unresolved", "units": "K s-1"},
        "RDAMP_AUXILIARY": {"variables": ["ptb_rdamp"], "definition": "Rayleigh damper, not DISS", "units": "K s-1"},
    }
    (out / "variable_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    with (out / "used_times_and_centres.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "time_h", "center_x_km", "center_y_km", "fd_minus_h", "fd_plus_h"])
        for case in config["cases"]:
            for hour in hours:
                cx, cy = centres[(case, float(hour))]
                writer.writerow([case, hour, cx, cy, hour - 1, hour + 1])
    stats, by_height = {}, {}
    for case in ("CTRL", "JET30"):
        stats[case], by_height[case] = metrics(all_fields[case]["TOTAL"], all_fields[case]["BUDGET_SUM"],
                                                all_fields[case]["UNRESOLVED"], radii_m, z_km * 1000)
    with (out / "closure_by_height.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "height_km", "mae_K_s", "total_rms_K_s"])
        writer.writeheader()
        for case, rows in by_height.items():
            for row in rows:
                writer.writerow({"case": case, **row})
    figure_keys = ("TOTAL", "ADV", "TURB_DIFF", "MOIST", "RADIATION", "BUDGET_SUM", "UNRESOLVED")
    plot_fields = {case: {key: all_fields[case][key] for key in figure_keys} for case in all_fields}
    window_label = f"{hours[0]:g}–{hours[-1]:g} h"
    plot_main(plot_fields, radius_km, z_km, out / "theta_budget_radial_height.png", window_label)
    plot_closure(plot_fields, radius_km, z_km, out / "theta_budget_closure.png", window_label)
    summary = {
        "status": "completed_partial_budget", "config": config, "input_cases": config["cases"],
        "time_window_h": [float(hours[0]), float(hours[-1])],
        "finite_difference": f"three-point centred formula using actual time coordinate; reads {hours[0] - 1:g}-{hours[-1] + 1:g} h",
        "source_average": f"trapezoid over {hours[0]:g}-{hours[-1]:g} h at hourly output times",
        "radius_km": [float(radius_km[0]), float(radius_km[-1])], "height_km": [float(z_km[0]), float(z_km[-1])],
        "minimum_azimuthal_coverage": coverage, "fd_weights_seconds_by_case": fd_weights,
        "closure_statistics": stats,
        "combination_checks": {"ADV_equals_HADV_plus_VADV": True, "TURB_DIFF_equals_TURB_plus_DIFF": True,
                                "BUDGET_SUM_equals_resolved_terms": True},
        "limitations": ["DISS is unavailable and is not set to zero or replaced by ptb_rdamp.",
                         "UNRESOLVED includes missing DISS, unavailable processes, output timing, and numerical/interpolation error.",
                         "ptb_* are hourly CM1 budget snapshots, not exact hourly model-step integrals.",
                         "JET30-CTRL differences are descriptive, not causal attribution."],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    (out / "README.md").write_text(f"""# CTRL-JET30 potential-temperature budget, {window_label}\n\nThis product diagnoses only the TC-centred potential-temperature tendency. It does not calculate surface-pressure tendency, project theta terms to pressure, or make a causal attribution.\n\n- `th` is total potential temperature; `u/v/w` are destaggered with the existing project helper.\n- HADV and VADV are calculated on the Cartesian scalar grid before polar sampling.\n- `TURB=ptb_hturb+ptb_vturb`; `DIFF=ptb_hidiff+ptb_vidiff`; `MOIST=ptb_mp+ptb_div`; `RADIATION=ptb_rad`.\n- `ptb_rdamp` is retained as an auxiliary field and is not DISS. No validated DISS output exists.\n- `UNRESOLVED = Total FD - (ADV + TURB_DIFF + MOIST + RADIATION)`. It includes missing DISS, unavailable terms, output timing, and numerical/interpolation error.\n- State FD uses actual times and {hours[0] - 1:g}-{hours[-1] + 1:g} h neighbours; source fields are trapezoid-averaged over {hours[0]:g}-{hours[-1]:g} h.\n- The existing Gaussian-smoothed psfc-minimum centre track, 3-km radial samples, native heights through 18 km, and 360 azimuths are used. No diagnostic smoothing is applied.\n\nProducts: `theta_budget_radial_height.png/pdf`, `theta_budget_closure.png/pdf`, `theta_budget_fields.npz`, `variable_mapping.json`, `used_times_and_centres.csv`, `closure_by_height.csv`, and `summary.json`.\n\nReproduce with `/data1/home/zhangyx/miniconda3/envs/cm1_tc/bin/python scripts/diagnose_thompson_theta_budget.py --config config/thompson_theta_budget_80h.json`.\n""", encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "status": summary["status"], "closure_statistics": stats}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
