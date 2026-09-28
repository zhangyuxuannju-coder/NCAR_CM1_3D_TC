#!/usr/bin/env python3
"""Check mapped CM1 theta and tangential-wind budgets on a fixed polar frame.

Each check holds the polar origin and basis fixed at the central output time.
It therefore validates Cartesian-to-polar field/source mapping in an Eulerian
frame.  It is not yet the moving-centre tendency needed for intensity-change
attribution.
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
from netCDF4 import Dataset
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.coordinates import destagger_to_scalar_grid
from src.tc_cylindrical import azimuthal_mean, regular_polar_geometry, sample_scalar_to_polar, sample_wind_to_polar


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/thompson_240h_se_attribution.json")
    p.add_argument("--hours", type=float, nargs="+", default=None)
    p.add_argument("--half-window-hours", type=int, default=1,
                   help="Centred finite-difference half-window in whole output hours.")
    return p


def scalar(ds: Dataset, name: str, it: int, z_keep: np.ndarray) -> np.ndarray:
    var = ds.variables[name]
    out, dims = destagger_to_scalar_grid(np.asarray(var[it], dtype=np.float64), list(var.dimensions[1:]))
    if tuple(dims) != ("zh", "yh", "xh"):
        raise ValueError(f"{name} is {dims} after destagger; expected scalar-grid z/y/x")
    return out[z_keep]


def mean_scalar(ds: Dataset, name: str, it: int, z_keep: np.ndarray, geometry: dict) -> np.ndarray:
    return azimuthal_mean(sample_scalar_to_polar(scalar(ds, name, it, z_keep), geometry))[0]


def mean_tangential(ds: Dataset, u_name: str, v_name: str, it: int, z_keep: np.ndarray, geometry: dict) -> np.ndarray:
    _, vt = sample_wind_to_polar(scalar(ds, u_name, it, z_keep), scalar(ds, v_name, it, z_keep), geometry)
    return azimuthal_mean(vt)[0]


def centre_lookup(path: Path) -> dict[tuple[str, int], tuple[float, float]]:
    with path.open(encoding="utf-8") as handle:
        return {(row["case"], int(round(float(row["time_h"])) )): (float(row["center_x_km"]), float(row["center_y_km"])) for row in csv.DictReader(handle)}


def weighted_metrics(residual: np.ndarray, components: list[np.ndarray], r_m: np.ndarray, z_m: np.ndarray,
                     r_bounds_km: tuple[float, float], z_bounds_km: tuple[float, float]) -> dict:
    dr = np.gradient(r_m)
    dz = np.gradient(z_m)
    weight = (2.0 * np.pi * r_m[None, :] * dr[None, :] * dz[:, None])
    region = ((r_m[None, :] >= r_bounds_km[0] * 1000.0) & (r_m[None, :] <= r_bounds_km[1] * 1000.0) &
              (z_m[:, None] >= z_bounds_km[0] * 1000.0) & (z_m[:, None] <= z_bounds_km[1] * 1000.0))
    weight = np.where(region, weight, 0.0)
    numerator = float(np.sqrt(np.sum(weight * residual**2) / np.sum(weight)))
    denominator = float(sum(np.sqrt(np.sum(weight * item**2) / np.sum(weight)) for item in components))
    return {"r_bounds_km": list(r_bounds_km), "z_bounds_km": list(z_bounds_km), "residual_rms": numerator,
            "component_rms_sum": denominator, "normalized_residual": numerator / max(denominator, 1e-30)}


def regional_metrics(residual: np.ndarray, components: list[np.ndarray], r_m: np.ndarray, z_m: np.ndarray) -> dict:
    regions = {
        "full": ((0.0, 1200.0), (0.0, 24.0)),
        "inner": ((0.0, 300.0), (0.0, 24.0)),
        "inner_free_troposphere": ((0.0, 300.0), (2.0, 12.0)),
        "upper_outflow": ((100.0, 300.0), (10.0, 17.0)),
        "outer": ((300.0, 1200.0), (0.0, 24.0)),
    }
    return {name: weighted_metrics(residual, components, r_m, z_m, r_bounds, z_bounds)
            for name, (r_bounds, z_bounds) in regions.items()}


def one_case(ds: Dataset, case: str, hour: float, centre: tuple[float, float], config: dict,
             half_window_hours: int) -> dict:
    time_h = np.asarray(ds.variables["time"][:], dtype=float) / 3600.0
    it = int(np.argmin(np.abs(time_h - hour)))
    offset = int(half_window_hours)
    if offset < 1:
        raise ValueError("half-window-hours must be at least one")
    if it < offset or it + offset >= len(time_h) or not np.isclose(time_h[it], hour):
        raise ValueError(f"{case} {hour:g} h cannot use a centred hourly difference")
    dt_s = (time_h[it + offset] - time_h[it - offset]) * 1800.0
    xh_m = np.asarray(ds.variables["xh"][:], dtype=float) * 1000.0
    yh_m = np.asarray(ds.variables["yh"][:], dtype=float) * 1000.0
    zh_m = np.asarray(ds.variables["zh"][:], dtype=float) * 1000.0
    z_keep = zh_m <= float(config["se"]["top_km"]) * 1000.0
    z_m = zh_m[z_keep]
    dr_m = float(config["se"]["radial_spacing_km"]) * 1000.0
    r_m = np.arange(0.5 * dr_m, float(config["se"]["radii_km"][1]) * 1000.0, dr_m)
    geometry = regular_polar_geometry(xh_m, yh_m, centre[0] * 1000.0, centre[1] * 1000.0, r_m, 360)

    theta_before = mean_scalar(ds, "th", it - offset, z_keep, geometry)
    theta_after = mean_scalar(ds, "th", it + offset, z_keep, geometry)
    theta_tendency = (theta_after - theta_before) / dt_s
    theta_pert_before = mean_scalar(ds, "thpert", it - offset, z_keep, geometry)
    theta_pert_after = mean_scalar(ds, "thpert", it + offset, z_keep, geometry)
    theta_pert_tendency = (theta_pert_after - theta_pert_before) / dt_s
    source_indices = np.arange(it - offset, it + offset + 1, dtype=int)
    source_time_s = time_h[source_indices] * 3600.0
    thermal_terms = {}
    for name in ds.variables:
        if name.startswith("ptb_"):
            samples = np.stack([mean_scalar(ds, name, int(sample), z_keep, geometry) for sample in source_indices])
            thermal_terms[name] = np.trapezoid(samples, x=source_time_s, axis=0) / (source_time_s[-1] - source_time_s[0])
    thermal_sum = sum(thermal_terms.values(), np.zeros_like(theta_tendency))
    thermal_residual = theta_tendency - thermal_sum
    thermal_pert_residual = theta_pert_tendency - thermal_sum

    vt_before = mean_tangential(ds, "u", "v", it - offset, z_keep, geometry)
    vt_after = mean_tangential(ds, "u", "v", it + offset, z_keep, geometry)
    vt_tendency = (vt_after - vt_before) / dt_s
    momentum_terms = {}
    suffixes = sorted({name[3:] for name in ds.variables if name.startswith("ub_") and f"vb_{name[3:]}" in ds.variables})
    for suffix in suffixes:
        samples = np.stack([
            mean_tangential(ds, f"ub_{suffix}", f"vb_{suffix}", int(sample), z_keep, geometry)
            for sample in source_indices
        ])
        momentum_terms[suffix] = np.trapezoid(samples, x=source_time_s, axis=0) / (source_time_s[-1] - source_time_s[0])
    momentum_sum = sum(momentum_terms.values(), np.zeros_like(vt_tendency))
    momentum_residual = vt_tendency - momentum_sum
    return {
        "arrays": {
            "r_m": r_m, "z_m": z_m, "theta_tendency": theta_tendency, "theta_budget_sum": thermal_sum,
            "theta_residual": thermal_residual, "theta_pert_tendency": theta_pert_tendency,
            "theta_pert_residual": thermal_pert_residual, "vt_tendency": vt_tendency, "vt_budget_sum": momentum_sum,
            "vt_residual": momentum_residual, **{f"thermal_{name}": value for name, value in thermal_terms.items()},
            **{f"momentum_{name}": value for name, value in momentum_terms.items()},
        },
        "summary": {
            "case": case, "hour": hour, "time_index": it, "centre_km": list(centre), "dt_s": dt_s,
            "half_window_hours": offset,
            "thermal": regional_metrics(thermal_residual, list(thermal_terms.values()), r_m, z_m),
            "thermal_perturbation_check": regional_metrics(thermal_pert_residual, list(thermal_terms.values()), r_m, z_m),
            "tangential_momentum": regional_metrics(momentum_residual, list(momentum_terms.values()), r_m, z_m),
            "thermal_terms": sorted(thermal_terms), "tangential_momentum_terms": suffixes,
            "frame": "fixed polar origin and basis at central time; budget terms trapezoid-averaged over the tendency window",
        },
    }


def plot(record: dict, path: Path) -> None:
    a = record["arrays"]
    r, z = a["r_m"] / 1000.0, a["z_m"] / 1000.0
    fields = [(a["theta_tendency"], "theta tendency (K s$^{-1}$)"), (a["theta_residual"], "theta residual (K s$^{-1}$)"),
              (a["vt_tendency"], "v_t tendency (m s$^{-2}$)"), (a["vt_residual"], "v_t residual (m s$^{-2}$)")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True, constrained_layout=True)
    for ax, (field, title) in zip(axes.ravel(), fields):
        lim = np.nanpercentile(np.abs(field), 99.0)
        image = ax.pcolormesh(r, z, field, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
        ax.set(title=title, xlim=(0, 300), ylim=(0, 24), xlabel="Radius (km)", ylabel="Height (km)")
        fig.colorbar(image, ax=ax)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parser().parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    hours = args.hours or list(config["windows"]["smoke_hours"])
    out = ROOT / config["output_dir"] / "budgets"
    out.mkdir(parents=True, exist_ok=True)
    centres = centre_lookup(ROOT / config["output_dir"] / "baseline" / "hourly_intensity_baseline.csv")
    summaries = []
    for case, input_path in config["cases"].items():
        with Dataset(input_path) as ds:
            for hour in hours:
                record = one_case(ds, case, hour, centres[(case, int(round(hour)))], config, args.half_window_hours)
                tag = f"{case}_t{hour:05.1f}h_hw{args.half_window_hours}".replace(".", "p")
                np.savez_compressed(out / f"{tag}_closure.npz", **record["arrays"])
                plot(record, out / f"{tag}_closure.png")
                summaries.append(record["summary"])
                print(json.dumps(record["summary"]), flush=True)
    passed = all(item["thermal"]["full"]["normalized_residual"] <= 0.2 and item["tangential_momentum"]["full"]["normalized_residual"] <= 0.2 for item in summaries)
    status = {"phase": "P2-closure", "passed_working_gate": passed, "gate": 0.2, "half_window_hours": args.half_window_hours, "records": summaries,
              "interpretation": "This validates fixed-frame mapped budget closure only; it is not moving-centre intensity attribution.",
              "next": "Inspect missing terms and residual patterns before applying a quantitative SE source ranking."}
    (out / f"closure_summary_hw{args.half_window_hours}.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps({"passed_working_gate": passed, "n_records": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
