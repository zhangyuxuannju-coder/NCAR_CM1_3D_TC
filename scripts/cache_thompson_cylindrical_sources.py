#!/usr/bin/env python3
"""Cache TC-centred equal-angle state and source fields for Thompson SE diagnostics.

This stage creates compact r-z products only.  It does not solve SE or claim a
budget closure: Cartesian tendency closure is a separate P2 check.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from netCDF4 import Dataset
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bui_forcing import assemble_bui_forcings
from src.coordinates import destagger_to_scalar_grid
from src.environmental_eddy import eddy_flux_divergence, eddy_scalar_flux_divergence
from src.tc_cylindrical import azimuthal_mean, regular_polar_geometry, sample_scalar_to_polar, sample_wind_to_polar


THERMAL = ("mp", "div", "hidiff", "vidiff", "hturb", "vturb", "rdamp", "rad")
MOMENTUM = ("hadv", "vadv", "hidiff", "vidiff", "hturb", "vturb", "pgrad", "rdamp", "cor")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/thompson_240h_se_attribution.json")
    p.add_argument("--hours", type=float, nargs="+", default=None)
    p.add_argument("--pairs", nargs="+", default=None,
                   help="strength-matched JET:CTRL hour pairs; caches only the specified case/time records")
    p.add_argument("--n-azimuth", type=int, default=360)
    p.add_argument("--coverage-floor", type=float, default=0.99)
    return p


def baseline_centres(path: Path) -> dict[tuple[str, int], tuple[float, float]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["case"], int(round(float(row["time_h"])))): (float(row["center_x_km"]), float(row["center_y_km"]))
        for row in rows
    }


def scalar_at_time(ds: Dataset, name: str, index: int, z_keep: np.ndarray) -> np.ndarray:
    var = ds.variables[name]
    raw = np.asarray(var[index], dtype=np.float64)
    field, dims = destagger_to_scalar_grid(raw, list(var.dimensions[1:]))
    if tuple(dims) != ("zh", "yh", "xh"):
        raise ValueError(f"{name} after destagger has dimensions {dims}, expected ['zh', 'yh', 'xh']")
    return field[z_keep]


def tangential_budget_at_time(ds: Dataset, suffix: str, index: int, z_keep: np.ndarray,
                              geometry: dict) -> tuple[np.ndarray, np.ndarray]:
    u_name, v_name = f"ub_{suffix}", f"vb_{suffix}"
    if u_name not in ds.variables or v_name not in ds.variables:
        raise KeyError(f"missing paired momentum budget fields {u_name}/{v_name}")
    u = scalar_at_time(ds, u_name, index, z_keep)
    v = scalar_at_time(ds, v_name, index, z_keep)
    _, tangential = sample_wind_to_polar(u, v, geometry)
    return azimuthal_mean(tangential)


def cache_one(case: str, input_path: Path, hour: float, centre: tuple[float, float], config: dict,
              out: Path, n_azimuth: int, coverage_floor: float) -> dict:
    with Dataset(input_path) as ds:
        time_h = np.asarray(ds.variables["time"][:], dtype=float) / 3600.0
        index = int(np.argmin(np.abs(time_h - hour)))
        if not np.isclose(time_h[index], hour, atol=1.0e-6):
            raise ValueError(f"requested {hour:g} h does not exist in {input_path}")
        xh_m = np.asarray(ds.variables["xh"][:], dtype=float) * 1000.0
        yh_m = np.asarray(ds.variables["yh"][:], dtype=float) * 1000.0
        zh_m = np.asarray(ds.variables["zh"][:], dtype=float) * 1000.0
        z_keep = zh_m <= float(config["se"]["top_km"]) * 1000.0
        z_m = zh_m[z_keep]
        dr_m = float(config["se"]["radial_spacing_km"]) * 1000.0
        r_upper_m = float(config["se"]["radii_km"][1]) * 1000.0
        r_m = np.arange(0.5 * dr_m, r_upper_m, dr_m)
        geometry = regular_polar_geometry(xh_m, yh_m, centre[0] * 1000.0, centre[1] * 1000.0, r_m, n_azimuth)

        u = scalar_at_time(ds, "u", index, z_keep)
        v = scalar_at_time(ds, "v", index, z_keep)
        ur_a, vt_a = sample_wind_to_polar(u, v, geometry)
        del u, v
        w_a = sample_scalar_to_polar(scalar_at_time(ds, "w", index, z_keep), geometry)
        theta_a = sample_scalar_to_polar(scalar_at_time(ds, "th", index, z_keep), geometry)
        rho_a = sample_scalar_to_polar(scalar_at_time(ds, "rho", index, z_keep), geometry)
        prs_a = sample_scalar_to_polar(scalar_at_time(ds, "prs", index, z_keep), geometry)
        ur, coverage = azimuthal_mean(ur_a)
        vt, coverage_vt = azimuthal_mean(vt_a)
        w, coverage_w = azimuthal_mean(w_a)
        theta, coverage_theta = azimuthal_mean(theta_a)
        rho, coverage_rho = azimuthal_mean(rho_a)
        prs, coverage_prs = azimuthal_mean(prs_a)
        coverage_min = float(np.nanmin(np.stack((coverage, coverage_vt, coverage_w, coverage_theta, coverage_rho, coverage_prs))))
        if coverage_min < coverage_floor:
            raise ValueError(f"equal-angle coverage {coverage_min:.3f} is below configured floor {coverage_floor:.3f}")

        ur_prime = ur_a - ur[:, None, :]
        vt_prime = vt_a - vt[:, None, :]
        w_prime = w_a - w[:, None, :]
        theta_prime = theta_a - theta[:, None, :]
        flux_ur_vt = np.nanmean(rho_a * ur_prime * vt_prime, axis=1)
        flux_w_vt = np.nanmean(rho_a * w_prime * vt_prime, axis=1)
        flux_ur_theta = np.nanmean(rho_a * ur_prime * theta_prime, axis=1)
        flux_w_theta = np.nanmean(rho_a * w_prime * theta_prime, axis=1)
        # A uniform or sheared environmental jet is non-axisymmetric in the
        # storm-centred frame.  Retain this state diagnostic separately from
        # the eddy torque: large non-axisymmetric wind need not imply a large
        # azimuthal-mean momentum-flux convergence.
        eddy_kinetic_energy = 0.5 * np.nanmean(ur_prime**2 + vt_prime**2 + w_prime**2, axis=1)
        eddy_momentum = eddy_flux_divergence(rho, flux_ur_vt, flux_w_vt, r_m, z_m)
        eddy_heat = eddy_scalar_flux_divergence(rho, flux_ur_theta, flux_w_theta, r_m, z_m)
        del ur_a, vt_a, w_a, theta_a, rho_a, prs_a

        thermal = {}
        for suffix in THERMAL:
            name = f"ptb_{suffix}"
            if name in ds.variables:
                thermal[suffix], source_coverage = azimuthal_mean(sample_scalar_to_polar(scalar_at_time(ds, name, index, z_keep), geometry))
                coverage_min = min(coverage_min, float(np.nanmin(source_coverage)))
        momentum = {}
        for suffix in MOMENTUM:
            try:
                momentum[suffix], source_coverage = tangential_budget_at_time(ds, suffix, index, z_keep, geometry)
                coverage_min = min(coverage_min, float(np.nanmin(source_coverage)))
            except KeyError:
                continue
        if coverage_min < coverage_floor:
            raise ValueError(f"source coverage {coverage_min:.3f} is below configured floor {coverage_floor:.3f}")
        sources = assemble_bui_forcings(eddy_heat["forcing"], eddy_momentum["forcing"], thermal, momentum)
        arrays = {
            "r_m": r_m, "z_m": z_m, "ur": ur, "vt": vt, "w": w, "theta": theta, "rho": rho, "prs": prs,
            "coverage": np.minimum.reduce((coverage, coverage_vt, coverage_w, coverage_theta, coverage_rho, coverage_prs)),
            "eddy_flux_ur_vt": flux_ur_vt, "eddy_flux_w_vt": flux_w_vt,
            "eddy_flux_ur_theta": flux_ur_theta, "eddy_flux_w_theta": flux_w_theta,
            "eddy_kinetic_energy": eddy_kinetic_energy,
            "F_lambda_eddy_radial": eddy_momentum["forcing_radial"], "F_lambda_eddy_vertical": eddy_momentum["forcing_vertical"],
            "Q_eddy_radial": eddy_heat["forcing_radial"], "Q_eddy_vertical": eddy_heat["forcing_vertical"],
            **sources,
            **{f"ptb_{key}": value for key, value in thermal.items()},
            **{f"F_lambda_budget_{key}": value for key, value in momentum.items()},
        }
    tag = f"{case}_t{hour:05.1f}h".replace(".", "p")
    np.savez_compressed(out / f"{tag}.npz", **arrays)
    return {
        "case": case, "hour": hour, "index": index, "centre_km": list(centre), "output": f"{tag}.npz",
        "coverage_min": coverage_min, "shape_zr": list(ur.shape), "available_thermal_terms": sorted(thermal),
        "available_tangential_budget_terms": sorted(momentum),
        "eddy_average": "equal-angle Reynolds mean, density retained inside covariance",
    }


def main() -> None:
    args = parser().parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    hours = list(args.hours) if args.hours is not None else list(config["windows"]["smoke_hours"])
    out = ROOT / config["output_dir"] / "cache"
    out.mkdir(parents=True, exist_ok=True)
    centres = baseline_centres(ROOT / config["output_dir"] / "baseline" / "hourly_intensity_baseline.csv")
    records = []
    requested: list[tuple[str, float]] = []
    if args.pairs:
        for item in args.pairs:
            jet_hour, ctrl_hour = (float(value) for value in item.split(":"))
            requested.extend((("JET", jet_hour), ("CTRL", ctrl_hour)))
    else:
        requested = [(case, hour) for case in config["cases"] for hour in hours]
    for case, hour in requested:
        path = config["cases"][case]
        key = (case, int(round(hour)))
        if key not in centres:
            raise KeyError(f"no P1 pressure centre for {case} at {hour:g} h")
        records.append(cache_one(case, Path(path), hour, centres[key], config, out, args.n_azimuth, args.coverage_floor))
        print(json.dumps(records[-1]), flush=True)
    manifest = {
        "config": config, "records": records,
        "field_identity": {
            "theta": "CM1 total potential temperature th, K",
            "rho": "CM1 dry-air density rho, kg m-3",
            "Q_total": "eddy heat-flux convergence plus available non-advective ptb terms, K s-1",
            "F_lambda_total": "eddy angular-momentum-flux convergence plus selected non-advective tangential tendencies, m s-2",
            "restriction": "This cache has not yet passed Cartesian or cylindrical tendency-budget closure."
        },
        "next": "Run P2 tendency closure before using these sources in a quantitative SE comparison."
    }
    (out / "cache_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "status.json").write_text(json.dumps({"phase": "P2-cache", "completed": ["state", "eddy fluxes", "model source terms"], "records": len(records), "next": manifest["next"]}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
