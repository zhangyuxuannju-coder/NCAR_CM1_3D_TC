#!/usr/bin/env python3
"""Build the reproducible hourly intensity and centre baseline for Thompson CTRL/JET."""

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
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.coordinates import destagger_axis
from src.jet_mechanism_diagnostics import cylindrical_wind, radial_bin_indices, radial_mean, storm_relative_geometry


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/thompson_240h_se_attribution.json")
    p.add_argument("--smooth-sigma-gridpoints", type=float, default=2.0)
    p.add_argument("--track-radius-km", type=float, default=180.0)
    p.add_argument("--radial-bin-km", type=float, default=3.0)
    p.add_argument("--max-r-km", type=float, default=300.0)
    return p


def _central_tendency(values: np.ndarray, time_h: np.ndarray, lag_h: float) -> np.ndarray:
    out = np.full(values.size, np.nan)
    for i, t in enumerate(time_h):
        before = np.where(np.isclose(time_h, t - lag_h))[0]
        after = np.where(np.isclose(time_h, t + lag_h))[0]
        if before.size and after.size:
            out[i] = (values[after[0]] - values[before[0]]) / (2.0 * lag_h)
    return out


def _choose_center(smoothed: np.ndarray, xh: np.ndarray, yh: np.ndarray, previous: tuple[float, float] | None,
                   track_radius_km: float) -> tuple[int, int, bool]:
    if previous is None:
        iy, ix = np.unravel_index(np.nanargmin(smoothed), smoothed.shape)
        return int(iy), int(ix), False
    xx, yy = np.meshgrid(xh, yh)
    local = (xx - previous[0]) ** 2 + (yy - previous[1]) ** 2 <= track_radius_km ** 2
    if np.any(local & np.isfinite(smoothed)):
        restricted = np.where(local, smoothed, np.inf)
        iy, ix = np.unravel_index(np.argmin(restricted), smoothed.shape)
        return int(iy), int(ix), False
    iy, ix = np.unravel_index(np.nanargmin(smoothed), smoothed.shape)
    return int(iy), int(ix), True


def diagnose_case(label: str, path: Path, sigma: float, track_radius_km: float,
                  radial_bin_km: float, max_r_km: float) -> tuple[list[dict], dict]:
    records: list[dict] = []
    with Dataset(path) as ds:
        time_h = np.asarray(ds.variables["time"][:], dtype=float) / 3600.0
        xh = np.asarray(ds.variables["xh"][:], dtype=float)
        yh = np.asarray(ds.variables["yh"][:], dtype=float)
        zh = np.asarray(ds.variables["zh"][:], dtype=float)
        psfc = ds.variables["psfc"]
        u_var = ds.variables["u"]
        v_var = ds.variables["v"]
        edges_m = np.arange(0.0, max_r_km * 1000.0 + radial_bin_km * 1000.0, radial_bin_km * 1000.0)
        centers: list[tuple[float, float]] = []
        fallbacks = 0
        previous: tuple[float, float] | None = None
        for it, hour in enumerate(time_h):
            p_raw = np.asarray(psfc[it], dtype=float)
            p_smooth = gaussian_filter(p_raw, sigma=sigma) if sigma > 0.0 else p_raw
            iy, ix, fallback = _choose_center(p_smooth, xh, yh, previous, track_radius_km)
            fallbacks += int(fallback)
            center_x, center_y = float(xh[ix]), float(yh[iy])
            previous = (center_x, center_y)
            centers.append(previous)
            u = destagger_axis(np.asarray(u_var[it, 0], dtype=float), axis=1)
            v = destagger_axis(np.asarray(v_var[it, 0], dtype=float), axis=0)
            geo = storm_relative_geometry(xh * 1000.0, yh * 1000.0, center_x * 1000.0, center_y * 1000.0)
            _, vt = cylindrical_wind(u, v, geo["cos_azimuth"], geo["sin_azimuth"])
            bin_index, valid = radial_bin_indices(geo["radius_m"], edges_m)
            vt_mean = radial_mean(vt, bin_index, valid, edges_m.size - 1)
            radii_km = 0.5 * (edges_m[:-1] + edges_m[1:]) / 1000.0
            rmw_index = int(np.nanargmax(vt_mean))
            records.append({
                "case": label,
                "time_index": int(it),
                "time_h": float(hour),
                "center_x_km": center_x,
                "center_y_km": center_y,
                "center_tracking_fallback": int(fallback),
                "psfc_center_hpa": float(p_raw[iy, ix] / 100.0),
                "psfc_smoothed_center_hpa": float(p_smooth[iy, ix] / 100.0),
                "psfc_domain_min_hpa": float(np.nanmin(p_raw) / 100.0),
                "vt0_axisymmetric_max_m_s": float(vt_mean[rmw_index]),
                "rmw0_km": float(radii_km[rmw_index]),
                "lowest_scalar_height_km": float(zh[0]),
            })
    for field in ("psfc_center_hpa", "vt0_axisymmetric_max_m_s"):
        values = np.array([r[field] for r in records])
        for lag in (6.0, 12.0, 24.0):
            tendency = _central_tendency(values, time_h, lag)
            for record, value in zip(records, tendency):
                record[f"{field}_tendency_{int(lag)}h"] = float(value)
    summary = {
        "case": label,
        "input": str(path),
        "n_times": len(records),
        "time_range_h": [float(time_h[0]), float(time_h[-1])],
        "tracking": {"method": "Gaussian-smoothed psfc local minimum", "sigma_gridpoints": sigma,
                     "radius_km": track_radius_km, "fallback_count": fallbacks},
        "wind_definition": "lowest scalar height, azimuthal mean, cyclonic-positive tangential wind",
        "lowest_scalar_height_km": float(zh[0]),
    }
    return records, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], out: Path) -> None:
    by_case = {case: [row for row in rows if row["case"] == case] for case in ("CTRL", "JET")}
    colours = {"CTRL": "#2166ac", "JET": "#b2182b"}
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    for case, case_rows in by_case.items():
        t = np.array([row["time_h"] for row in case_rows])
        axes[0].plot(t, [row["psfc_center_hpa"] for row in case_rows], color=colours[case], label=case, lw=2.4)
        axes[1].plot(t, [row["vt0_axisymmetric_max_m_s"] for row in case_rows], color=colours[case], label=case, lw=2.4)
        axes[2].plot(t, [row["rmw0_km"] for row in case_rows], color=colours[case], label=case, lw=2.4)
    axes[0].set(ylabel="Centre psfc (hPa)", title="Thompson R2 OML100: hourly intensity baseline")
    axes[1].set(ylabel="Axisymmetric max v_t (m s$^{-1}$)")
    axes[2].set(ylabel="RMW (km)", xlabel="Time (h)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parser().parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out = ROOT / config["output_dir"] / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    summaries = []
    for case in ("CTRL", "JET"):
        rows, summary = diagnose_case(case, Path(config["cases"][case]), args.smooth_sigma_gridpoints,
                                      args.track_radius_km, args.radial_bin_km, args.max_r_km)
        all_rows.extend(rows)
        summaries.append(summary)
    write_csv(out / "hourly_intensity_baseline.csv", all_rows)
    plot(all_rows, out / "hourly_intensity_baseline.png")
    status = {
        "phase": "P1-baseline",
        "completed": ["hourly centre tracking", "centre pressure", "lowest-level axisymmetric tangential wind", "RMW"],
        "limitations": [
            "Centre tracking is pressure based; a low-level-vorticity-centre sensitivity remains required.",
            "The wind metric is at the lowest scalar model level and is not a 10-m wind.",
            "This baseline does not yet diagnose a tendency budget or an SE response."
        ],
        "next": "Use the raw hourly series to set phases, then cache TC-centred state and source fields for P2."
    }
    (out / "baseline_manifest.json").write_text(json.dumps({"config": config, "cases": summaries, "status": status}, indent=2), encoding="utf-8")
    (out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "cases": summaries, "status": status}, indent=2))


if __name__ == "__main__":
    main()
