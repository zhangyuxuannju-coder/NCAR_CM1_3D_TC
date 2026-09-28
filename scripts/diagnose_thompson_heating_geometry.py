#!/usr/bin/env python3
"""Describe the position and amount of available CM1 heating snapshots.

The CM1 ptb fields are output-step snapshots, so this script is a structural
comparison only.  It does not infer time-integrated diabatic forcing or a
closed SE heat budget.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/thompson_240h_se_attribution.json")
    p.add_argument("--hours", type=float, nargs="+", default=None)
    return p.parse_args()


def tag(hour: float) -> str:
    return f"t{hour:05.1f}h".replace(".", "p")


def cell_width(x: np.ndarray) -> np.ndarray:
    edges = np.empty(x.size + 1)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = max(0.0, x[0] - 0.5 * (x[1] - x[0]))
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    return np.diff(edges)


def centroid(q: np.ndarray, rho: np.ndarray, r_m: np.ndarray, z_m: np.ndarray, sign: str) -> dict[str, float]:
    rr, zz = np.meshgrid(r_m, z_m)
    volume = 2.0 * np.pi * rr * cell_width(r_m)[None, :] * cell_width(z_m)[:, None]
    field = np.maximum(q, 0.0) if sign == "positive" else np.maximum(-q, 0.0)
    valid = (rr <= 300_000.0) & (zz <= 20_000.0) & np.isfinite(field) & np.isfinite(rho)
    qvol = field * volume * valid
    qmass = qvol * np.maximum(rho, 0.0)
    def one(weight: np.ndarray) -> tuple[float, float, float]:
        total = float(np.sum(weight))
        if total <= 0:
            return float("nan"), float("nan"), total
        return float(np.sum(weight * rr) / total / 1000.0), float(np.sum(weight * zz) / total / 1000.0), total
    rv, zv, total_v = one(qvol)
    rm, zm, total_m = one(qmass)
    return {
        "r_centroid_volume_km": rv, "z_centroid_volume_km": zv, "integral_volume_K_m3_s-1": total_v,
        "r_centroid_mass_km": rm, "z_centroid_mass_km": zm, "integral_mass_K_kg_s-1": total_m,
        "peak_K_s-1": float(np.nanmax(field[valid])) if np.any(valid) else float("nan"),
    }


def load(cache: Path, case: str, hour: float) -> dict[str, np.ndarray]:
    with np.load(cache / f"{case}_{tag(hour)}.npz") as a:
        keys = ("r_m", "z_m", "rho", "ptb_mp", "ptb_rad", "Q_diabatic")
        missing = [key for key in keys if key not in a]
        if missing:
            raise KeyError(f"{case}, {hour:g} h lacks {missing}")
        return {key: np.asarray(a[key], dtype=float) for key in keys}


def plot(hour: float, ctrl: dict[str, np.ndarray], jet: dict[str, np.ndarray], out: Path) -> None:
    keys = ("ptb_mp", "ptb_rad", "Q_diabatic")
    labels = ("microphysics", "radiation", "diabatic total")
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), constrained_layout=True, sharex=True, sharey=True)
    r, z = ctrl["r_m"] / 1000.0, ctrl["z_m"] / 1000.0
    for row, (key, label) in enumerate(zip(keys, labels)):
        fields = (ctrl[key], jet[key], jet[key] - ctrl[key])
        finite = np.concatenate([np.abs(x[np.isfinite(x)]) for x in fields])
        vmax = max(float(np.percentile(finite, 99.0)) if finite.size else 0.0, 1.0e-12)
        for col, (field, title) in enumerate(zip(fields, ("CTRL", "JET", "JET - CTRL"))):
            im = axes[row, col].contourf(r, z, field, levels=np.linspace(-vmax, vmax, 25), cmap="RdBu_r", extend="both")
            axes[row, col].set(title=f"{label}: {title}", xlim=(0, 300), ylim=(0, 20), xlabel="Radius (km)")
            axes[row, col].set_ylabel("Height (km)")
            fig.colorbar(im, ax=axes[row, col], pad=0.02, label="K s$^{-1}$")
    fig.suptitle(f"Available CM1 heating snapshots, {hour:g} h", fontweight="bold")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    a = parse_args()
    config = json.loads((ROOT / a.config).read_text())
    hours = a.hours if a.hours is not None else config["windows"]["smoke_hours"]
    cache = ROOT / config["output_dir"] / "cache"
    out = ROOT / config["output_dir"] / "heating_geometry"
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for hour in hours:
        cases = {case: load(cache, case, hour) for case in ("CTRL", "JET")}
        for case, d in cases.items():
            for field in ("ptb_mp", "ptb_rad", "Q_diabatic"):
                for sign in ("positive", "negative"):
                    rows.append({"hour": hour, "case": case, "field": field, "sign": sign,
                                 **centroid(d[field], d["rho"], d["r_m"], d["z_m"], sign)})
        plot(hour, cases["CTRL"], cases["JET"], out / f"heating_geometry_{tag(hour)}.png")
    with (out / "heating_geometry.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    status = {
        "status": "STRUCTURAL_SNAPSHOT_ONLY",
        "field_definition": "Q_diabatic is available ptb_mp + ptb_rad; no unobserved processes are imputed",
        "integration_domain": "r <= 300 km, z <= 20 km; cylindrical volume and optional rho weighting",
        "restriction": "ptb output is an output-step snapshot, not an interval mean or a closed heat budget",
        "hours": hours,
    }
    (out / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
