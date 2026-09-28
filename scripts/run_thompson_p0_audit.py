#!/usr/bin/env python3
"""Read-only input and grid audit for the Thompson 240 h SE attribution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.se_nonuniform import solve_flux_form_dirichlet


REQUIRED = {
    "psfc": (("time", "yh", "xh"), "Pa"),
    "th": (("time", "zh", "yh", "xh"), "K"),
    "prs": (("time", "zh", "yh", "xh"), "Pa"),
    "rho": (("time", "zh", "yh", "xh"), "kg/m^3"),
    "u": (("time", "zh", "yh", "xf"), "m/s"),
    "v": (("time", "zh", "yf", "xh"), "m/s"),
    "w": (("time", "zf", "yh", "xh"), "m/s"),
}


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/thompson_240h_se_attribution.json")
    return parser.parse_args()


def variable_record(ds: Dataset, name: str) -> dict:
    if name not in ds.variables:
        return {"present": False}
    var = ds.variables[name]
    return {
        "present": True,
        "dimensions": list(var.dimensions),
        "units": str(getattr(var, "units", "")),
        "shape": list(var.shape),
    }


def case_audit(path: Path) -> dict:
    with Dataset(path) as ds:
        dimensions = {name: len(dim) for name, dim in ds.dimensions.items()}
        coords = {}
        for name in ("time", "xh", "yh", "zh", "xf", "yf", "zf"):
            arr = np.asarray(ds.variables[name][:], dtype=np.float64)
            coords[name] = {
                "values": arr.tolist(),
                "units": str(getattr(ds.variables[name], "units", "")),
                "strictly_increasing": bool(np.all(np.diff(arr) > 0.0)),
                "spacing_min": float(np.min(np.diff(arr))),
                "spacing_max": float(np.max(np.diff(arr))),
            }
        checked = {name: variable_record(ds, name) for name in REQUIRED}
        extra = {name: variable_record(ds, name) for name in (
            "ptb_mp", "ptb_div", "ptb_hidiff", "ptb_vidiff", "ptb_hturb", "ptb_vturb",
            "ptb_rdamp", "ub_hadv", "ub_vadv", "ub_hidiff", "ub_vturb", "ub_pgrad",
            "vb_hadv", "vb_vadv", "vb_hidiff", "vb_vturb", "vb_pgrad", "zhval",
        )}
        indices = sorted(set((0, len(ds.dimensions["time"]) // 4, len(ds.dimensions["time"]) // 2,
                              3 * len(ds.dimensions["time"]) // 4, len(ds.dimensions["time"]) - 1)))
        sample_finite = {}
        for name in ("psfc", "th", "prs", "rho", "u", "v", "w"):
            var = ds.variables[name]
            fractions = []
            for index in indices:
                arr = np.asarray(var[index], dtype=np.float64)
                fractions.append(float(np.isfinite(arr).mean()))
            sample_finite[name] = {"time_indices": indices, "finite_fraction": fractions}
        attrs = {
            name: (getattr(ds, name).item() if isinstance(getattr(ds, name), np.generic) else getattr(ds, name))
            for name in ("CM1 version", "fcor", "ptype", "oceanmodel", "nx", "ny", "nz")
            if name in ds.ncattrs()
        }
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "dimensions": dimensions,
        "coordinates": coords,
        "required_fields": checked,
        "budget_and_height_fields": extra,
        "sample_finite_fraction": sample_finite,
        "global_attributes": attrs,
    }


def field_contract(record: dict) -> list[str]:
    errors: list[str] = []
    for name, (dims, units) in REQUIRED.items():
        got = record["required_fields"][name]
        if not got["present"]:
            errors.append(f"missing required field {name}")
            continue
        if tuple(got["dimensions"]) != dims:
            errors.append(f"{name}: dimensions {got['dimensions']} expected {list(dims)}")
        if got["units"] != units:
            errors.append(f"{name}: units {got['units']!r} expected {units!r}")
    return errors


def grid_equivalence(ctrl: dict, jet: dict) -> list[str]:
    errors: list[str] = []
    for name in ("time", "xh", "yh", "zh", "xf", "yf", "zf"):
        left = np.asarray(ctrl["coordinates"][name]["values"], dtype=np.float64)
        right = np.asarray(jet["coordinates"][name]["values"], dtype=np.float64)
        if left.shape != right.shape or not np.array_equal(left, right):
            errors.append(f"coordinate mismatch: {name}")
    return errors


def numerical_smoke() -> dict:
    r = 1_000.0 + 300_000.0 * np.linspace(0.0, 1.0, 21) ** 1.3
    z = 50.0 + 20_000.0 * np.linspace(0.0, 1.0, 19) ** 1.2
    rr, zz = np.meshgrid(r, z)
    rhs = np.exp(-((rr - 100_000.0) / 60_000.0) ** 2 - ((zz - 8_000.0) / 3_000.0) ** 2)
    result = solve_flux_form_dirichlet(np.ones_like(rhs), np.zeros_like(rhs), np.ones_like(rhs), rhs, r, z)
    return {
        "operator": "non-uniform flux form, homogeneous Dirichlet finite-domain boundary",
        "relative_matrix_residual": result.relative_residual,
        "absolute_matrix_residual": result.absolute_residual,
        "rhs_l2": result.rhs_l2,
    }


def main() -> None:
    args = args_parser()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out = ROOT / config["output_dir"] / "audit"
    out.mkdir(parents=True, exist_ok=True)
    cases = {name: case_audit(Path(path)) for name, path in config["cases"].items()}
    errors = field_contract(cases["CTRL"]) + field_contract(cases["JET"]) + grid_equivalence(cases["CTRL"], cases["JET"])
    f_ctrl = float(cases["CTRL"]["global_attributes"].get("fcor", np.nan))
    f_cfg = float(config["se"]["f_s-1"])
    if not np.isclose(f_ctrl, f_cfg, rtol=0.0, atol=1.0e-10):
        errors.append(f"configured f={f_cfg} disagrees with CTRL metadata fcor={f_ctrl}")
    smoke = numerical_smoke()
    if smoke["relative_matrix_residual"] > 1.0e-8:
        errors.append("non-uniform SE numerical smoke residual exceeds 1e-8")
    payload = {
        "experiment_id": config["experiment_id"],
        "inputs": cases,
        "checks": {"errors": errors, "passed": not errors, "numerical_smoke": smoke},
        "limitations": [
            "This audit samples finite values at five times; it does not establish a full CM1 tendency closure.",
            "The selected SE boundary condition is a finite-domain balanced diagnostic assumption.",
            "No SST, mixed-layer temperature, or surface enthalpy-flux variable is present in the declared input contract."
        ],
    }
    (out / "audit_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    status = {"phase": "P0", "completed": ["input metadata", "coordinate equality", "field contract", "non-uniform solver smoke"],
              "passed": not errors, "failures": errors,
              "next": "Run P1 intensity and centre baseline before any batch SE attribution."}
    (out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
