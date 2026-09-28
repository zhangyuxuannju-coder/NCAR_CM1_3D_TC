#!/usr/bin/env python3
"""Strength/RMW match R2 JET and CTRL within one explicitly bounded phase."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter

sys.path.insert(0, ".")
from scripts.solve_matched_operator_forcing_outer100 import read_case

CTRL = "/data/zhangyx/DATA/cm1out_25N_CTRL_R2_Thompson_OML100_240h.nc"
JET = "/data/zhangyx/DATA/cm1out_25N_JET_R2_Thompson_OML100_240h.nc"


class Args:
    max_r_km = 1200.0
    dr_km = 12.0
    max_z_km = 20.0
    f = 6.1636e-5
    output_dir = "."


def pressure_series(path):
    with Dataset(path) as ds:
        time = np.asarray(ds["time"][:], float) / 3600.0
        values = np.array([np.nanmin(gaussian_filter(np.asarray(ds["psfc"][i], float), 2.0)) for i in range(time.size)])
    return time, values / 100.0 if np.nanmedian(values) > 1e4 else values


def tc_metrics(path, hour):
    avg = read_case(path, hour, Args())
    r, z, vt = np.asarray(avg["r_km"], float), np.asarray(avg["z_km"], float), np.asarray(avg["ut"], float)
    low_level = np.nanmax(vt[z <= 2.0], axis=0)
    index = int(np.nanargmax(low_level))
    return {"vmax_ms": float(low_level[index]), "rmw_km": float(r[index])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--step", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.end < args.start or args.step <= 0:
        raise ValueError("Require end >= start and a positive step")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jhours = np.arange(args.start, args.end + 0.01, args.step)
    chours = np.arange(args.start, args.end + 0.01, 1.0)
    tc, pc = pressure_series(CTRL)
    tj, pj = pressure_series(JET)
    selected, all_candidates = [], []
    for jhour in jhours:
        jp = float(np.interp(jhour, tj, pj)); jm = tc_metrics(JET, jhour)
        # pressure ranking limits expensive cylindrical reads, but the final
        # choice uses pressure, low-level Vmax, and RMW jointly.
        shortlist = sorted(chours, key=lambda x: abs(float(np.interp(x, tc, pc)) - jp))[:12]
        group = []
        for chour in shortlist:
            cp = float(np.interp(chour, tc, pc)); cm = tc_metrics(CTRL, float(chour))
            dp, dv, dr = jp-cp, jm["vmax_ms"]-cm["vmax_ms"], jm["rmw_km"]-cm["rmw_km"]
            rec = {"jet_h":float(jhour), "ctrl_h":float(chour), "dP":dp, "dV":dv, "dR":dr,
                   "cost":(dp/2)**2+(dv/2)**2+(dr/12)**2,
                   "jet":{"pmin_hpa":jp, **jm}, "ctrl":{"pmin_hpa":cp, **cm}}
            group.append(rec); all_candidates.append(rec)
        selected.append(min(group, key=lambda x:x["cost"]))
    metadata = {"matching_window_h":[args.start,args.end], "jet_sampling_interval_h":args.step,
                "cost":"(dP/2 hPa)^2 + (dV/2 m s-1)^2 + (dRMW/12 km)^2",
                "constraint":"JET and CTRL candidate times both restricted to this window"}
    (args.output_dir / "selected_matches.json").write_text(json.dumps(selected, indent=2))
    (args.output_dir / "all_candidates.json").write_text(json.dumps(all_candidates, indent=2))
    (args.output_dir / "matching_metadata.json").write_text(json.dumps(metadata, indent=2))
    for x in selected:
        print(f"J{x['jet_h']:05.1f}/C{x['ctrl_h']:05.1f}: dP={x['dP']:+.2f} hPa, dV={x['dV']:+.2f} m/s, dRMW={x['dR']:+.0f} km, cost={x['cost']:.3f}")


if __name__ == "__main__":
    main()
