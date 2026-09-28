from pathlib import Path
import json
import sys

from netCDF4 import Dataset
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, ".")
from scripts.solve_matched_operator_forcing_outer100 import read_case

CTRL = "/data/zhangyx/DATA/cm1out_25N_CTRL_R2_Thompson_OML100_240h.nc"
JET = "/data/zhangyx/DATA/cm1out_25N_JET_R2_Thompson_OML100_240h.nc"
OUT = Path("output/r2_thompson_oml100_strength_matching")
JET_HOURS = [70.0, 75.0, 80.0, 85.0, 90.0]
CTRL_HOURS = np.arange(80.0, 140.01, 1.0)


class Args:
    max_r_km = 1200.0
    dr_km = 12.0
    max_z_km = 20.0
    f = 6.1636e-5
    output_dir = str(OUT)


def pmin(path):
    with Dataset(path) as dataset:
        time = np.asarray(dataset["time"][:], float) / 3600.0
        pressure = np.array([
            np.nanmin(gaussian_filter(np.asarray(dataset["psfc"][i], float), 2.0))
            for i in range(time.size)
        ])
    if np.nanmedian(pressure) > 1.0e4:
        pressure /= 100.0
    return time, pressure


def metrics(path, hour):
    avg = read_case(path, hour, Args())
    r = np.asarray(avg["r_km"], float)
    z = np.asarray(avg["z_km"], float)
    vt = np.asarray(avg["ut"], float)
    low = z <= 2.0
    profile = np.nanmax(vt[low], axis=0)
    ir = int(np.nanargmax(profile))
    return {"hour": float(hour), "vmax_ms": float(profile[ir]), "rmw_km": float(r[ir])}


def value_at(time, values, hour):
    return float(np.interp(hour, time, values))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tc, pc = pmin(CTRL)
    tj, pj = pmin(JET)
    candidates = []
    all_records = []
    for jhour in JET_HOURS:
        jp = value_at(tj, pj, jhour)
        ranked = sorted(CTRL_HOURS, key=lambda hour: abs(value_at(tc, pc, hour) - jp))[:12]
        jmet = metrics(JET, jhour)
        for chour in ranked:
            cmet = metrics(CTRL, float(chour))
            dp = jp - value_at(tc, pc, float(chour))
            dv = jmet["vmax_ms"] - cmet["vmax_ms"]
            dr = jmet["rmw_km"] - cmet["rmw_km"]
            cost = (dp / 2.0) ** 2 + (dv / 2.0) ** 2 + (dr / 12.0) ** 2
            record = {
                "jet_h": jhour, "ctrl_h": float(chour), "dP": dp,
                "dV": dv, "dR": dr, "cost": cost,
                "jet": {"pmin_hpa": jp, **jmet},
                "ctrl": {"pmin_hpa": value_at(tc, pc, float(chour)), **cmet},
            }
            all_records.append(record)
        candidates.append(min((x for x in all_records if x["jet_h"] == jhour), key=lambda x: x["cost"]))
    (OUT / "all_candidates.json").write_text(json.dumps(all_records, indent=2))
    (OUT / "selected_matches.json").write_text(json.dumps(candidates, indent=2))
    print(json.dumps(candidates, indent=2))


if __name__ == "__main__":
    main()
