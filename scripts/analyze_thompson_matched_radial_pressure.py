#!/usr/bin/env python3
"""Strength/RMW-matched radial-circulation and column-mass diagnostics."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
PRODUCT = ROOT / "output" / "thompson_240h_se_attribution"
OUT = PRODUCT / "radial_pressure_matched"
G = 9.81
def cache_file(case, hour):
    tag = f"{hour:05.1f}".replace(".", "p")
    return PRODUCT / "cache" / f"{case}_t{tag}h.npz"
def volume_mean(field, mask, r_m, z_m):
    rr = np.broadcast_to(r_m[None, :], field.shape)
    weights = rr*np.gradient(r_m)[None, :]*np.gradient(z_m)[:, None]
    use = mask & np.isfinite(field) & np.isfinite(weights)
    return float(np.sum(field[use]*weights[use])/np.sum(weights[use]))
def pressure_proxy(ur, rho, r_m, z_m, radius_km):
    j = int(np.argmin(np.abs(r_m-radius_km*1000.0)))
    column_flux = np.trapezoid(rho[:, j]*ur[:, j], z_m)
    return float(-2.0*G*column_flux/r_m[j]*3600.0/100.0)
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    matches = pd.read_csv(PRODUCT/"matching"/"strength_matched_pairs.csv")
    matches = matches[matches["accepted"].astype(bool)].copy()
    intensity = pd.read_csv(PRODUCT/"baseline"/"hourly_intensity_baseline.csv")
    rows, sections = [], []
    for _, pair in matches.iterrows():
        rec = {"jet_hour": pair.jet_hour, "ctrl_hour": pair.ctrl_hour,
               "delta_pressure_hpa": pair.delta_pressure_hpa, "delta_vt_m_s": pair.delta_vt_ms,
               "delta_rmw_km": pair.delta_rmw_km,
               "delta_vt_growth_6h_m_s_h": pair.delta_growth_6h_ms_per_h}
        loaded = {}
        for case, hour, rmw in (("JET", pair.jet_hour, pair.jet_rmw_km), ("CTRL", pair.ctrl_hour, pair.ctrl_rmw_km)):
            with np.load(cache_file(case, hour)) as data:
                d = {k: np.asarray(data[k]) for k in ("r_m", "z_m", "ur", "w", "rho")}
            loaded[case] = d
            rr, zz = np.meshgrid(d["r_m"], d["z_m"])
            masks = {"lowlevel_ur_m_s": (zz<=2000)&(rr<=150000),
                     "eyewall_w_m_s": (zz>=1000)&(zz<=12000)&(rr>=.75*rmw*1000)&(rr<=1.5*rmw*1000),
                     "upper_ur_m_s": (zz>=10000)&(zz<=17000)&(rr>=100000)&(rr<=300000)}
            for name, field in (("lowlevel_ur_m_s", d["ur"]), ("eyewall_w_m_s", d["w"]), ("upper_ur_m_s", d["ur"])):
                rec[f"{case.lower()}_{name}"] = volume_mean(field, masks[name], d["r_m"], d["z_m"])
            for radius in (50, 100, 150):
                rec[f"{case.lower()}_column_dpdt_proxy_r{radius}_hpa_h"] = pressure_proxy(d["ur"], d["rho"], d["r_m"], d["z_m"], radius)
            ii = intensity[(intensity.case==case)&np.isclose(intensity.time_h, hour)].iloc[0]
            rec[f"{case.lower()}_observed_dpdt_6h_hpa_h"] = ii.psfc_center_hpa_tendency_6h
        for name in ("lowlevel_ur_m_s", "eyewall_w_m_s", "upper_ur_m_s", "column_dpdt_proxy_r50_hpa_h",
                     "column_dpdt_proxy_r100_hpa_h", "column_dpdt_proxy_r150_hpa_h", "observed_dpdt_6h_hpa_h"):
            rec[f"delta_{name}"] = rec[f"jet_{name}"]-rec[f"ctrl_{name}"]
        rows.append(rec); sections.append((pair, loaded["JET"]["ur"]-loaded["CTRL"]["ur"], loaded["JET"]))
    table = pd.DataFrame(rows); table.to_csv(OUT/"matched_radial_pressure_metrics.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.3), sharex=True, sharey=True, constrained_layout=True)
    vmax = np.nanpercentile(np.abs(np.concatenate([x[1].ravel() for x in sections])), 98)
    for ax, (pair, delta, grid) in zip(axes.flat, sections):
        mesh = ax.pcolormesh(grid["r_m"]/1000, grid["z_m"]/1000, delta, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        ax.axvspan(100, 300, color="k", alpha=.05); ax.axhspan(10, 17, color="k", alpha=.05)
        ax.set_title(f"JET {pair.jet_hour:.0f} h - CTRL {pair.ctrl_hour:.0f} h\n" f"dVt={pair.delta_vt_ms:+.2f} m/s, dRMW={pair.delta_rmw_km:+.0f} km")
        ax.set(xlim=(0,300), ylim=(0,20), xlabel="Radius (km)", ylabel="Height (km)")
    fig.colorbar(mesh, ax=axes, label="Radial wind difference (m s$^{-1}$); positive outward")
    fig.suptitle("Strength/RMW-matched radial-circulation difference", fontweight="bold")
    fig.savefig(OUT/"matched_radial_wind_structure.png", dpi=220); plt.close(fig)
    faster = table[table.delta_vt_growth_6h_m_s_h>0]
    summary = {"definitions": {"matching": "accepted phase-constrained matches; abs(delta Vt)<=2 m/s and abs(delta RMW)<=6 km",
        "low_level": "volume-weighted ur, r<=150 km and z<=2 km; outward positive",
        "eyewall": "volume-weighted w, 0.75-1.5 RMW and 1-12 km",
        "upper_outflow": "volume-weighted ur, r=100-300 km and z=10-17 km",
        "pressure_proxy": "sidewall-only cylindrical column-mass flux; excludes top flux, moving-centre terms, and thermodynamic/hydrostatic redistribution"},
        "accepted_pair_count": int(len(table)), "faster_jet_pair_count": int(len(faster)),
        "faster_jet_mean_deltas": {k: float(faster[k].mean()) for k in ("delta_lowlevel_ur_m_s", "delta_eyewall_w_m_s", "delta_upper_ur_m_s", "delta_column_dpdt_proxy_r100_hpa_h", "delta_observed_dpdt_6h_hpa_h")},
        "sign_counts_all_pairs": {"stronger_inflow": int((table.delta_lowlevel_ur_m_s<0).sum()), "stronger_ascent": int((table.delta_eyewall_w_m_s>0).sum()), "stronger_upper_outflow": int((table.delta_upper_ur_m_s>0).sum())},
        "evidence_status": "descriptive matched-structure evidence; pressure proxy is not a closed pressure-tendency budget"}
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
if __name__ == "__main__": main()
