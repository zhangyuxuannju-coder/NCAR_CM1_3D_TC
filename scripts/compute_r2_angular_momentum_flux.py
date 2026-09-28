#!/usr/bin/env python3
"""Control-surface absolute-angular-momentum fluxes for Thompson R2 JET/CTRL.

This deliberately evaluates the instantaneous, azimuthally averaged *total*
flux <rho u_r M_a> from the native Cartesian fields.  It is therefore not the
local advective term -u_r dM/dr and not a closed time-tendency budget.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np

ROOT = Path("output/jet_outflow_mechanism_validation")
OUT = ROOT / "angular_momentum_flux_control_surfaces"
FIG = OUT / "figures"
DATA = {
    "CTRL": Path("/data/zhangyx/DATA/cm1out_25N_CTRL_R2_Thompson_OML100_240h.nc"),
    "JET": Path("/data/zhangyx/DATA/cm1out_25N_JET_R2_Thompson_OML100_240h.nc"),
}
F_COR = 6.2e-5
DR_M = 5000.0
RMAX_M = 600000.0
LAYERS = {"low_0p5_5km": (500.0, 5000.0), "outflow_10_17km": (10000.0, 17000.0)}
CONTROLS = ("rmw15", "r050", "r100", "r200", "r300")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def lookup_state() -> dict[tuple[str, float], dict[str, float]]:
    fields = ("center_x_km", "center_y_km", "rmw_km", "vmax_ms", "pmin_hpa")
    rows = read_csv(ROOT / "timeseries.csv")
    return {(r["case"], float(r["time_h"])): {k: float(r[k]) for k in fields} for r in rows}


def destagger_to_scalar(raw: np.ndarray, dims: list[str]) -> np.ndarray:
    a = np.asarray(raw, dtype=np.float64)
    names = list(dims)
    for staggered, scalar in (("xf", "xh"), ("yf", "yh"), ("zf", "zh")):
        if staggered in names:
            axis = names.index(staggered)
            lo = [slice(None)] * a.ndim; hi = [slice(None)] * a.ndim
            lo[axis] = slice(0, -1); hi[axis] = slice(1, None)
            a = 0.5 * (a[tuple(lo)] + a[tuple(hi)])
            names[axis] = scalar
    if tuple(names) != ("zh", "yh", "xh"):
        raise ValueError(f"unexpected scalar dimensions {names}")
    return a


def ring_mean(field: np.ndarray, bins: np.ndarray, valid: np.ndarray, nr: int) -> np.ndarray:
    # Equal-angle grid cell area is uniform in the Cartesian model grid; each
    # ring average is consequently a conventional azimuthal average.
    flat = np.asarray(field, float).reshape(field.shape[0], -1)
    out = np.full((flat.shape[0], nr), np.nan)
    for iz, values in enumerate(flat):
        use = valid & np.isfinite(values)
        den = np.bincount(bins[use], minlength=nr)
        num = np.bincount(bins[use], weights=values[use], minlength=nr)
        out[iz] = np.divide(num, den, out=np.full(nr, np.nan), where=den > 0)
    return out


def layer_flux(profile: np.ndarray, z_m: np.ndarray, r_m: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    use = (z_m >= bounds[0]) & (z_m <= bounds[1])
    if use.sum() < 2:
        raise ValueError(f"insufficient levels for {bounds}")
    return 2.0 * np.pi * r_m * np.trapezoid(profile[use], x=z_m[use], axis=0)


def one_time(ds: Dataset, case: str, hour: float, meta: dict[str, float], geometry_cache: dict) -> list[dict[str, float]]:
    time_h = np.asarray(ds.variables["time"][:], float) / 3600.0
    it = int(np.argmin(abs(time_h - hour)))
    if not np.isclose(time_h[it], hour, atol=1e-5):
        raise ValueError(f"{case} {hour:g} h unavailable")
    x_m = geometry_cache["x_m"]; y_m = geometry_cache["y_m"]; z_all = geometry_cache["z_m"]
    zkeep = z_all <= 17000.0
    z_m = z_all[zkeep]
    u = destagger_to_scalar(np.asarray(ds.variables["u"][it], float), list(ds.variables["u"].dimensions[1:]))[zkeep]
    v = destagger_to_scalar(np.asarray(ds.variables["v"][it], float), list(ds.variables["v"].dimensions[1:]))[zkeep]
    rho = destagger_to_scalar(np.asarray(ds.variables["rho"][it], float), list(ds.variables["rho"].dimensions[1:]))[zkeep]
    xx, yy = np.meshgrid(x_m, y_m)
    dx = xx - meta["center_x_km"] * 1000.0; dy = yy - meta["center_y_km"] * 1000.0
    radius = np.hypot(dx, dy); cosaz = np.divide(dx, radius, out=np.ones_like(dx), where=radius > 1.0); sinaz = np.divide(dy, radius, out=np.zeros_like(dy), where=radius > 1.0)
    edges = geometry_cache["edges"]; r_m = geometry_cache["r_m"]; nr = r_m.size
    bins = np.digitize(radius.ravel(), edges) - 1; valid = (bins >= 0) & (bins < nr)
    ur = u * cosaz + v * sinaz; vt = -u * sinaz + v * cosaz
    ma = vt * radius[None, :, :] + 0.5 * F_COR * radius[None, :, :] ** 2
    # Favre decomposition is exact after azimuthal averaging:
    # <rho ur Ma> = rhobar utilde Mtilde + rhobar <ur'' Ma''>_Favre.
    rhobar = ring_mean(rho, bins, valid, nr)
    rho_ur = ring_mean(rho * ur, bins, valid, nr)
    rho_ma = ring_mean(rho * ma, bins, valid, nr)
    total = ring_mean(rho * ur * ma, bins, valid, nr)
    mean = np.divide(rho_ur * rho_ma, rhobar, out=np.full_like(total, np.nan), where=rhobar > 0.0)
    eddy = total - mean
    rows: list[dict[str, float]] = []
    for layer, bounds in LAYERS.items():
        fluxes = {"total": layer_flux(total, z_m, r_m, bounds), "favre_mean": layer_flux(mean, z_m, r_m, bounds), "favre_eddy": layer_flux(eddy, z_m, r_m, bounds)}
        for control in CONTROLS:
            radius_km = 1.5 * meta["rmw_km"] if control == "rmw15" else float(control[1:])
            for component, profile in fluxes.items():
                outward = float(np.interp(radius_km * 1000.0, r_m, profile))
                rows.append({"case": case, "time_h": hour, "layer": layer, "control": control, "control_radius_km": radius_km, "component": component, "outward_flux_Nm": outward, "inward_flux_Nm": -outward, **meta})
    return rows


def load_requests() -> dict[str, set[float]]:
    # Same-time samples expose intensity-path evolution; all matched pair times
    # are included even when they do not coincide with the 5-h sampling.
    needed = {"JET": set(np.arange(50.0, 150.1, 5.0)), "CTRL": set(np.arange(50.0, 150.1, 5.0))}
    for phase in ("50_75", "80_150"):
        for row in read_csv(ROOT / f"matched_{phase}_pairs.csv"):
            needed["JET"].add(float(row["jet_h"])); needed["CTRL"].add(float(row["ctrl_h"]))
    return needed


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def by_key(rows: list[dict[str, float]]) -> dict[tuple, dict[str, float]]:
    return {(r["case"], r["time_h"], r["layer"], r["control"], r["component"]): r for r in rows}


def plot_same_time(rows: list[dict[str, float]]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, constrained_layout=True)
    scale = 1e15
    for ir, control in enumerate(("rmw15", "r100")):
        ax = axes[0, ir]
        for case, color in (("JET", "#D55E00"), ("CTRL", "#0072B2")):
            q = [r for r in rows if r["case"] == case and r["layer"] == "low_0p5_5km" and r["control"] == control and r["component"] == "total" and r["time_h"] % 5 == 0]
            q.sort(key=lambda x: x["time_h"])
            ax.plot([x["time_h"] for x in q], [x["inward_flux_Nm"] / scale for x in q], "o-", ms=3, color=color, label=case)
        radius_label = "1.5 RMW" if control == "rmw15" else "100 km"
        ax.axhline(0, color="0.25", lw=.8); ax.set_title(f"Total low-level inward $M_a$ flux: {radius_label}"); ax.legend(frameon=False)
        ax.set_ylabel(r"inward flux ($10^{15}$ N m; positive = import)")
        ax = axes[1, ir]
        for component, color in (("total", "k"), ("favre_mean", "#0072B2"), ("favre_eddy", "#D55E00")):
            j = {(r["time_h"]): r for r in rows if r["case"] == "JET" and r["layer"] == "low_0p5_5km" and r["control"] == control and r["component"] == component and r["time_h"] % 5 == 0}
            c = {(r["time_h"]): r for r in rows if r["case"] == "CTRL" and r["layer"] == "low_0p5_5km" and r["control"] == control and r["component"] == component and r["time_h"] % 5 == 0}
            hh = sorted(set(j) & set(c)); ax.plot(hh, [(j[h]["inward_flux_Nm"] - c[h]["inward_flux_Nm"]) / scale for h in hh], "o-", ms=3, color=color, label=component.replace("_", " "))
        ax.axhline(0, color="0.25", lw=.8); ax.set_title(f"JET − CTRL, {radius_label}"); ax.set_xlabel("Model time (h)"); ax.set_ylabel(r"difference ($10^{15}$ N m)"); ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Low-level absolute-angular-momentum flux through storm-centred control cylinders\ninstantaneous total flux; radial outward positive before sign reversal", fontsize=13)
    out = FIG / "same_time_lowlevel_inward_angular_momentum_flux.png"; fig.savefig(out, dpi=220); plt.close(fig); return out


def plot_matched(rows: list[dict[str, float]], phase: str) -> Path:
    pairs = read_csv(ROOT / f"matched_{phase}_pairs.csv"); idx = by_key(rows); scale = 1e15
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    for ax, control in zip(axes, ("rmw15", "r100")):
        x = np.arange(len(pairs)); width = .25
        for shift, (component, color) in zip((-width, 0, width), (("total", "k"), ("favre_mean", "#0072B2"), ("favre_eddy", "#D55E00"))):
            delta = []
            for pair in pairs:
                j = idx[("JET", float(pair["jet_h"]), "low_0p5_5km", control, component)]["inward_flux_Nm"]
                c = idx[("CTRL", float(pair["ctrl_h"]), "low_0p5_5km", control, component)]["inward_flux_Nm"]
                delta.append((j-c)/scale)
            ax.bar(x+shift, delta, width, color=color, label=component.replace("_", " "))
        ax.axhline(0, color="0.2", lw=.8); ax.set_xticks(x, [f"J{float(p['jet_h']):g}/C{float(p['ctrl_h']):g}" for p in pairs], rotation=35, ha="right")
        ax.set_title("1.5 RMW" if control == "rmw15" else "100-km control cylinder"); ax.set_ylabel(r"JET − CTRL inward $M_a$ flux ($10^{15}$ N m)"); ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Intensity/RMW-matched low-level angular-momentum import: {phase.replace('_', '–')} h\npositive total bar supports larger instantaneous JET inward import; not a closed tendency budget", fontsize=12)
    out = FIG / f"matched_{phase}_lowlevel_inward_angular_momentum_flux.png"; fig.savefig(out, dpi=220); plt.close(fig); return out


def summaries(rows: list[dict[str, float]]) -> list[dict[str, object]]:
    idx = by_key(rows); out=[]
    for phase in ("50_75", "80_150"):
        pairs=read_csv(ROOT / f"matched_{phase}_pairs.csv")
        for control in ("rmw15", "r100"):
            for component in ("total", "favre_mean", "favre_eddy"):
                d=np.array([idx[("JET",float(p["jet_h"]),"low_0p5_5km",control,component)]["inward_flux_Nm"]-idx[("CTRL",float(p["ctrl_h"]),"low_0p5_5km",control,component)]["inward_flux_Nm"] for p in pairs])
                out.append({"comparison":"intensity_RMW_matched","phase":phase,"layer":"low_0p5_5km","control":control,"component":component,"n_pairs":len(pairs),"mean_JET_minus_CTRL_inward_Nm":float(d.mean()),"median_JET_minus_CTRL_inward_Nm":float(np.median(d)),"positive_fraction":float((d>0).mean())})
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(exist_ok=True)
    state = lookup_state(); needed = load_requests(); rows=[]; record=[]
    for case in ("CTRL", "JET"):
        with Dataset(DATA[case]) as ds:
            geom={"x_m":np.asarray(ds.variables["xh"][:],float)*1000.0,"y_m":np.asarray(ds.variables["yh"][:],float)*1000.0,"z_m":np.asarray(ds.variables["zh"][:],float)*1000.0}
            geom["edges"]=np.arange(0.0,RMAX_M+DR_M,DR_M); geom["r_m"]=0.5*(geom["edges"][:-1]+geom["edges"][1:])
            for hour in sorted(needed[case]):
                rows.extend(one_time(ds,case,hour,state[(case,hour)],geom)); record.append({"case":case,"hour":hour})
                print(f"{case} {hour:g} h",flush=True)
    write_csv(OUT/"control_surface_fluxes.csv",rows); summary=summaries(rows); write_csv(OUT/"matched_summary.csv",summary)
    figures=[plot_same_time(rows),plot_matched(rows,"50_75"),plot_matched(rows,"80_150")]
    (OUT/"README.json").write_text(json.dumps({"definition":"F_M,r=2*pi*r*integral <rho*u_r*M_a> dz; u_r outward positive; plotted inward import=-F_M,r.","M_a":"r*v_theta+0.5*f*r^2, f=6.2e-5 s-1","layers":LAYERS,"averaging":"native Cartesian fields destaggered to scalar grid; storm centre from timeseries.csv; equal-area radial-bin azimuthal average; Favre mean plus exact residual.","comparisons":"same time 50-150 h every 5 h plus all intensity/RMW-matched pair hours","limitations":"instantaneous control-surface flux, not a time-integrated or closed angular-momentum tendency budget; pressure, vertical, surface and model terms are not included in this flux-only diagnosis.","figures":[str(p) for p in figures],"records":record},indent=2),encoding="utf-8")

if __name__ == "__main__": main()
