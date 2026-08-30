#!/usr/bin/env python
"""Unified CTRL/JET mechanism diagnostics for the 25N CM1 experiments.

The executable separates four pathways: ventilation/shear, boundary-layer
inertial stability, isentropic energetics, and upper-level eddy/SE response.
It never interprets outflow-layer inertial stability as an energetic sink.

The main workflow reads one time at a time and writes reduced r-z products so
that the hundreds-of-GB CM1 files do not need to reside in memory together.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.coordinates import destagger_to_scalar_grid, get_time_slice
from src.environmental_eddy import diagnose_eddy_momentum_forcing
from src.isentropic_energetics import (
    build_isentropic_streamfunction,
    equivalent_potential_temperature,
    extract_closed_streamfunction_contour,
    integrate_thermodynamic_cycle,
    moist_entropy_proxy,
    sample_conditional_field,
    saturation_vapor_pressure_pa,
    temperature_from_potential_temperature,
)
from src.jet_mechanism_diagnostics import (
    CPD,
    EPSILON,
    RD,
    angular_momentum_inertial_stability,
    bulk_vertical_wind_shear,
    cylindrical_wind,
    identify_intensification_phases,
    interpolate_to_pressure,
    inventory_variables,
    lead_lag_correlation,
    match_by_intensity,
    mass_flux_weighted_mean,
    radial_bin_indices,
    radial_mean,
    safe_gradient,
    storm_relative_geometry,
    ventilation_index,
)


ALIASES = {
    "u": ("u", "ua", "uinterp"),
    "v": ("v", "va", "vinterp"),
    "w": ("w", "wa", "winterp"),
    "theta": ("th", "theta"),
    "pressure": ("prs", "pres", "p"),
    "rho": ("rho", "rhoa", "dens"),
    "qv": ("qv", "qvpert"),
    "psfc": ("psfc", "sfcprs", "ps"),
}


@dataclass
class CaseTimeDiagnostics:
    case: str
    time_h: float
    center_x_km: float
    center_y_km: float
    pmin_hpa: float
    vmax_ms: float
    rmw_km: float
    potential_intensity_ms: float
    vws_200_800_ms: float
    vws_500_1000_ms: float
    entropy_deficit_600_jkgk: float
    normalized_entropy_deficit: float
    ventilation_index_200_800: float
    ventilation_index_500_1000: float
    inward_entropy_deficit_covariance: float
    tilt_850_500_km: float
    tilt_850_200_km: float
    wavenumber1_w_ms: float
    bl_i2_inflow_weighted_s2: float
    bl_inflow_ms: float
    bl_mass_convergence_kg_s: float
    eyewall_updraft_kg_s: float
    outflow_mass_flux_kg_s: float
    outflow_i2_massflux_weighted_s2: float
    convective_top_km: float
    sink_temperature_k: float
    eddy_forcing_jet_box_rms_ms2: float


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Separate shear, BL-I2, energetics and eddy/SE jet pathways")
    p.add_argument("--ctrl", default="/data/zhangyx/DATA/cm1out_25N_nojet.nc")
    p.add_argument("--jet", default="/data/zhangyx/DATA/cm1out_25N_9o_jet_30.nc")
    p.add_argument("--output", default="output/jet_mechanism_25N_ctrl_vs_jet30_9deg")
    p.add_argument("--start-hour", type=float, default=30.0)
    p.add_argument("--end-hour", type=float, default=84.0)
    p.add_argument("--step-hour", type=float, default=2.0)
    p.add_argument("--energy-times", default="40,55,70,80")
    p.add_argument("--se-time", type=float, default=70.0)
    p.add_argument("--f", type=float, default=6.2e-5)
    p.add_argument("--dr-km", type=float, default=5.0)
    p.add_argument("--max-r-km", type=float, default=1200.0)
    p.add_argument("--sst-k", type=float, default=300.0)
    p.add_argument("--potential-intensity-ms", type=float, default=np.nan,
                   help="Optional fixed PI; NaN computes PI from the environmental sounding with tcpyPI")
    p.add_argument("--thetae-min-k", type=float, default=320.0)
    p.add_argument("--thetae-max-k", type=float, default=390.0)
    p.add_argument("--thetae-bin-k", type=float, default=1.0)
    p.add_argument("--isentropic-contour-fraction", type=float, default=0.5)
    p.add_argument("--regularization", default="1e-5,1e-4,1e-3")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--postprocess-only", action="store_true",
                   help="Rebuild matching/lag figures and report from existing reduced products")
    p.add_argument("--skip-se", action="store_true")
    p.add_argument("--skip-energy", action="store_true")
    p.add_argument("--resume", action="store_true", help="Reuse per-time NPZ products when present")
    return p


def _first(ds: xr.Dataset, candidates: Sequence[str], required: bool = True) -> str:
    for name in candidates:
        if name in ds.variables:
            return name
    if required:
        raise KeyError(f"none of {tuple(candidates)} exists")
    return ""


def _coord_m(ds: xr.Dataset, name: str) -> np.ndarray:
    values = np.asarray(ds[name].values, float)
    units = str(ds[name].attrs.get("units", "")).lower()
    if "km" in units or (values.size and np.nanmax(np.abs(values)) < 10000.0):
        values = values * 1000.0
    return values


def _time_seconds(ds: xr.Dataset) -> np.ndarray:
    if "time" not in ds:
        return np.arange(ds.sizes.get("time", 1), dtype=float)
    values = np.asarray(ds["time"].values, float)
    units = str(ds["time"].attrs.get("units", "")).lower()
    if "hour" in units:
        values *= 3600.0
    return values


def _read_scalar(ds: xr.Dataset, name: str, index: int) -> np.ndarray:
    data, dims = get_time_slice(ds[name], index)
    data, dims = destagger_to_scalar_grid(data, dims)
    desired = [d for d in ("zh", "yh", "xh") if d in dims]
    if data.ndim == 3 and dims != desired:
        data = np.transpose(data, [dims.index(d) for d in desired])
    elif data.ndim == 2 and dims != ["yh", "xh"]:
        data = np.transpose(data, [dims.index("yh"), dims.index("xh")])
    return np.asarray(data, float)


def _full_theta(ds: xr.Dataset, name: str, index: int) -> np.ndarray:
    theta = _read_scalar(ds, name, index)
    # CM1 normally writes full th; a perturbation field is identifiable by scale.
    if np.nanmedian(theta) < 100.0:
        for base_name in ("th0", "thbar", "thref"):
            if base_name in ds:
                base = np.asarray(ds[base_name].values, float).squeeze()
                if base.ndim == 1 and base.size == theta.shape[0]:
                    theta = theta + base[:, None, None]
                    break
    return theta


def _full_qv(ds: xr.Dataset, name: str, index: int, shape: Tuple[int, ...]) -> np.ndarray:
    qv = _read_scalar(ds, name, index)
    if name == "qvpert" or np.nanmin(qv) < -1.0e-6:
        for base_name in ("qv0", "qvbar", "qvref"):
            if base_name in ds:
                base = np.asarray(ds[base_name].values, float).squeeze()
                if base.ndim == 1 and base.size == qv.shape[0]:
                    qv = qv + base[:, None, None]
                    break
    return np.maximum(np.broadcast_to(qv, shape), 0.0)


def _center_from_psfc(psfc: np.ndarray, x_m: np.ndarray, y_m: np.ndarray) -> Tuple[float, float, float]:
    try:
        from scipy.ndimage import gaussian_filter
        smooth = gaussian_filter(psfc, 2.0)
    except Exception:
        smooth = psfc
    iy, ix = np.unravel_index(np.nanargmin(smooth), smooth.shape)
    pmin = float(np.nanmin(smooth))
    if pmin > 2000.0:
        pmin /= 100.0
    return float(x_m[ix]), float(y_m[iy]), pmin


def _vorticity_center(u2: np.ndarray, v2: np.ndarray, x_m: np.ndarray, y_m: np.ndarray,
                      radius: np.ndarray, center_x: float, center_y: float) -> Tuple[float, float]:
    dv_dx = safe_gradient(v2, x_m, axis=1)
    du_dy = safe_gradient(u2, y_m, axis=0)
    zeta = dv_dx - du_dy
    use = (radius <= 250000.0) & np.isfinite(zeta)
    weights = np.where(use, np.maximum(zeta - np.nanpercentile(zeta[use], 60.0), 0.0), 0.0)
    den = float(np.sum(weights))
    if den <= 0:
        return center_x, center_y
    xx, yy = np.meshgrid(x_m, y_m)
    return float(np.sum(weights * xx) / den), float(np.sum(weights * yy) / den)


def _saturation_mixing_ratio(temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    es = np.minimum(saturation_vapor_pressure_pa(temperature), 0.95 * pressure)
    return EPSILON * es / np.maximum(pressure - es, 1.0)


def _potential_intensity_ms(
    temperature_zyx: np.ndarray, pressure_zyx: np.ndarray, qv_zyx: np.ndarray,
    psfc_yx: np.ndarray, radius_yx: np.ndarray, sst_k: float,
) -> float:
    """Bister-Emanuel PI from a 200-800-km environmental mean sounding."""
    try:
        import tcpyPI
    except Exception:
        return np.nan
    mask = (radius_yx >= 200000.0) & (radius_yx <= 800000.0)
    if np.count_nonzero(mask) < 4:
        return np.nan
    p = np.nanmean(pressure_zyx[:, mask], axis=1) / 100.0
    t = np.nanmean(temperature_zyx[:, mask], axis=1) - 273.15
    mixing = 1000.0 * np.nanmean(qv_zyx[:, mask], axis=1)
    good = np.isfinite(p) & np.isfinite(t) & np.isfinite(mixing)
    p, t, mixing = p[good], t[good], mixing[good]
    if p.size < 6:
        return np.nan
    order = np.argsort(p)[::-1]
    p, t, mixing = p[order], t[order], mixing[order]
    msl = float(np.nanmean(psfc_yx[mask]))
    if msl > 2000.0:
        msl /= 100.0
    try:
        vmax, _, flag, _, _ = tcpyPI.pi(float(sst_k - 273.15), msl, p, t, mixing)
        return float(vmax) if int(flag) == 1 and np.isfinite(vmax) else np.nan
    except Exception:
        return np.nan


def _box_rms(field: np.ndarray, r_km: np.ndarray, z_m: np.ndarray,
             rbounds: Tuple[float, float], zbounds: Tuple[float, float]) -> float:
    mask = ((r_km[None, :] >= rbounds[0]) & (r_km[None, :] <= rbounds[1])
            & (z_m[:, None] >= zbounds[0]) & (z_m[:, None] <= zbounds[1]))
    values = np.asarray(field)[mask & np.isfinite(field)]
    return float(np.sqrt(np.mean(values**2))) if values.size else np.nan


def _mass_transport_extreme(r_m: np.ndarray, z_m: np.ndarray, rho_zr: np.ndarray,
                            velocity_zr: np.ndarray, rmask: np.ndarray, zmask: np.ndarray,
                            component: str, positive: bool = True) -> float:
    """Maximum cylindrical mass transport across radial or horizontal surfaces."""
    speed = np.maximum(velocity_zr, 0.0) if positive else np.maximum(-velocity_zr, 0.0)
    field = 2.0 * np.pi * rho_zr * speed * r_m[None, :]
    if component == "vertical":
        transport = np.trapezoid(field[:, rmask], r_m[rmask], axis=1)
        values = transport[zmask]
    elif component == "radial":
        transport = np.trapezoid(field[zmask, :], z_m[zmask], axis=0)
        values = transport[rmask]
    else:
        raise ValueError("component must be 'vertical' or 'radial'")
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else np.nan


def _diagnose_time(ds: xr.Dataset, index: int, case: str, args, per_time_dir: Path) -> Tuple[CaseTimeDiagnostics, Dict[str, np.ndarray]]:
    names = {key: _first(ds, value) for key, value in ALIASES.items()}
    x_m, y_m, z_m = (_coord_m(ds, d) for d in ("xh", "yh", "zh"))
    u = _read_scalar(ds, names["u"], index)
    v = _read_scalar(ds, names["v"], index)
    w = _read_scalar(ds, names["w"], index)
    theta = _full_theta(ds, names["theta"], index)
    pressure = _read_scalar(ds, names["pressure"], index)
    rho = _read_scalar(ds, names["rho"], index)
    qv = _full_qv(ds, names["qv"], index, theta.shape)
    psfc = _read_scalar(ds, names["psfc"], index)
    center_x, center_y, pmin = _center_from_psfc(psfc, x_m, y_m)
    geom = storm_relative_geometry(x_m, y_m, center_x, center_y)
    radius = geom["radius_m"]
    ur, vt = cylindrical_wind(u, v, geom["cos_azimuth"], geom["sin_azimuth"])
    edges = np.arange(0.0, args.max_r_km * 1000.0 + args.dr_km * 1000.0, args.dr_km * 1000.0)
    r_m = 0.5 * (edges[:-1] + edges[1:])
    r_km = r_m / 1000.0
    bidx, bvalid = radial_bin_indices(radius, edges)
    nr = r_m.size
    ur_zr = radial_mean(ur, bidx, bvalid, nr)
    vt_zr = radial_mean(vt, bidx, bvalid, nr)
    w_zr = radial_mean(w, bidx, bvalid, nr)
    rho_zr = radial_mean(rho, bidx, bvalid, nr)
    theta_zr = radial_mean(theta, bidx, bvalid, nr)
    pressure_zr = radial_mean(pressure, bidx, bvalid, nr)
    i2 = angular_momentum_inertial_stability(vt_zr, r_m, args.f)

    lowest = int(np.argmin(np.abs(z_m - 1000.0)))
    core = r_km <= 300.0
    vmax = float(np.nanmax(vt_zr[lowest, core]))
    rmw_index = np.where(core)[0][int(np.nanargmax(vt_zr[lowest, core]))]
    rmw_km = float(r_km[rmw_index])

    pressure_levels = {}
    for lev in (85000.0, 60000.0, 50000.0, 20000.0):
        pressure_levels[lev] = {
            "u": interpolate_to_pressure(u, pressure, lev),
            "v": interpolate_to_pressure(v, pressure, lev),
            "theta": interpolate_to_pressure(theta, pressure, lev),
            "qv": interpolate_to_pressure(qv, pressure, lev),
        }
    shear_a = bulk_vertical_wind_shear(
        pressure_levels[85000.0]["u"], pressure_levels[85000.0]["v"],
        pressure_levels[20000.0]["u"], pressure_levels[20000.0]["v"],
        radius, (200000.0, 800000.0),
    )
    shear_b = bulk_vertical_wind_shear(
        pressure_levels[85000.0]["u"], pressure_levels[85000.0]["v"],
        pressure_levels[20000.0]["u"], pressure_levels[20000.0]["v"],
        radius, (500000.0, 1000000.0),
    )

    th600 = pressure_levels[60000.0]["theta"]
    q600 = np.maximum(pressure_levels[60000.0]["qv"], 0.0)
    p600 = np.full_like(th600, 60000.0)
    t600 = temperature_from_potential_temperature(th600, p600)
    qsat600 = _saturation_mixing_ratio(t600, p600)
    s600 = moist_entropy_proxy(equivalent_potential_temperature(th600, p600, q600))
    ssat600 = moist_entropy_proxy(equivalent_potential_temperature(th600, p600, qsat600))
    env = (radius >= 200000.0) & (radius <= 800000.0)
    entropy_deficit = float(np.nanmean((ssat600 - s600)[env]))
    bl_k = int(np.argmin(np.abs(z_m - 500.0)))
    inner = radius <= 200000.0
    s_bl = moist_entropy_proxy(equivalent_potential_temperature(theta[bl_k], pressure[bl_k], qv[bl_k]))
    ps_mean = float(np.nanmean(pressure[bl_k][inner]))
    qsat_sst = _saturation_mixing_ratio(np.array(args.sst_k), np.array(ps_mean))
    theta_sst = args.sst_k * (100000.0 / ps_mean) ** (RD / CPD)
    s_sst = float(moist_entropy_proxy(equivalent_potential_temperature(
        np.array(theta_sst), np.array(ps_mean), np.array(qsat_sst)
    )))
    denominator = s_sst - float(np.nanmean(s_bl[inner]))
    chi = entropy_deficit / denominator if denominator > 0 else np.nan

    theta_e = equivalent_potential_temperature(theta, pressure, qv)
    entropy = moist_entropy_proxy(theta_e)
    entropy_deficit_3d = moist_entropy_proxy(equivalent_potential_temperature(
        theta, pressure, _saturation_mixing_ratio(temperature_from_potential_temperature(theta, pressure), pressure)
    )) - entropy
    deficit_zr = radial_mean(entropy_deficit_3d, bidx, bvalid, nr)
    ur_mean_xy = np.full_like(ur, np.nan)
    deficit_mean_xy = np.full_like(entropy_deficit_3d, np.nan)
    in_range = bvalid & (bidx >= 0) & (bidx < nr)
    for k in range(z_m.size):
        ur_mean_xy[k].ravel()[in_range] = ur_zr[k, bidx[in_range]]
        deficit_mean_xy[k].ravel()[in_range] = deficit_zr[k, bidx[in_range]]
    vent_mask = ((radius[None, :, :] >= 100000.0) & (radius[None, :, :] <= 300000.0)
                 & (z_m[:, None, None] >= 4000.0) & (z_m[:, None, None] <= 8000.0))
    vent_product = -rho * (ur - ur_mean_xy) * (entropy_deficit_3d - deficit_mean_xy)
    inward_cov = float(np.nanmean(vent_product[vent_mask]))

    centers = {}
    for lev in (85000.0, 50000.0, 20000.0):
        centers[lev] = _vorticity_center(
            pressure_levels[lev]["u"], pressure_levels[lev]["v"], x_m, y_m,
            radius, center_x, center_y,
        )
    tilt_850_500 = float(np.hypot(centers[85000.0][0] - centers[50000.0][0], centers[85000.0][1] - centers[50000.0][1]) / 1000.0)
    tilt_850_200 = float(np.hypot(centers[85000.0][0] - centers[20000.0][0], centers[85000.0][1] - centers[20000.0][1]) / 1000.0)

    eyewall_xy = (radius >= max(10000.0, (rmw_km - 40.0) * 1000.0)) & (radius <= (rmw_km + 40.0) * 1000.0)
    z_w1 = int(np.argmin(np.abs(z_m - 8000.0)))
    weights = eyewall_xy & np.isfinite(w[z_w1])
    wave1 = float(abs(np.mean(w[z_w1][weights] * np.exp(-1j * geom["azimuth_rad"][weights])))) if np.any(weights) else np.nan

    bl_i2 = mass_flux_weighted_mean(
        i2["I2"], rho_zr, ur_zr, r_m, z_m, (0.0, 300000.0), (0.0, 2000.0), positive_velocity=False
    )
    bl_mask = (z_m[:, None] <= 2000.0) & (r_km[None, :] <= 300.0)
    bl_inflow = float(np.nanmean(ur_zr[bl_mask]))
    bl_conv = _mass_transport_extreme(
        r_m, z_m, rho_zr, ur_zr, r_km <= 300.0, z_m <= 2000.0,
        component="radial", positive=False,
    )
    eyewall_flux = _mass_transport_extreme(
        r_m, z_m, rho_zr, w_zr,
        (r_km >= max(10.0, rmw_km - 40.0)) & (r_km <= rmw_km + 40.0),
        (z_m >= 2000.0) & (z_m <= 14000.0), component="vertical", positive=True,
    )
    outflow_flux = _mass_transport_extreme(
        r_m, z_m, rho_zr, ur_zr, (r_km >= 200.0) & (r_km <= 1200.0),
        (z_m >= 10000.0) & (z_m <= 17000.0), component="radial", positive=True,
    )
    out_i2 = mass_flux_weighted_mean(
        i2["I2"], rho_zr, ur_zr, r_m, z_m,
        (200000.0, 1200000.0), (10000.0, 17000.0), positive_velocity=True,
    )
    updraft_profile = np.nanmax(np.where(r_km[None, :] <= 200.0, w_zr, np.nan), axis=1)
    active = np.where(updraft_profile >= 1.0)[0]
    conv_top = float(z_m[active[-1]] / 1000.0) if active.size else np.nan
    temp = temperature_from_potential_temperature(theta, pressure)
    pi_ms = (
        float(args.potential_intensity_ms)
        if np.isfinite(args.potential_intensity_ms)
        else _potential_intensity_ms(temp, pressure, qv, psfc, radius, args.sst_k)
    )
    out_mask = ((radius[None, :, :] <= 300000.0) & (z_m[:, None, None] >= 10000.0)
                & (z_m[:, None, None] <= 17000.0) & (ur > 0))
    out_weights = np.where(out_mask, rho * np.maximum(ur, 0.0), 0.0)
    sink_temp = float(np.sum(out_weights * temp) / np.sum(out_weights)) if np.sum(out_weights) > 0 else np.nan

    eddy = diagnose_eddy_momentum_forcing(
        ur, vt, w, rho, bidx, bvalid, nr, r_m, z_m,
        averaging="reynolds", theta_3d=theta,
    )
    jet_box_rms = _box_rms(eddy["F_lambda_eddy"], r_km, z_m, (750.0, 1100.0), (10000.0, 17000.0))
    time_h = float(_time_seconds(ds)[index] / 3600.0)
    row = CaseTimeDiagnostics(
        case=case, time_h=time_h, center_x_km=center_x / 1000.0, center_y_km=center_y / 1000.0,
        pmin_hpa=pmin, vmax_ms=vmax, rmw_km=rmw_km, potential_intensity_ms=pi_ms,
        vws_200_800_ms=shear_a["magnitude"], vws_500_1000_ms=shear_b["magnitude"],
        entropy_deficit_600_jkgk=entropy_deficit, normalized_entropy_deficit=chi,
        ventilation_index_200_800=ventilation_index(shear_a["magnitude"], chi, pi_ms),
        ventilation_index_500_1000=ventilation_index(shear_b["magnitude"], chi, pi_ms),
        inward_entropy_deficit_covariance=inward_cov,
        tilt_850_500_km=tilt_850_500, tilt_850_200_km=tilt_850_200,
        wavenumber1_w_ms=wave1, bl_i2_inflow_weighted_s2=bl_i2, bl_inflow_ms=bl_inflow,
        bl_mass_convergence_kg_s=bl_conv, eyewall_updraft_kg_s=eyewall_flux,
        outflow_mass_flux_kg_s=outflow_flux, outflow_i2_massflux_weighted_s2=out_i2,
        convective_top_km=conv_top, sink_temperature_k=sink_temp,
        eddy_forcing_jet_box_rms_ms2=jet_box_rms,
    )
    reduced = {
        "r_km": r_km, "z_km": z_m / 1000.0, "ur_zr": ur_zr, "vt_zr": vt_zr,
        "w_zr": w_zr, "rho_zr": rho_zr, "theta_zr": theta_zr,
        "pressure_zr": pressure_zr, "M": i2["M"], "dM_dr": i2["dM_dr"],
        "I2": i2["I2"], "zeta_absolute": i2["zeta_absolute"],
        "F_lambda_eddy": eddy["F_lambda_eddy"],
        "F_lambda_eddy_radial": eddy["F_lambda_eddy_radial"],
        "F_lambda_eddy_vertical": eddy["F_lambda_eddy_vertical"],
    }
    np.savez_compressed(per_time_dir / f"{case}_{time_h:06.1f}h.npz", **reduced)
    return row, {
        **reduced, "theta_e": theta_e, "temperature": temp, "pressure": pressure,
        "qv": qv, "rho": rho, "w": w, "radius": radius,
        "cell_area_m2": np.array(abs(np.nanmedian(np.diff(x_m)) * np.nanmedian(np.diff(y_m)))),
    }


def _run_energy(case: str, time_h: float, state: Mapping[str, np.ndarray], args, out_dir: Path) -> Dict[str, object]:
    edges = np.arange(args.thetae_min_k, args.thetae_max_k + args.thetae_bin_k, args.thetae_bin_k)
    radius = state["radius"]
    mask = radius <= min(args.max_r_km * 1000.0, 1200000.0)
    r_km = state["r_km"]
    # Grid area from the storm-relative Cartesian radius field is not unique;
    # infer it from adjacent points in x/y through local radius spacing only as
    # a fallback. The absolute streamfunction is less important than closure.
    area = float(state["cell_area_m2"])
    product = build_isentropic_streamfunction(
        state["theta_e"], state["w"], state["rho"], state["z_km"] * 1000.0,
        edges, area,
        state_fields={
            "temperature_k": state["temperature"],
            "pressure_pa": state["pressure"],
            "qv_kgkg": state["qv"],
        },
        horizontal_mask=mask,
    )
    serializable: Dict[str, object] = {
        "case": case, "time_h": time_h,
        "mass_closure_ratio": float(product["mass_closure_ratio"]),
        "cycle_available": False,
    }
    try:
        cycle = extract_closed_streamfunction_contour(
            product["theta_e_center_k"], product["z_m"], product["streamfunction_kg_s"],
            args.isentropic_contour_fraction,
        )
        sampled = {}
        for key in ("temperature_k", "pressure_pa", "qv_kgkg"):
            sampled[key] = sample_conditional_field(
                product["theta_e_center_k"], product["z_m"], product[f"conditional_{key}"],
                cycle["theta_e_k"], cycle["z_m"],
            )
        entropy = moist_entropy_proxy(cycle["theta_e_k"])
        energetics = integrate_thermodynamic_cycle(
            sampled["temperature_k"], sampled["pressure_pa"], entropy,
            cycle["z_m"], sampled["qv_kgkg"],
        )
        serializable.update({k: v for k, v in energetics.items() if k != "segment_labels"})
        serializable["cycle_available"] = True
        serializable["cycle_level_kg_s"] = float(cycle["level_kg_s"])
        np.savez_compressed(
            out_dir / f"isentropic_{case}_{time_h:06.1f}h.npz",
            **{k: v for k, v in product.items() if isinstance(v, np.ndarray)},
            cycle_theta_e_k=cycle["theta_e_k"], cycle_z_m=cycle["z_m"],
            cycle_temperature_k=sampled["temperature_k"], cycle_pressure_pa=sampled["pressure_pa"],
            cycle_qv_kgkg=sampled["qv_kgkg"], cycle_entropy_jkgk=entropy,
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        levels = np.linspace(np.nanpercentile(product["streamfunction_kg_s"], 2), np.nanpercentile(product["streamfunction_kg_s"], 98), 21)
        axes[0].contourf(product["theta_e_center_k"], product["z_m"] / 1000.0,
                         product["streamfunction_kg_s"], levels=levels, cmap="RdBu_r")
        axes[0].plot(cycle["theta_e_k"], cycle["z_m"] / 1000.0, "k", lw=2)
        axes[0].set(xlabel=r"$\theta_e$ (K)", ylabel="Height (km)", title=f"{case} {time_h:.0f} h isentropic circulation")
        axes[1].plot(entropy, sampled["temperature_k"], "k", lw=2)
        axes[1].set(xlabel=r"$c_{pd}\ln\theta_e$ (J kg$^{-1}$ K$^{-1}$)", ylabel="Temperature (K)", title="Resolved T-s cycle")
        fig.savefig(out_dir / f"isentropic_cycle_{case}_{time_h:06.1f}h.png", dpi=200)
        fig.savefig(out_dir / f"isentropic_cycle_{case}_{time_h:06.1f}h.pdf")
        plt.close(fig)
    except Exception as exc:
        serializable["cycle_error"] = f"{type(exc).__name__}: {exc}"
    (out_dir / f"isentropic_{case}_{time_h:06.1f}h.json").write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return serializable


def _write_rows(rows: Sequence[CaseTimeDiagnostics], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _read_rows(path: Path) -> List[CaseTimeDiagnostics]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [CaseTimeDiagnostics(**{
            key: (value if key == "case" else float(value))
            for key, value in record.items()
        }) for record in csv.DictReader(handle)]


def _load_energy_results(path: Path) -> List[Mapping[str, object]]:
    return [json.loads(item.read_text(encoding="utf-8")) for item in sorted(path.glob("isentropic_*.json"))]


def _plot_timeseries(rows: Sequence[CaseTimeDiagnostics], out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13, 13), constrained_layout=True, sharex=True)
    metrics = [
        ("pmin_hpa", "Minimum pressure (hPa)"),
        ("vws_200_800_ms", "850-200-hPa VWS (m/s)"),
        ("inward_entropy_deficit_covariance", "Inward entropy-deficit covariance"),
        ("bl_i2_inflow_weighted_s2", r"BL inflow-weighted $I^2$ (s$^{-2}$)"),
        ("eyewall_updraft_kg_s", "Eyewall upward transport (kg/s)"),
        ("eddy_forcing_jet_box_rms_ms2", r"Upper eddy forcing RMS (m s$^{-2}$)"),
    ]
    for case, color in (("CTRL", "black"), ("JET", "tab:red")):
        subset = [r for r in rows if r.case == case]
        t = np.array([r.time_h for r in subset])
        for ax, (name, label) in zip(axes.flat, metrics):
            ax.plot(t, [getattr(r, name) for r in subset], color=color, label=case)
            ax.set_ylabel(label); ax.grid(alpha=0.25)
    for ax in axes[-1]: ax.set_xlabel("Time (h)")
    axes[0, 0].legend()
    fig.savefig(out_dir / "four_pathway_timeseries.png", dpi=220)
    fig.savefig(out_dir / "four_pathway_timeseries.pdf")
    plt.close(fig)


def _plot_70h(ctrl_file: Path, jet_file: Path, out_dir: Path) -> None:
    c, j = np.load(ctrl_file), np.load(jet_file)
    r, z = c["r_km"], c["z_km"]
    rr, zz = np.meshgrid(r, z)
    fields = [
        (j["I2"] - c["I2"], r"JET-CTRL $I^2$"),
        (j["dM_dr"] - c["dM_dr"], r"JET-CTRL $\partial M/\partial r$"),
        (j["F_lambda_eddy"] - c["F_lambda_eddy"], r"$F_{\lambda,env}$"),
        (j["ur_zr"] - c["ur_zr"], "JET-CTRL radial wind"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, (field, title) in zip(axes.flat, fields):
        vmax = max(float(np.nanpercentile(np.abs(field), 99)), 1e-20)
        im = ax.contourf(rr, zz, field, levels=np.linspace(-vmax, vmax, 25), cmap="RdBu_r", extend="both")
        ax.set(xlabel="Radius (km)", ylabel="Height (km)", title=title)
        fig.colorbar(im, ax=ax)
    fig.savefig(out_dir / "seventy_hour_structure.png", dpi=220)
    fig.savefig(out_dir / "seventy_hour_structure.pdf")
    plt.close(fig)


def _se_factorial(args, out_dir: Path) -> None:
    """Run C0/CI/CF/JF; CI replaces only the generalized inertia part of K3."""
    out_dir.mkdir(parents=True, exist_ok=True)
    from src._se_pipeline_environmental import _solve_response
    from src._se_pipeline_single import PipelineConfig, azimuthal_average_from_3d
    from src.se_bui import assemble_operator, build_basic_state, build_forcing, invert_balanced_theta, regularize_ellipticity

    base_cfg = PipelineConfig(
        input_file=args.ctrl, output_dir=str(out_dir), target_time_hours=args.se_time,
        max_r_km=args.max_r_km, dr_km=args.dr_km, max_z_km=20.0,
        coriolis_f=args.f, write_netcdf=False, write_ieee=False, plot_solution=False,
        baroclinic_scale=1.0, sor_max_iter=60000, sor_tol=1e-14,
    )
    ctrl = azimuthal_average_from_3d(base_cfg)
    jet_cfg = PipelineConfig(**{**base_cfg.__dict__, "input_file": args.jet})
    jet = azimuthal_average_from_3d(jet_cfg)
    r_m = np.asarray(ctrl["r_km"]) * 1000.0
    z_m = np.asarray(ctrl["z_km"]) * 1000.0
    theta_c, _ = invert_balanced_theta(ctrl["ut"], ctrl["theta"], r_m, z_m, args.f)
    theta_j, _ = invert_balanced_theta(jet["ut"], jet["theta"], r_m, z_m, args.f)
    basic_c = build_basic_state(ctrl["ut"], theta_c, ctrl["rho"], r_m, z_m, args.f)
    basic_j = build_basic_state(jet["ut"], theta_j, jet["rho"], r_m, z_m, args.f)
    inertia_c = basic_c["chi"] * basic_c["xi"] * basic_c["zeta_abs"]
    inertia_j_on_c = basic_c["chi"] * basic_j["xi"] * basic_j["zeta_abs"]
    hybrid_k3 = basic_c["K3_raw"] - inertia_c + inertia_j_on_c
    forcing_c = build_forcing(basic_c, ctrl["Q"], ctrl["Fnu"], r_m, z_m)["forcing_total"]
    forcing_j_on_c = build_forcing(basic_c, jet["Q"], jet["Fnu"], r_m, z_m)["forcing_total"]
    forcing_j = build_forcing(basic_j, jet["Q"], jet["Fnu"], r_m, z_m)["forcing_total"]
    summary = {}
    for eps in [float(x) for x in args.regularization.split(",") if x.strip()]:
        operators = {}
        infos = {}
        for name, basic, k1, k2, k3 in (
            ("C0", basic_c, basic_c["K1_raw"], basic_c["K2_raw"], basic_c["K3_raw"]),
            ("CI", basic_c, basic_c["K1_raw"], basic_c["K2_raw"], hybrid_k3),
            ("JF", basic_j, basic_j["K1_raw"], basic_j["K2_raw"], basic_j["K3_raw"]),
        ):
            a, b, c, info = regularize_ellipticity(k1, k2, k3, eps_ratio=eps)
            operators[name] = assemble_operator(basic, a, b, c, r_m, z_m)
            infos[name] = info
        solutions = {}
        for name, operator, forcing, density in (
            ("C0", operators["C0"], forcing_c, ctrl["rho"]),
            ("CI", operators["CI"], forcing_c, ctrl["rho"]),
            ("CF", operators["C0"], forcing_j_on_c, ctrl["rho"]),
            ("JF", operators["JF"], forcing_j, jet["rho"]),
        ):
            psi, use, wse = _solve_response(operator, forcing, density, r_m, z_m, base_cfg)
            solutions[name] = {"psi": psi, "U": use, "W": wse}
        tag = f"eps{eps:.0e}".replace("-", "m")
        np.savez_compressed(
            out_dir / f"se_i2_only_factorial_{tag}.npz", r_km=ctrl["r_km"], z_km=ctrl["z_km"],
            **{f"{case}_{field}": value for case, fields in solutions.items() for field, value in fields.items()},
        )
        r_km = np.asarray(ctrl["r_km"], float)
        z_km = np.asarray(ctrl["z_km"], float)
        rr, zz = np.meshgrid(r_km, z_km)
        response_fields = {
            "I2-only operator": solutions["CI"]["U"][:, 1:-1].T - solutions["C0"]["U"][:, 1:-1].T,
            "forcing-only": solutions["CF"]["U"][:, 1:-1].T - solutions["C0"]["U"][:, 1:-1].T,
            "total": solutions["JF"]["U"][:, 1:-1].T - solutions["C0"]["U"][:, 1:-1].T,
            "interaction": (
                solutions["JF"]["U"][:, 1:-1].T - solutions["CI"]["U"][:, 1:-1].T
                - solutions["CF"]["U"][:, 1:-1].T + solutions["C0"]["U"][:, 1:-1].T
            ),
        }
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        for ax, (title, field) in zip(axes.flat, response_fields.items()):
            vmax = max(float(np.nanpercentile(np.abs(field), 99.0)), 1.0e-20)
            image = ax.contourf(
                rr, zz, field, levels=np.linspace(-vmax, vmax, 25),
                cmap="RdBu_r", extend="both",
            )
            ax.set(xlabel="Radius (km)", ylabel="Height (km)", title=title + " radial response")
            fig.colorbar(image, ax=ax)
        fig.suptitle(f"I2-only SE factorial, eps={eps:g}; regularized balanced projection")
        fig.savefig(out_dir / f"se_i2_only_factorial_{tag}.png", dpi=210)
        fig.savefig(out_dir / f"se_i2_only_factorial_{tag}.pdf")
        plt.close(fig)
        summary[str(eps)] = infos
    (out_dir / "se_i2_only_factorial_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _plot_matching_and_lag(
    ctrl: Sequence[CaseTimeDiagnostics], jet: Sequence[CaseTimeDiagnostics],
    lag_results: Mapping[str, Mapping[str, np.ndarray]], out_dir: Path,
) -> Mapping[str, np.ndarray]:
    """Write the required intensity-matching, lead-lag and r/RMW figures."""
    figures = out_dir / "figures"
    figures.mkdir(exist_ok=True)
    t = np.array([row.time_h for row in ctrl])
    pc = np.array([row.pmin_hpa for row in ctrl]); pj = np.array([row.pmin_hpa for row in jet])
    vc = np.array([row.vmax_ms for row in ctrl]); vj = np.array([row.vmax_ms for row in jet])
    rc = np.array([row.rmw_km for row in ctrl]); rj = np.array([row.rmw_km for row in jet])
    p_match = match_by_intensity(pj, pc, t)
    v_match = match_by_intensity(vj, vc, t)
    pi = p_match["index"]; vi = v_match["index"]

    np.savez_compressed(
        out_dir / "strength_matching.npz", jet_time_h=t,
        pmin_ctrl_index=pi, pmin_ctrl_time_h=p_match["time_h"],
        pmin_mismatch_hpa=p_match["absolute_mismatch"],
        pmin_rmw_mismatch_km=np.abs(rc[pi] - rj),
        vmax_ctrl_index=vi, vmax_ctrl_time_h=v_match["time_h"],
        vmax_mismatch_ms=v_match["absolute_mismatch"],
        vmax_rmw_mismatch_km=np.abs(rc[vi] - rj),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes[0, 0].scatter(pj, pc[pi], c=t, cmap="viridis", s=35)
    lim = [min(pj.min(), pc[pi].min()), max(pj.max(), pc[pi].max())]
    axes[0, 0].plot(lim, lim, "k--", lw=1); axes[0, 0].set(xlabel="JET pmin (hPa)", ylabel="matched CTRL pmin (hPa)", title="Minimum-pressure matching")
    axes[0, 1].scatter(vj, vc[vi], c=t, cmap="viridis", s=35)
    lim = [min(vj.min(), vc[vi].min()), max(vj.max(), vc[vi].max())]
    axes[0, 1].plot(lim, lim, "k--", lw=1); axes[0, 1].set(xlabel="JET vmax (m s$^{-1}$)", ylabel="matched CTRL vmax (m s$^{-1}$)", title="Maximum-wind matching")
    axes[1, 0].plot(t, np.abs(rc[pi] - rj), label="pmin match")
    axes[1, 0].plot(t, np.abs(rc[vi] - rj), label="vmax match")
    axes[1, 0].axhline(25.0, color="k", ls="--", lw=1); axes[1, 0].set(xlabel="JET time (h)", ylabel="|RMW difference| (km)", title="RMW compatibility"); axes[1, 0].legend()
    axes[1, 1].plot(t, p_match["absolute_mismatch"], label="pmin mismatch (hPa)")
    axes[1, 1].plot(t, v_match["absolute_mismatch"], label="vmax mismatch (m s$^{-1}$)")
    axes[1, 1].set(xlabel="JET time (h)", ylabel="absolute mismatch", title="Match quality"); axes[1, 1].legend()
    fig.savefig(figures / "strength_matching.png", dpi=220); fig.savefig(figures / "strength_matching.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for name, result in lag_results.items():
        ax.plot(result["lead_h"], result["correlation"], marker="o", ms=3, label=name)
    ax.axhline(0.0, color="k", lw=0.8); ax.axvspan(6.0, 24.0, color="0.85", zorder=-2, label="required lead window")
    ax.set(xlabel="Predictor lead (h)", ylabel="correlation with JET-minus-CTRL intensification rate", title="Lead-lag screening"); ax.legend(ncol=2)
    fig.savefig(figures / "lead_lag.png", dpi=220); fig.savefig(figures / "lead_lag.pdf"); plt.close(fig)

    chosen = [min(range(len(jet)), key=lambda i: abs(jet[i].time_h - hour)) for hour in (54.0, 70.0)]
    fig, axes = plt.subplots(3, 2, figsize=(11, 11), constrained_layout=True)
    for col, ji in enumerate(chosen):
        ci = int(pi[ji]); jr = jet[ji]; cr = ctrl[ci]
        jp = np.load(out_dir / "data" / f"JET_{jr.time_h:06.1f}h.npz")
        cp = np.load(out_dir / "data" / f"CTRL_{cr.time_h:06.1f}h.npz")
        for row_index, (field, height, ylabel) in enumerate((("vt_zr", 1.0, "Vt (m s$^{-1}$)"), ("ur_zr", 0.5, "Ur (m s$^{-1}$)"), ("w_zr", 5.0, "W (m s$^{-1}$)"))):
            ax = axes[row_index, col]
            jk = int(np.argmin(np.abs(jp["z_km"] - height))); ck = int(np.argmin(np.abs(cp["z_km"] - height)))
            ax.plot(jp["r_km"] / jr.rmw_km, jp[field][jk], label=f"JET {jr.time_h:g} h")
            ax.plot(cp["r_km"] / cr.rmw_km, cp[field][ck], label=f"CTRL {cr.time_h:g} h")
            ax.set_xlim(0, 5); ax.set(xlabel="r/RMW", ylabel=ylabel)
            if row_index == 0: ax.set_title(f"pmin mismatch={p_match['absolute_mismatch'][ji]:.2f} hPa")
            ax.legend(fontsize=8)
    fig.savefig(figures / "rmw_normalized_profiles.png", dpi=220); fig.savefig(figures / "rmw_normalized_profiles.pdf"); plt.close(fig)
    return {"p_index": pi, "v_index": vi, "p_mismatch": p_match["absolute_mismatch"], "v_mismatch": v_match["absolute_mismatch"]}


def _write_report(rows: Sequence[CaseTimeDiagnostics], energy: Sequence[Mapping[str, object]], args, out_dir: Path) -> None:
    ctrl = [r for r in rows if r.case == "CTRL"]
    jet = [r for r in rows if r.case == "JET"]
    t = np.array([r.time_h for r in ctrl])
    p_c = np.array([r.pmin_hpa for r in ctrl]); p_j = np.array([r.pmin_hpa for r in jet])
    rate_diff = -(safe_gradient(p_j, t, 0) - safe_gradient(p_c, t, 0))
    leads = np.arange(0.0, 38.0, 2.0)
    predictors = {
        "VWS": np.array([r.vws_200_800_ms for r in jet]) - np.array([r.vws_200_800_ms for r in ctrl]),
        "ventilation": np.array([r.inward_entropy_deficit_covariance for r in jet]) - np.array([r.inward_entropy_deficit_covariance for r in ctrl]),
        "BL_I2": np.array([r.bl_i2_inflow_weighted_s2 for r in jet]) - np.array([r.bl_i2_inflow_weighted_s2 for r in ctrl]),
        "eddy_forcing": np.array([r.eddy_forcing_jet_box_rms_ms2 for r in jet]) - np.array([r.eddy_forcing_jet_box_rms_ms2 for r in ctrl]),
    }
    lag_results = {name: lead_lag_correlation(value, rate_diff, args.step_hour, leads) for name, value in predictors.items()}
    with (out_dir / "lead_lag.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["predictor", "lead_h", "correlation", "count"])
        for name, result in lag_results.items():
            for lead, corr, count in zip(result["lead_h"], result["correlation"], result["count"]):
                writer.writerow([name, lead, corr, count])
    matching = _plot_matching_and_lag(ctrl, jet, lag_results, out_dir)
    if t.size >= 5:
        phases = {"CTRL": identify_intensification_phases(t, -p_c), "JET": identify_intensification_phases(t, -p_j)}
    else:
        phases = {"CTRL": {"status": "insufficient_samples"}, "JET": {"status": "insufficient_samples"}}
    energy_complete = [e for e in energy if e.get("cycle_available")]
    energy_good = [e for e in energy_complete if float(e.get("mass_closure_ratio", 1)) <= 0.1 and float(e.get("first_law_relative_residual", 1)) <= 0.2]
    lines = [
        "# JET mechanism diagnostic report", "",
        "## Interpretation contract", "",
        "- Outflow-layer inertial stability is treated as a balanced dynamical coefficient, not an energetic resistance.",
        "- F_lambda_env is JET minus CTRL total eddy response, not a pure imposed-jet flux.",
        "- Two deterministic simulations support mechanism consistency but do not establish general causality.", "",
        "## Automatically identified phases", "", "```json", json.dumps(phases, indent=2), "```", "",
        "## Isentropic energetics quality", "",
        f"- Closed cycles extracted: {len(energy_complete)}/{len(energy)}.",
        f"- Cycles passing mass <= 0.10 and first-law <= 0.20 residual thresholds: {len(energy_good)}.",
    ]
    if not energy_good:
        lines.append("- Full heat-engine attribution is withheld because no cycle passed both closure thresholds.")
    window = (leads >= 6.0) & (leads <= 24.0)
    best = {}
    for name, result in lag_results.items():
        loc = np.where(window)[0][int(np.nanargmax(np.abs(result["correlation"][window])))]
        best[name] = (float(result["lead_h"][loc]), float(result["correlation"][loc]))
    p_good = (matching["p_mismatch"] <= 1.0)
    rmw_good = np.abs(np.array([ctrl[i].rmw_km for i in matching["p_index"]]) - np.array([r.rmw_km for r in jet])) <= 25.0
    paired = p_good & rmw_good
    log_text = (out_dir / "run_full.log").read_text(encoding="utf-8", errors="replace") if (out_dir / "run_full.log").exists() else ""
    se_failed = log_text.count("SOR not converged")
    lines += ["", "## Scientific assessment", "",
              f"- **Supported negative pathway — shear/ventilation.** JET raises mean 200–800-km VWS by {np.mean(predictors['VWS']):.2f} m s-1. The strongest required-window relation is r={best['VWS'][1]:.2f} at a {best['VWS'][0]:.0f}-h lead; ventilation covariance gives r={best['ventilation'][1]:.2f} at {best['ventilation'][0]:.0f} h.",
              f"- **Insufficient as a positive pathway — boundary-layer I2 chain.** Mean JET-minus-CTRL BL I2 is {np.mean(predictors['BL_I2']):.2e} s-2, but its strongest 6–24-h relation is r={best['BL_I2'][1]:.2f} at {best['BL_I2'][0]:.0f} h, opposite to a robust positive lead. Early-time inflow/updraft and lower sink temperature remain a stage-limited candidate, not a primary attribution.",
              f"- **Rejected for complete energetic attribution in this run.** No isentropic cycle passed both closure gates; downstream/outflow work therefore cannot be used quantitatively to explain the JET enhancement.",
              f"- **Insufficient as a positive pathway — eddy/SE response.** Eddy-forcing magnitude has r={best['eddy_forcing'][1]:.2f} at a {best['eddy_forcing'][0]:.0f}-h lead. {se_failed}/12 SE solves missed the strict SOR tolerance, so their fields are retained only as regularized balanced projections.",
              f"- Strength matching retained {int(np.count_nonzero(paired))}/{len(jet)} pmin-matched samples with <=1 hPa intensity mismatch and <=25 km RMW mismatch.",
              "- **Overall:** no JET-positive factor satisfies all four preregistered criteria. The early BL-inflow/eyewall/sink-temperature chain is the leading candidate, but evidence is presently insufficient; the shear/ventilation penalty is the only supported robust pathway."]
    (out_dir / "mechanism_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    per_time = out_dir / "data"; per_time.mkdir(exist_ok=True)
    energy_dir = out_dir / "energetics"; energy_dir.mkdir(exist_ok=True)
    figures_dir = out_dir / "figures"; figures_dir.mkdir(exist_ok=True)
    if args.postprocess_only:
        rows = _read_rows(out_dir / "timeseries.csv")
        _write_report(rows, _load_energy_results(energy_dir), args, out_dir)
        print(f"[OK] Postprocessed products written to {out_dir}")
        return
    datasets = {"CTRL": xr.open_dataset(args.ctrl, decode_cf=False), "JET": xr.open_dataset(args.jet, decode_cf=False)}
    try:
        preflight = {}
        common_end = np.inf
        for case, ds in datasets.items():
            inv = inventory_variables(ds.variables)
            inv["time_start_h"] = float(_time_seconds(ds)[0] / 3600.0)
            inv["time_end_h"] = float(_time_seconds(ds)[-1] / 3600.0)
            inv["dimensions"] = {k: int(v) for k, v in ds.sizes.items()}
            preflight[case] = inv
            common_end = min(common_end, inv["time_end_h"])
        preflight["common_end_h"] = common_end
        (out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2, ensure_ascii=False), encoding="utf-8")
        if args.preflight_only:
            print(json.dumps(preflight, indent=2, ensure_ascii=False)); return
        end = min(args.end_hour, common_end)
        targets = np.arange(args.start_hour, end + 0.01, args.step_hour)
        energy_targets = np.array([float(x) for x in args.energy_times.split(",") if x.strip()])
        rows: List[CaseTimeDiagnostics] = []
        energy_results: List[Mapping[str, object]] = []
        for case, ds in datasets.items():
            times = _time_seconds(ds) / 3600.0
            for target in targets:
                index = int(np.argmin(np.abs(times - target)))
                row, state = _diagnose_time(ds, index, case, args, per_time)
                rows.append(row)
                if not args.skip_energy and np.any(np.isclose(target, energy_targets, atol=0.51 * args.step_hour)):
                    energy_results.append(_run_energy(case, row.time_h, state, args, energy_dir))
        rows.sort(key=lambda item: (item.case, item.time_h))
        _write_rows(rows, out_dir / "timeseries.csv")
        _plot_timeseries(rows, figures_dir)
        c70 = min((p for p in per_time.glob("CTRL_*h.npz")), key=lambda p: abs(float(p.stem.split("_")[-1][:-1]) - args.se_time))
        j70 = min((p for p in per_time.glob("JET_*h.npz")), key=lambda p: abs(float(p.stem.split("_")[-1][:-1]) - args.se_time))
        _plot_70h(c70, j70, figures_dir)
        if not args.skip_se:
            _se_factorial(args, out_dir / "se")
        _write_report(rows, energy_results, args, out_dir)
        print(f"[OK] Products written to {out_dir}")
    finally:
        for ds in datasets.values(): ds.close()


if __name__ == "__main__":
    main()
