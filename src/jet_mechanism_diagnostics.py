"""Common diagnostics for separating environmental-jet pathways.

The functions in this module are deliberately independent of CM1 I/O.  They
operate on scalar-grid arrays and make the sign conventions explicit:

* radial velocity is positive outward;
* tangential velocity is positive cyclonic;
* positive ``ventilation_flux`` is inward transport of entropy deficit;
* a positive lead means that the predictor leads the intensity tendency.

Outflow-layer inertial stability is treated only as a dynamical coefficient.
No function in this module assigns it an energetic-resistance interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


RD = 287.04
CPD = 1004.0
P0 = 100000.0
EPSILON = 0.622


def safe_gradient(field: np.ndarray, coord: np.ndarray, axis: int) -> np.ndarray:
    """Return a coordinate-aware gradient, including two-point grids."""
    arr = np.asarray(field, dtype=np.float64)
    x = np.asarray(coord, dtype=np.float64)
    if arr.shape[axis] != x.size:
        raise ValueError("coordinate length does not match field axis")
    if x.size < 2:
        return np.zeros_like(arr)
    return np.gradient(arr, x, axis=axis, edge_order=2 if x.size > 2 else 1)


def storm_relative_geometry(
    x_m: np.ndarray, y_m: np.ndarray, center_x_m: float, center_y_m: float
) -> Dict[str, np.ndarray]:
    """Build radius, azimuth and unit vectors on a Cartesian scalar grid."""
    xx, yy = np.meshgrid(np.asarray(x_m, float), np.asarray(y_m, float))
    dx = xx - float(center_x_m)
    dy = yy - float(center_y_m)
    radius = np.hypot(dx, dy)
    azimuth = np.arctan2(dy, dx)
    return {
        "radius_m": radius,
        "azimuth_rad": azimuth,
        "cos_azimuth": np.cos(azimuth),
        "sin_azimuth": np.sin(azimuth),
    }


def cylindrical_wind(
    u: np.ndarray, v: np.ndarray, cos_azimuth: np.ndarray, sin_azimuth: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian winds to outward-radial and cyclonic-tangential wind."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    c = np.asarray(cos_azimuth, dtype=np.float64)
    s = np.asarray(sin_azimuth, dtype=np.float64)
    if u.shape != v.shape or u.shape[-2:] != c.shape or c.shape != s.shape:
        raise ValueError("wind and azimuth arrays have incompatible shapes")
    return u * c + v * s, -u * s + v * c


def radial_bin_indices(radius_m: np.ndarray, edges_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return flattened radial-bin indices and their valid mask."""
    edges = np.asarray(edges_m, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("radial edges must be strictly increasing")
    index = np.digitize(np.asarray(radius_m).ravel(), edges) - 1
    valid = (index >= 0) & (index < edges.size - 1)
    return index, valid


def radial_mean(
    field: np.ndarray, bin_index: np.ndarray, valid: np.ndarray, nr: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Azimuthally average ``(..., y, x)`` onto ``(..., r)`` bins."""
    arr = np.asarray(field, dtype=np.float64)
    lead_shape = arr.shape[:-2]
    flat = arr.reshape((-1, arr.shape[-2] * arr.shape[-1]))
    if bin_index.size != flat.shape[1] or valid.size != flat.shape[1]:
        raise ValueError("radial mapping does not match horizontal field size")
    if weights is None:
        weight_flat = np.ones_like(flat)
    else:
        w = np.asarray(weights, dtype=np.float64)
        weight_flat = np.broadcast_to(w, arr.shape).reshape(flat.shape)
    out = np.full((flat.shape[0], nr), np.nan)
    for n in range(flat.shape[0]):
        use = valid & np.isfinite(flat[n]) & np.isfinite(weight_flat[n]) & (weight_flat[n] > 0)
        if not np.any(use):
            continue
        idx = bin_index[use]
        den = np.bincount(idx, weights=weight_flat[n, use], minlength=nr)
        num = np.bincount(idx, weights=weight_flat[n, use] * flat[n, use], minlength=nr)
        out[n] = np.divide(num, den, out=np.full(nr, np.nan), where=den > 0)
    return out.reshape(lead_shape + (nr,))


def angular_momentum_inertial_stability(
    vt_zr: np.ndarray, r_m: np.ndarray, f: float
) -> Dict[str, np.ndarray]:
    """Return M, dM/dr, absolute vorticity and classic axisymmetric I squared."""
    vt = np.asarray(vt_zr, dtype=np.float64)
    r = np.asarray(r_m, dtype=np.float64)
    if vt.shape[-1] != r.size:
        raise ValueError("last vt dimension must be radius")
    dr0 = float(np.nanmedian(np.diff(r))) if r.size > 1 else 1.0
    r_safe = np.maximum(r, 0.5 * max(dr0, 1.0))
    absolute_momentum = vt * r_safe + 0.5 * float(f) * r_safe**2
    d_m_dr = safe_gradient(absolute_momentum, r, axis=-1)
    zeta_abs = d_m_dr / r_safe
    velocity_factor = float(f) + 2.0 * vt / r_safe
    i2 = velocity_factor * zeta_abs
    return {
        "M": absolute_momentum,
        "dM_dr": d_m_dr,
        "zeta_absolute": zeta_abs,
        "velocity_factor": velocity_factor,
        "I2": i2,
    }


def mass_flux_weighted_mean(
    field_zr: np.ndarray,
    rho_zr: np.ndarray,
    velocity_zr: np.ndarray,
    r_m: np.ndarray,
    z_m: np.ndarray,
    r_bounds_m: Tuple[float, float],
    z_bounds_m: Tuple[float, float],
    positive_velocity: bool = True,
) -> float:
    """Mass-flux-weight a field in an r-z box."""
    field = np.asarray(field_zr, float)
    rho = np.asarray(rho_zr, float)
    vel = np.asarray(velocity_zr, float)
    r = np.asarray(r_m, float)
    z = np.asarray(z_m, float)
    if not (field.shape == rho.shape == vel.shape == (z.size, r.size)):
        raise ValueError("field/rho/velocity must have shape (z, r)")
    rr = r[None, :]
    mask = (
        (rr >= r_bounds_m[0]) & (rr <= r_bounds_m[1])
        & (z[:, None] >= z_bounds_m[0]) & (z[:, None] <= z_bounds_m[1])
    )
    transport = rho * rr * (np.maximum(vel, 0.0) if positive_velocity else np.maximum(-vel, 0.0))
    use = mask & np.isfinite(field) & np.isfinite(transport)
    den = float(np.sum(transport[use]))
    return float(np.sum(field[use] * transport[use]) / den) if den > 0 else np.nan


def interpolate_to_pressure(
    field_zyx: np.ndarray, pressure_zyx_pa: np.ndarray, target_pa: float
) -> np.ndarray:
    """Vectorized columnwise linear interpolation to a pressure surface."""
    field = np.asarray(field_zyx, float)
    pressure = np.asarray(pressure_zyx_pa, float)
    if field.shape != pressure.shape or field.ndim != 3:
        raise ValueError("field and pressure must share shape (z, y, x)")
    crossing = (
        np.isfinite(pressure[:-1]) & np.isfinite(pressure[1:])
        & np.isfinite(field[:-1]) & np.isfinite(field[1:])
        & ((pressure[:-1] - target_pa) * (pressure[1:] - target_pa) <= 0.0)
    )
    has_crossing = np.any(crossing, axis=0)
    lower_index = np.argmax(crossing, axis=0)
    jj, ii = np.indices(has_crossing.shape)
    p0 = pressure[lower_index, jj, ii]
    p1 = pressure[lower_index + 1, jj, ii]
    q0 = field[lower_index, jj, ii]
    q1 = field[lower_index + 1, jj, ii]
    fraction = np.divide(
        target_pa - p0, p1 - p0,
        out=np.zeros_like(p0, dtype=float), where=np.abs(p1 - p0) > 1.0e-12,
    )
    out = q0 + fraction * (q1 - q0)
    out[~has_crossing] = np.nan
    return out


def annulus_vector_mean(
    u_2d: np.ndarray, v_2d: np.ndarray, radius_m: np.ndarray, bounds_m: Tuple[float, float]
) -> Tuple[float, float]:
    """Area-sampled environmental vector mean in an annulus."""
    mask = (radius_m >= bounds_m[0]) & (radius_m <= bounds_m[1])
    mask &= np.isfinite(u_2d) & np.isfinite(v_2d)
    if not np.any(mask):
        return np.nan, np.nan
    return float(np.mean(u_2d[mask])), float(np.mean(v_2d[mask]))


def bulk_vertical_wind_shear(
    u850: np.ndarray, v850: np.ndarray, u200: np.ndarray, v200: np.ndarray,
    radius_m: np.ndarray, bounds_m: Tuple[float, float],
) -> Dict[str, float]:
    """Return annulus-mean 850-200-hPa vector shear."""
    u_lo, v_lo = annulus_vector_mean(u850, v850, radius_m, bounds_m)
    u_hi, v_hi = annulus_vector_mean(u200, v200, radius_m, bounds_m)
    du, dv = u_hi - u_lo, v_hi - v_lo
    return {"du": du, "dv": dv, "magnitude": float(np.hypot(du, dv))}


def ventilation_index(shear_ms: float, entropy_deficit: float, potential_intensity_ms: float) -> float:
    """Tang-Emanuel nondimensional ventilation index."""
    if not np.isfinite(potential_intensity_ms) or potential_intensity_ms <= 0:
        return np.nan
    return float(shear_ms) * float(entropy_deficit) / float(potential_intensity_ms)


def inward_entropy_deficit_flux(
    ur_prime: np.ndarray,
    deficit_prime: np.ndarray,
    rho: np.ndarray,
    cell_area_m2: float | np.ndarray,
    layer_thickness_m: float | np.ndarray,
    mask: np.ndarray,
) -> float:
    """Positive inward eddy transport of entropy deficit (kg K-like s-1)."""
    ur = np.asarray(ur_prime, float)
    deficit = np.asarray(deficit_prime, float)
    rho = np.asarray(rho, float)
    use = np.asarray(mask, bool) & np.isfinite(ur) & np.isfinite(deficit) & np.isfinite(rho)
    flux = -rho * ur * deficit * np.asarray(cell_area_m2) * np.asarray(layer_thickness_m)
    return float(np.sum(flux[use]))


def match_by_intensity(
    target: np.ndarray, reference: np.ndarray, reference_time_h: np.ndarray
) -> Dict[str, np.ndarray]:
    """Nearest-neighbour intensity matching with absolute mismatch metadata."""
    target = np.asarray(target, float)
    reference = np.asarray(reference, float)
    tref = np.asarray(reference_time_h, float)
    if reference.shape != tref.shape:
        raise ValueError("reference intensity and time must share shape")
    indices = np.full(target.shape, -1, dtype=int)
    mismatch = np.full(target.shape, np.nan)
    matched_time = np.full(target.shape, np.nan)
    good_ref = np.isfinite(reference) & np.isfinite(tref)
    for idx in np.ndindex(target.shape):
        if not np.isfinite(target[idx]) or not np.any(good_ref):
            continue
        choices = np.where(good_ref)[0]
        j = choices[int(np.argmin(np.abs(reference[good_ref] - target[idx])))]
        indices[idx] = j
        mismatch[idx] = abs(reference[j] - target[idx])
        matched_time[idx] = tref[j]
    return {"index": indices, "time_h": matched_time, "absolute_mismatch": mismatch}


def lead_lag_correlation(
    predictor: np.ndarray,
    response: np.ndarray,
    dt_h: float,
    leads_h: Sequence[float],
) -> Dict[str, np.ndarray]:
    """Correlation of predictor(t) with response(t+lead)."""
    x = np.asarray(predictor, float)
    y = np.asarray(response, float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("predictor and response must be one-dimensional and aligned")
    correlations = np.full(len(leads_h), np.nan)
    counts = np.zeros(len(leads_h), dtype=int)
    for n, lead in enumerate(leads_h):
        shift = int(round(float(lead) / float(dt_h)))
        if abs(shift) >= x.size:
            continue
        if shift >= 0:
            xx, yy = x[: x.size - shift or None], y[shift:]
        else:
            xx, yy = x[-shift:], y[: y.size + shift]
        good = np.isfinite(xx) & np.isfinite(yy)
        counts[n] = np.count_nonzero(good)
        if counts[n] >= 3 and np.nanstd(xx[good]) > 0 and np.nanstd(yy[good]) > 0:
            correlations[n] = np.corrcoef(xx[good], yy[good])[0, 1]
    return {"lead_h": np.asarray(leads_h, float), "correlation": correlations, "count": counts}


def identify_intensification_phases(
    time_h: np.ndarray, intensity: np.ndarray, stronger_is_larger: bool = True
) -> Dict[str, float]:
    """Identify onset, peak 12-h tendency and late suppression from one series."""
    t = np.asarray(time_h, float)
    x = np.asarray(intensity, float)
    if t.ndim != 1 or x.shape != t.shape or t.size < 5:
        raise ValueError("time and intensity need at least five aligned samples")
    tendency = safe_gradient(x, t, axis=0)
    if not stronger_is_larger:
        tendency = -tendency
    finite = np.isfinite(tendency)
    if not np.any(finite):
        return {"onset_h": np.nan, "peak_rate_h": np.nan, "late_h": np.nan}
    threshold = 0.25 * float(np.nanmax(tendency))
    onset_candidates = np.where(finite & (tendency >= threshold))[0]
    onset = int(onset_candidates[0]) if onset_candidates.size else int(np.nanargmax(tendency))
    peak = int(np.nanargmax(tendency))
    late_candidates = np.where((np.arange(t.size) > peak) & finite & (tendency <= 0))[0]
    late = int(late_candidates[0]) if late_candidates.size else t.size - 1
    return {"onset_h": float(t[onset]), "peak_rate_h": float(t[peak]), "late_h": float(t[late])}


def inventory_variables(variable_names: Iterable[str]) -> Dict[str, object]:
    """Classify data capability without silently treating missing fields as zero."""
    names = set(variable_names)
    groups = {
        "dynamics": ({"u", "v", "w", "psfc"},),
        "thermodynamics": ({"th", "prs", "rho", "qv"}, {"theta", "pres", "rhoa", "qvpert"}),
        "hydrometeors": ({"qc", "qr", "qi", "qs", "qg"},),
    }
    result: Dict[str, object] = {"variables": sorted(names)}
    result["has_dynamics"] = all(v in names for v in groups["dynamics"][0])
    result["has_thermodynamics"] = all(v in names for v in groups["thermodynamics"][0])
    result["hydrometeors_present"] = sorted(groups["hydrometeors"][0] & names)
    result["momentum_budget_present"] = sorted(v for v in names if v.startswith(("ub_", "vb_", "wb_")))
    result["thermal_budget_present"] = sorted(v for v in names if v.startswith(("ptb_", "thb_", "qvb_")))
    return result
