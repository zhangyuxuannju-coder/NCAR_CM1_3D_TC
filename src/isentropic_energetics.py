"""Isentropic mean-circulation and thermodynamic-cycle diagnostics.

This is a MAFALDA-style implementation following the central construction of
Pauluis and Zhang (2017) and Li, Wang and Tan (2023): vertical mass transport
is conditionally sampled in equivalent-potential-temperature/height space,
and closed streamfunction contours are used as mean parcel cycles.

The module intentionally reports closure residuals.  The pressure-work and
``T ds`` integrals below are diagnostics of the resolved moist cycle; they are
not silently promoted to a complete energy budget when condensate/Gibbs or
unresolved tendency terms are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


RD = 287.04
RV = 461.5
CPD = 1004.0
CPV = 1850.0
LV0 = 2.5008e6
P0 = 100000.0
EPSILON = RD / RV
G = 9.80665


def saturation_vapor_pressure_pa(temperature_k: np.ndarray) -> np.ndarray:
    """Bolton saturation vapor pressure over liquid water."""
    t_c = np.asarray(temperature_k, float) - 273.15
    return 611.2 * np.exp(17.67 * t_c / np.maximum(t_c + 243.5, 1.0))


def temperature_from_potential_temperature(
    theta_k: np.ndarray, pressure_pa: np.ndarray
) -> np.ndarray:
    """Convert dry potential temperature to temperature."""
    return np.asarray(theta_k, float) * (np.asarray(pressure_pa, float) / P0) ** (RD / CPD)


def dewpoint_from_vapor_mixing_ratio(
    qv_kgkg: np.ndarray, pressure_pa: np.ndarray
) -> np.ndarray:
    """Dewpoint from water-vapor mixing ratio using Bolton's inversion."""
    qv = np.maximum(np.asarray(qv_kgkg, float), 1.0e-10)
    p = np.asarray(pressure_pa, float)
    vapor_pressure = p * qv / (EPSILON + qv)
    log_ratio = np.log(np.maximum(vapor_pressure, 1.0) / 611.2)
    return 273.15 + 243.5 * log_ratio / (17.67 - log_ratio)


def equivalent_potential_temperature(
    theta_k: np.ndarray, pressure_pa: np.ndarray, qv_kgkg: np.ndarray
) -> np.ndarray:
    """Bolton equivalent potential temperature suitable for conditional bins."""
    theta = np.asarray(theta_k, float)
    p = np.asarray(pressure_pa, float)
    qv = np.maximum(np.asarray(qv_kgkg, float), 0.0)
    temperature = temperature_from_potential_temperature(theta, p)
    dewpoint = dewpoint_from_vapor_mixing_ratio(qv, p)
    tlcl = 1.0 / (
        1.0 / np.maximum(dewpoint - 56.0, 1.0)
        + np.log(np.maximum(temperature, 100.0) / np.maximum(dewpoint, 100.0)) / 800.0
    ) + 56.0
    mixing_gkg = 1000.0 * qv
    theta_l = temperature * (P0 / np.maximum(p, 1000.0)) ** (
        0.2854 * (1.0 - 0.28e-3 * mixing_gkg)
    )
    return theta_l * np.exp(
        (3.376 / np.maximum(tlcl, 100.0) - 0.00254)
        * mixing_gkg
        * (1.0 + 0.81e-3 * mixing_gkg)
    )


def moist_entropy_proxy(theta_e_k: np.ndarray) -> np.ndarray:
    """Entropy coordinate CPD ln(theta_e); an additive reference is irrelevant."""
    return CPD * np.log(np.maximum(np.asarray(theta_e_k, float), 1.0))


def _conditional_sum_and_mean(
    coordinate: np.ndarray,
    edges: np.ndarray,
    extensive: np.ndarray,
    intensive: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray | None]:
    index = np.digitize(coordinate, edges) - 1
    nbins = edges.size - 1
    good = (
        np.isfinite(coordinate) & np.isfinite(extensive)
        & (index >= 0) & (index < nbins)
    )
    total = np.bincount(index[good], weights=extensive[good], minlength=nbins)
    if intensive is None:
        return total, None
    value = np.asarray(intensive, float)
    good &= np.isfinite(value)
    weight = np.abs(extensive)
    den = np.bincount(index[good], weights=weight[good], minlength=nbins)
    num = np.bincount(index[good], weights=weight[good] * value[good], minlength=nbins)
    mean = np.divide(num, den, out=np.full(nbins, np.nan), where=den > 0)
    return total, mean


def build_isentropic_streamfunction(
    theta_e_zyx: np.ndarray,
    w_zyx: np.ndarray,
    rho_zyx: np.ndarray,
    z_m: np.ndarray,
    theta_e_edges_k: np.ndarray,
    cell_area_m2: float | np.ndarray,
    state_fields: Mapping[str, np.ndarray] | None = None,
    horizontal_mask: np.ndarray | None = None,
) -> Dict[str, np.ndarray | float]:
    """Construct the theta-e/z mass streamfunction and conditional state.

    ``vertical_mass_flux[z, bin]`` is the domain-integrated mass transport in
    each theta-e class.  Its cumulative sum from low to high theta-e defines
    the streamfunction.  A closed-domain quality metric is returned rather
    than assumed.
    """
    theta_e = np.asarray(theta_e_zyx, float)
    w = np.asarray(w_zyx, float)
    rho = np.asarray(rho_zyx, float)
    z = np.asarray(z_m, float)
    edges = np.asarray(theta_e_edges_k, float)
    if not (theta_e.shape == w.shape == rho.shape) or theta_e.ndim != 3:
        raise ValueError("theta_e, w and rho must share shape (z, y, x)")
    if theta_e.shape[0] != z.size or np.any(np.diff(edges) <= 0):
        raise ValueError("vertical coordinate or theta-e edges are incompatible")
    area = np.broadcast_to(np.asarray(cell_area_m2, float), theta_e.shape[1:])
    mask2d = np.ones(theta_e.shape[1:], bool) if horizontal_mask is None else np.asarray(horizontal_mask, bool)
    if mask2d.shape != theta_e.shape[1:]:
        raise ValueError("horizontal_mask must match (y, x)")

    nbins = edges.size - 1
    flux = np.zeros((z.size, nbins), dtype=np.float64)
    occupancy = np.zeros_like(flux)
    states: Dict[str, np.ndarray] = {}
    for name in (state_fields or {}):
        states[name] = np.full_like(flux, np.nan)

    for k in range(z.size):
        use = mask2d & np.isfinite(theta_e[k]) & np.isfinite(w[k]) & np.isfinite(rho[k])
        coord = theta_e[k][use]
        mass_flux = rho[k][use] * w[k][use] * area[use]
        flux[k], _ = _conditional_sum_and_mean(coord, edges, mass_flux)
        occupancy[k], _ = _conditional_sum_and_mean(
            coord, edges, np.maximum(rho[k][use] * area[use], 0.0)
        )
        for name, field in (state_fields or {}).items():
            arr = np.asarray(field, float)
            if arr.shape != theta_e.shape:
                raise ValueError(f"state field {name!r} does not match theta_e")
            _, states[name][k] = _conditional_sum_and_mean(
                coord, edges, mass_flux, arr[k][use]
            )

    streamfunction = np.cumsum(flux, axis=1)
    net_vertical = np.sum(flux, axis=1)
    gross_vertical = np.sum(np.abs(flux), axis=1)
    closure = float(
        np.sqrt(np.nanmean(net_vertical**2))
        / max(np.sqrt(np.nanmean(gross_vertical**2)), 1.0e-20)
    )
    result: Dict[str, np.ndarray | float] = {
        "theta_e_center_k": 0.5 * (edges[:-1] + edges[1:]),
        "z_m": z,
        "vertical_mass_flux_kg_s": flux,
        "streamfunction_kg_s": streamfunction,
        "occupancy_kg_m2": occupancy,
        "net_vertical_mass_flux_kg_s": net_vertical,
        "mass_closure_ratio": closure,
    }
    result.update({f"conditional_{name}": value for name, value in states.items()})
    return result


def extract_closed_streamfunction_contour(
    theta_e_center_k: np.ndarray,
    z_m: np.ndarray,
    streamfunction_ztheta: np.ndarray,
    level_fraction: float = 0.5,
) -> Dict[str, np.ndarray | float]:
    """Extract the longest closed contour at a fraction of the dominant cell."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    theta = np.asarray(theta_e_center_k, float)
    z = np.asarray(z_m, float)
    psi = np.asarray(streamfunction_ztheta, float)
    if psi.shape != (z.size, theta.size):
        raise ValueError("streamfunction must have shape (z, theta_e)")
    if not 0.0 < level_fraction < 1.0:
        raise ValueError("level_fraction must be between zero and one")
    pos = float(np.nanmax(psi))
    neg = float(np.nanmin(psi))
    extreme = pos if abs(pos) >= abs(neg) else neg
    if not np.isfinite(extreme) or abs(extreme) <= 0:
        raise ValueError("streamfunction has no finite circulation cell")
    level = level_fraction * extreme
    fig = Figure()
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    contour = ax.contour(theta, z, psi, levels=[level])
    candidates = []
    paths = contour.get_paths() if hasattr(contour, "get_paths") else [p for c in contour.collections for p in c.get_paths()]
    for path in paths:
        vertices = np.asarray(path.vertices, float)
        if vertices.shape[0] < 8:
            continue
        scale = max(np.ptp(theta), np.ptp(z), 1.0)
        closed = np.linalg.norm(vertices[0] - vertices[-1]) <= 0.02 * scale
        if closed:
            candidates.append(vertices)
    fig.clear()
    if not candidates:
        raise ValueError("no closed mean-circulation contour found at requested level")
    vertices = max(candidates, key=lambda item: item.shape[0])
    if not np.allclose(vertices[0], vertices[-1]):
        vertices = np.vstack([vertices, vertices[0]])
    return {"theta_e_k": vertices[:, 0], "z_m": vertices[:, 1], "level_kg_s": level}


def sample_conditional_field(
    theta_grid_k: np.ndarray,
    z_grid_m: np.ndarray,
    field_ztheta: np.ndarray,
    theta_path_k: np.ndarray,
    z_path_m: np.ndarray,
) -> np.ndarray:
    """Bilinearly sample a conditional mean along a cycle path."""
    from scipy.interpolate import RegularGridInterpolator

    interpolator = RegularGridInterpolator(
        (np.asarray(z_grid_m, float), np.asarray(theta_grid_k, float)),
        np.asarray(field_ztheta, float), bounds_error=False, fill_value=np.nan,
    )
    return interpolator(np.column_stack([z_path_m, theta_path_k]))


def _classify_segments(z0: np.ndarray, z1: np.ndarray) -> np.ndarray:
    zmid = 0.5 * (z0 + z1)
    dz = z1 - z0
    labels = np.full(zmid.shape, "return", dtype="U20")
    labels[zmid <= 2000.0] = "boundary_layer"
    labels[(zmid > 2000.0) & (dz > 25.0)] = "eyewall_ascent"
    labels[(zmid >= 10000.0) & (np.abs(dz) <= 250.0)] = "upper_outflow"
    labels[(zmid > 2000.0) & (dz < -25.0)] = "descent"
    return labels


def integrate_thermodynamic_cycle(
    temperature_k: np.ndarray,
    pressure_pa: np.ndarray,
    entropy_j_kg_k: np.ndarray,
    z_m: np.ndarray,
    qv_kgkg: np.ndarray | None = None,
) -> Dict[str, object]:
    """Integrate resolved pressure work and T ds around a closed path."""
    t = np.asarray(temperature_k, float)
    p = np.asarray(pressure_pa, float)
    s = np.asarray(entropy_j_kg_k, float)
    z = np.asarray(z_m, float)
    if not (t.shape == p.shape == s.shape == z.shape) or t.ndim != 1:
        raise ValueError("cycle arrays must be aligned one-dimensional vectors")
    if t.size < 5:
        raise ValueError("cycle needs at least five points")
    if not (np.isclose(z[0], z[-1]) and np.isclose(s[0], s[-1], rtol=0, atol=5.0)):
        t = np.r_[t, t[0]]; p = np.r_[p, p[0]]; s = np.r_[s, s[0]]; z = np.r_[z, z[0]]
    qv = np.zeros_like(t) if qv_kgkg is None else np.asarray(qv_kgkg, float)
    if qv.shape != t.shape:
        if qv.size == t.size - 1:
            qv = np.r_[qv, qv[0]]
        else:
            raise ValueError("qv must match the closed cycle")
    valid = np.isfinite(t) & np.isfinite(p) & np.isfinite(s) & np.isfinite(z) & np.isfinite(qv)
    if np.count_nonzero(valid) < 5:
        raise ValueError("conditional cycle contains too many missing values")
    # Retain path order and fill isolated missing conditional bins by index interpolation.
    idx = np.arange(t.size)
    for arr in (t, p, s, z, qv):
        bad = ~np.isfinite(arr)
        if np.any(bad):
            arr[bad] = np.interp(idx[bad], idx[~bad], arr[~bad])

    tmid = 0.5 * (t[:-1] + t[1:])
    pmid = np.maximum(0.5 * (p[:-1] + p[1:]), 1000.0)
    qmid = np.maximum(0.5 * (qv[:-1] + qv[1:]), 0.0)
    alpha = RD * tmid * (1.0 + qmid / EPSILON) / pmid
    pressure_work = -alpha * np.diff(p)
    heat = tmid * np.diff(s)
    labels = _classify_segments(z[:-1], z[1:])
    branches: Dict[str, Dict[str, float]] = {}
    for label in ("boundary_layer", "eyewall_ascent", "upper_outflow", "descent", "return"):
        use = labels == label
        branches[label] = {
            "pressure_work_j_kg": float(np.sum(pressure_work[use])),
            "heat_j_kg": float(np.sum(heat[use])),
            "segment_count": int(np.count_nonzero(use)),
        }
    heat_in = float(np.sum(np.maximum(heat, 0.0)))
    heat_out = float(-np.sum(np.minimum(heat, 0.0)))
    source_temp = float(np.sum(tmid * np.maximum(heat, 0.0)) / heat_in) if heat_in > 0 else np.nan
    sink_temp = float(np.sum(tmid * np.maximum(-heat, 0.0)) / heat_out) if heat_out > 0 else np.nan
    net_work = float(np.sum(pressure_work))
    theoretical_max = heat_in * (1.0 - sink_temp / source_temp) if source_temp > 0 and sink_temp > 0 else np.nan
    gross_work = float(np.sum(np.abs(pressure_work)))
    outflow_fraction = (
        abs(branches["upper_outflow"]["pressure_work_j_kg"]) / gross_work
        if gross_work > 0 else np.nan
    )
    net_heat = float(np.sum(heat))
    return {
        "pressure_work_j_kg": net_work,
        "heat_input_j_kg": heat_in,
        "heat_rejection_j_kg": heat_out,
        "net_tds_j_kg": net_heat,
        "source_temperature_k": source_temp,
        "sink_temperature_k": sink_temp,
        "carnot_efficiency": float(1.0 - sink_temp / source_temp) if source_temp > 0 else np.nan,
        "resolved_efficiency": net_work / heat_in if heat_in > 0 else np.nan,
        "theoretical_max_work_j_kg": theoretical_max,
        "outflow_gross_work_fraction": outflow_fraction,
        "first_law_residual_j_kg": net_heat - net_work,
        "first_law_relative_residual": abs(net_heat - net_work) / max(heat_in, abs(net_work), 1.0),
        "branches": branches,
        "segment_labels": labels,
    }
