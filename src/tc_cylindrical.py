"""Regular-azimuth sampling for TC-centred CM1 diagnostics.

Fields are linearly interpolated from the scalar Cartesian grid to a regular
set of azimuths and physical radii.  This avoids treating the number of native
Cartesian cells in an annulus as an angular quadrature weight.  The routines
return coverage explicitly; callers must not silently treat incomplete circles
as full azimuthal means.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import map_coordinates


def regular_polar_geometry(
    xh_m: np.ndarray,
    yh_m: np.ndarray,
    center_x_m: float,
    center_y_m: float,
    radii_m: np.ndarray,
    n_azimuth: int,
) -> Dict[str, np.ndarray]:
    """Return scalar-grid index coordinates for counter-clockwise azimuths from east."""
    x = np.asarray(xh_m, dtype=np.float64)
    y = np.asarray(yh_m, dtype=np.float64)
    radii = np.asarray(radii_m, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or radii.ndim != 1 or n_azimuth < 4:
        raise ValueError("x/y/radii must be 1-D and n_azimuth must be at least four")
    if np.any(np.diff(x) <= 0.0) or np.any(np.diff(y) <= 0.0) or np.any(radii < 0.0):
        raise ValueError("coordinates must increase and radii must be non-negative")
    azimuth = np.arange(n_azimuth, dtype=np.float64) * (2.0 * np.pi / n_azimuth)
    xx = float(center_x_m) + radii[None, :] * np.cos(azimuth)[:, None]
    yy = float(center_y_m) + radii[None, :] * np.sin(azimuth)[:, None]
    valid = (xx >= x[0]) & (xx <= x[-1]) & (yy >= y[0]) & (yy <= y[-1])
    x_index = np.interp(xx, x, np.arange(x.size, dtype=np.float64))
    y_index = np.interp(yy, y, np.arange(y.size, dtype=np.float64))
    return {
        "azimuth_rad": azimuth,
        "x_m": xx,
        "y_m": yy,
        "x_index": x_index,
        "y_index": y_index,
        "valid": valid,
    }


def sample_scalar_to_polar(field_zyx: np.ndarray, geometry: Dict[str, np.ndarray]) -> np.ndarray:
    """Linearly sample a scalar ``(z,y,x)`` field to ``(z,azimuth,radius)``."""
    field = np.asarray(field_zyx, dtype=np.float64)
    if field.ndim != 3:
        raise ValueError("field must use (z, y, x) order")
    y_index = np.asarray(geometry["y_index"], dtype=np.float64)
    x_index = np.asarray(geometry["x_index"], dtype=np.float64)
    valid = np.asarray(geometry["valid"], dtype=bool)
    if y_index.shape != x_index.shape or y_index.shape != valid.shape:
        raise ValueError("polar geometry arrays have incompatible shapes")
    sampled = np.full((field.shape[0],) + y_index.shape, np.nan, dtype=np.float64)
    coords = np.vstack((y_index.ravel(), x_index.ravel()))
    for level in range(field.shape[0]):
        values = map_coordinates(field[level], coords, order=1, mode="constant", cval=np.nan, prefilter=False)
        sampled[level] = values.reshape(y_index.shape)
    sampled[:, ~valid] = np.nan
    return sampled


def azimuthal_mean(sampled_zar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return equal-angle mean and finite azimuth fraction, both in ``(z,r)`` order."""
    values = np.asarray(sampled_zar, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("sampled values must use (z, azimuth, radius) order")
    coverage = np.isfinite(values).mean(axis=1)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(values, axis=1)
    return mean, coverage


def sample_wind_to_polar(u_zyx: np.ndarray, v_zyx: np.ndarray, geometry: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Sample Cartesian winds and return outward radial and cyclonic tangential wind."""
    u = sample_scalar_to_polar(u_zyx, geometry)
    v = sample_scalar_to_polar(v_zyx, geometry)
    azimuth = np.asarray(geometry["azimuth_rad"], dtype=np.float64)
    cos_azimuth = np.cos(azimuth)[None, :, None]
    sin_azimuth = np.sin(azimuth)[None, :, None]
    radial = u * cos_azimuth + v * sin_azimuth
    tangential = -u * sin_azimuth + v * cos_azimuth
    return radial, tangential
