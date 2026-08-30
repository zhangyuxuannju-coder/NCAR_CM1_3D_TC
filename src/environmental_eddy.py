"""Environmental-eddy momentum and heat forcing diagnostics.

The routines in this module diagnose the azimuthal-eddy contribution to the
tangential momentum equation from storm-centred three-dimensional CM1 fields.
Reynolds averaging is the default because it matches the prime convention in
Bui et al. (2009).  Favre averaging remains available as a mass-weighted
extension when a consistently Favre-averaged basic state is also used.

With outward radial velocity ``u``, tangential velocity ``v`` and vertical
velocity ``w``, the diagnosed acceleration is

    F_lambda,eddy = -1/(rho_bar r^2) d[r^2 <rho u'' v''>]/dr
                    -1/rho_bar d[<rho w'' v''>]/dz,

where double primes denote departures from the selected azimuthal mean.  For
Reynolds averaging the primes are ordinary azimuthal departures but density
remains inside the diagnosed mass flux.  The Favre form is conservative for
compressible flow; Reynolds averaging matches the prime convention used by
Bui et al. (2009).  The corresponding eddy heat forcing is

    Q_eddy = -1/(rho_bar r) d[r <rho u'' theta''>]/dr
             -1/rho_bar d[<rho w'' theta''>]/dz.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

def safe_gradient(field: np.ndarray, coords: np.ndarray, axis: int) -> np.ndarray:
    """NumPy gradient with stable behaviour on short diagnostic grids."""
    coord = np.asarray(coords, dtype=np.float64)
    if coord.size < 2:
        return np.zeros_like(field, dtype=np.float64)
    edge_order = 2 if coord.size >= 3 else 1
    return np.gradient(field, coord, axis=axis, edge_order=edge_order)


def _bin_mean(
    data_3d: np.ndarray,
    bin_index_1d: np.ndarray,
    valid_mask_1d: np.ndarray,
    nr: int,
) -> np.ndarray:
    """Return an unweighted azimuthal mean on ``(z, radius)`` bins."""
    data = np.asarray(data_3d, dtype=np.float64)
    nz = data.shape[0]
    out = np.full((nz, nr), np.nan, dtype=np.float64)
    for k in range(nz):
        flat = data[k].ravel()
        use = valid_mask_1d & np.isfinite(flat)
        if not np.any(use):
            continue
        idx = bin_index_1d[use]
        count = np.bincount(idx, minlength=nr)
        total = np.bincount(idx, weights=flat[use], minlength=nr)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[k] = total / count
    return out


def _expand_mean(
    mean_zr: np.ndarray,
    bin_index_1d: np.ndarray,
    valid_mask_1d: np.ndarray,
    ny: int,
    nx: int,
) -> np.ndarray:
    """Expand a ``(z, radius)`` mean back to the Cartesian scalar grid."""
    nz, nr = mean_zr.shape
    out = np.full((nz, ny, nx), np.nan, dtype=np.float64)
    in_range = valid_mask_1d & (bin_index_1d >= 0) & (bin_index_1d < nr)
    for k in range(nz):
        flat = out[k].ravel()
        flat[in_range] = mean_zr[k, bin_index_1d[in_range]]
    return out


def eddy_flux_divergence(
    rho_bar_zr: np.ndarray,
    radial_mass_flux_zr: np.ndarray,
    vertical_mass_flux_zr: np.ndarray,
    r_m: np.ndarray,
    z_m: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Convert eddy momentum fluxes to tangential acceleration.

    ``radial_mass_flux_zr`` is ``<rho u''v''>`` and
    ``vertical_mass_flux_zr`` is ``<rho w''v''>``.  Returned fields have units
    of m s-2 when the input velocity, density and coordinates are SI.
    """
    rho_bar = np.asarray(rho_bar_zr, dtype=np.float64)
    flux_r = np.asarray(radial_mass_flux_zr, dtype=np.float64)
    flux_z = np.asarray(vertical_mass_flux_zr, dtype=np.float64)
    r = np.asarray(r_m, dtype=np.float64)
    z = np.asarray(z_m, dtype=np.float64)

    if rho_bar.shape != flux_r.shape or rho_bar.shape != flux_z.shape:
        raise ValueError("rho_bar and both eddy flux arrays must share shape (z, r)")
    if rho_bar.shape != (z.size, r.size):
        raise ValueError("eddy flux arrays do not match the supplied z/r coordinates")

    if r.size > 1:
        r_safe = np.maximum(r, 0.5 * float(np.nanmin(np.diff(r))))
    else:
        r_safe = np.maximum(r, 1.0)
    rho_safe = np.maximum(rho_bar, 1.0e-10)

    radial_numerator = flux_r * r_safe[None, :] ** 2
    forcing_radial = -safe_gradient(radial_numerator, r, axis=1) / (
        rho_safe * r_safe[None, :] ** 2
    )
    forcing_vertical = -safe_gradient(flux_z, z, axis=0) / rho_safe
    forcing_total = forcing_radial + forcing_vertical

    return {
        "forcing": forcing_total,
        "forcing_radial": forcing_radial,
        "forcing_vertical": forcing_vertical,
    }


def eddy_scalar_flux_divergence(
    rho_bar_zr: np.ndarray,
    radial_mass_flux_zr: np.ndarray,
    vertical_mass_flux_zr: np.ndarray,
    r_m: np.ndarray,
    z_m: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Return cylindrical convergence of eddy scalar (heat) fluxes.

    The two flux arguments are ``<rho u'' theta''>`` and
    ``<rho w'' theta''>``. Returned tendencies have units K s-1.
    """
    rho_bar = np.asarray(rho_bar_zr, dtype=np.float64)
    flux_r = np.asarray(radial_mass_flux_zr, dtype=np.float64)
    flux_z = np.asarray(vertical_mass_flux_zr, dtype=np.float64)
    r = np.asarray(r_m, dtype=np.float64)
    z = np.asarray(z_m, dtype=np.float64)

    if rho_bar.shape != flux_r.shape or rho_bar.shape != flux_z.shape:
        raise ValueError("rho_bar and both scalar-flux arrays must share shape (z, r)")
    if rho_bar.shape != (z.size, r.size):
        raise ValueError("scalar-flux arrays do not match the supplied z/r coordinates")

    if r.size > 1:
        r_safe = np.maximum(r, 0.5 * float(np.nanmin(np.diff(r))))
    else:
        r_safe = np.maximum(r, 1.0)
    rho_safe = np.maximum(rho_bar, 1.0e-10)

    forcing_radial = -safe_gradient(flux_r * r_safe[None, :], r, axis=1) / (
        rho_safe * r_safe[None, :]
    )
    forcing_vertical = -safe_gradient(flux_z, z, axis=0) / rho_safe
    return {
        "forcing": forcing_radial + forcing_vertical,
        "forcing_radial": forcing_radial,
        "forcing_vertical": forcing_vertical,
    }


def diagnose_eddy_momentum_forcing(
    ur_3d: np.ndarray,
    ut_3d: np.ndarray,
    w_3d: np.ndarray,
    rho_3d: np.ndarray,
    bin_index_1d: np.ndarray,
    valid_mask_1d: np.ndarray,
    nr: int,
    r_m: np.ndarray,
    z_m: np.ndarray,
    averaging: str = "reynolds",
    theta_3d: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """Diagnose eddy momentum and optional heat forcing on ``(z, r)``.

    Parameters
    ----------
    averaging:
        ``"reynolds"`` reproduces the prime convention in Bui et al. (2009).
        ``"favre"`` is a conservative compressible extension, but should be
        paired with a consistently Favre-averaged basic state.
    """
    ur = np.asarray(ur_3d, dtype=np.float64)
    ut = np.asarray(ut_3d, dtype=np.float64)
    w = np.asarray(w_3d, dtype=np.float64)
    rho = np.asarray(rho_3d, dtype=np.float64)
    if not (ur.shape == ut.shape == w.shape == rho.shape):
        raise ValueError("ur, ut, w and rho must share shape (z, y, x)")
    theta = None if theta_3d is None else np.asarray(theta_3d, dtype=np.float64)
    if theta is not None and theta.shape != ur.shape:
        raise ValueError("theta_3d must share shape (z, y, x) with the velocity fields")

    averaging_key = averaging.strip().lower()
    if averaging_key not in {"favre", "reynolds"}:
        raise ValueError("averaging must be 'favre' or 'reynolds'")

    nz, ny, nx = ur.shape
    if nz != len(z_m):
        raise ValueError("vertical coordinate length does not match input fields")

    rho_bar = _bin_mean(rho, bin_index_1d, valid_mask_1d, nr)
    rho_safe = np.maximum(rho_bar, 1.0e-10)

    if averaging_key == "favre":
        ur_mean = _bin_mean(rho * ur, bin_index_1d, valid_mask_1d, nr) / rho_safe
        ut_mean = _bin_mean(rho * ut, bin_index_1d, valid_mask_1d, nr) / rho_safe
        w_mean = _bin_mean(rho * w, bin_index_1d, valid_mask_1d, nr) / rho_safe
        theta_mean = (
            _bin_mean(rho * theta, bin_index_1d, valid_mask_1d, nr) / rho_safe
            if theta is not None
            else None
        )
    else:
        ur_mean = _bin_mean(ur, bin_index_1d, valid_mask_1d, nr)
        ut_mean = _bin_mean(ut, bin_index_1d, valid_mask_1d, nr)
        w_mean = _bin_mean(w, bin_index_1d, valid_mask_1d, nr)
        theta_mean = (
            _bin_mean(theta, bin_index_1d, valid_mask_1d, nr)
            if theta is not None
            else None
        )

    ur_prime = ur - _expand_mean(ur_mean, bin_index_1d, valid_mask_1d, ny, nx)
    ut_prime = ut - _expand_mean(ut_mean, bin_index_1d, valid_mask_1d, ny, nx)
    w_prime = w - _expand_mean(w_mean, bin_index_1d, valid_mask_1d, ny, nx)
    theta_prime = (
        theta - _expand_mean(theta_mean, bin_index_1d, valid_mask_1d, ny, nx)
        if theta is not None and theta_mean is not None
        else None
    )

    if averaging_key == "favre":
        flux_r_mass = _bin_mean(
            rho * ur_prime * ut_prime, bin_index_1d, valid_mask_1d, nr
        )
        flux_z_mass = _bin_mean(
            rho * w_prime * ut_prime, bin_index_1d, valid_mask_1d, nr
        )
        eddy_kinetic_energy = 0.5 * _bin_mean(
            rho * (ur_prime**2 + ut_prime**2 + w_prime**2),
            bin_index_1d,
            valid_mask_1d,
            nr,
        ) / rho_safe
    else:
        # Keep density inside the Reynolds covariance.  The previous
        # rho_bar*<u'v'> approximation contradicted the documented
        # <rho u'v'> flux and discarded azimuthal density--eddy correlations,
        # which matter for a thermally balanced environmental jet.
        flux_r_mass = _bin_mean(
            rho * ur_prime * ut_prime, bin_index_1d, valid_mask_1d, nr
        )
        flux_z_mass = _bin_mean(
            rho * w_prime * ut_prime, bin_index_1d, valid_mask_1d, nr
        )
        eddy_kinetic_energy = 0.5 * _bin_mean(
            ur_prime**2 + ut_prime**2 + w_prime**2,
            bin_index_1d,
            valid_mask_1d,
            nr,
        )

    divergence = eddy_flux_divergence(
        rho_bar, flux_r_mass, flux_z_mass, r_m, z_m
    )
    result = {
        "averaging": np.array([averaging_key]),
        "rho_bar": rho_bar,
        "ur_mean": ur_mean,
        "ut_mean": ut_mean,
        "w_mean": w_mean,
        "radial_mass_flux": flux_r_mass,
        "vertical_mass_flux": flux_z_mass,
        "eddy_kinetic_energy": eddy_kinetic_energy,
        "F_lambda_eddy": divergence["forcing"],
        "F_lambda_eddy_radial": divergence["forcing_radial"],
        "F_lambda_eddy_vertical": divergence["forcing_vertical"],
    }
    if theta_prime is not None:
        if averaging_key == "favre":
            heat_flux_r = _bin_mean(
                rho * ur_prime * theta_prime, bin_index_1d, valid_mask_1d, nr
            )
            heat_flux_z = _bin_mean(
                rho * w_prime * theta_prime, bin_index_1d, valid_mask_1d, nr
            )
        else:
            heat_flux_r = _bin_mean(
                rho * ur_prime * theta_prime, bin_index_1d, valid_mask_1d, nr
            )
            heat_flux_z = _bin_mean(
                rho * w_prime * theta_prime, bin_index_1d, valid_mask_1d, nr
            )
        heat = eddy_scalar_flux_divergence(
            rho_bar, heat_flux_r, heat_flux_z, r_m, z_m
        )
        result.update(
            {
                "theta_mean": theta_mean,
                "eddy_radial_heat_mass_flux": heat_flux_r,
                "eddy_vertical_heat_mass_flux": heat_flux_z,
                "Q_eddy": heat["forcing"],
                "Q_eddy_radial": heat["forcing_radial"],
                "Q_eddy_vertical": heat["forcing_vertical"],
            }
        )
    return result


def environmental_difference(
    jet_forcing_zr: np.ndarray,
    ctrl_forcing_zr: np.ndarray,
) -> np.ndarray:
    """Return the jet-induced environmental eddy forcing ``JET - CTRL``."""
    jet = np.asarray(jet_forcing_zr, dtype=np.float64)
    ctrl = np.asarray(ctrl_forcing_zr, dtype=np.float64)
    if jet.shape != ctrl.shape:
        raise ValueError("JET and CTRL forcing arrays must have the same shape")
    return jet - ctrl

