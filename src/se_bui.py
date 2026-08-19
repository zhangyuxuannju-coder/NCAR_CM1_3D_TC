"""Physically explicit Bui et al. (2009) general Sawyer--Eliassen core.

This module keeps the notation of the general height-coordinate equation:

    chi = 1/theta
    Cg  = v^2/r + f v
    xi  = f + 2 v/r

The balanced thermal field satisfies ``g chi_r + (chi Cg)_z = 0``.  The SE
right-hand side is

    g (chi^2 Q)_r + (Cg chi^2 Q)_z - (chi xi F_lambda)_z.

Unlike the legacy NCL-compatible path, this implementation does not apply an
upper-level exponential sponge or an implicit 0.4 baroclinic scaling.  A scale
may still be supplied explicitly for controlled sensitivity experiments.
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

G = 9.806


def invert_balanced_theta(
    vt_zr: np.ndarray,
    theta_model_zr: np.ndarray,
    r_m: np.ndarray,
    z_m: np.ndarray,
    f: float,
    theta_floor: float = 150.0,
    outer_smooth_window: int = 1,
    corrector_steps: int = 2,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Integrate the general thermal-wind equation inward from the outer edge.

    The method-of-lines integration advances ``chi=1/theta`` in radius using a
    predictor/corrector discretization of ``chi_r = -(chi Cg)_z/g``.  It is a
    direct height-coordinate implementation of the balance relation rather
    than the legacy approximation ``theta_r ~ -theta Cg_z/g``.
    """
    vt = np.asarray(vt_zr, dtype=np.float64)
    theta_model = np.asarray(theta_model_zr, dtype=np.float64)
    r = np.asarray(r_m, dtype=np.float64)
    z = np.asarray(z_m, dtype=np.float64)
    if vt.shape != theta_model.shape or vt.shape != (z.size, r.size):
        raise ValueError("vt/theta fields must share shape (z, r)")

    if r.size > 1:
        r_safe = np.maximum(r, 0.5 * float(np.nanmin(np.diff(r))))
    else:
        r_safe = np.maximum(r, 1.0)
    cg = vt**2 / r_safe[None, :] + f * vt

    theta_outer = np.asarray(theta_model[:, -1], dtype=np.float64)
    window = max(1, int(outer_smooth_window))
    if window % 2 == 0:
        window += 1
    if window > 1:
        pad = window // 2
        padded = np.pad(theta_outer, (pad, pad), mode="edge")
        theta_outer = np.convolve(
            padded, np.ones(window, dtype=np.float64) / window, mode="valid"
        )
    fallback = np.nanmedian(theta_model, axis=1)
    theta_outer = np.where(np.isfinite(theta_outer), theta_outer, fallback)
    theta_outer = np.maximum(theta_outer, theta_floor)

    chi = np.full_like(vt, np.nan, dtype=np.float64)
    chi[:, -1] = 1.0 / theta_outer
    n_correct = max(1, int(corrector_steps))

    for j in range(r.size - 2, -1, -1):
        dr = float(r[j + 1] - r[j])
        rhs_outer = -safe_gradient(chi[:, j + 1] * cg[:, j + 1], z, axis=0) / G
        chi_j = chi[:, j + 1] - rhs_outer * dr
        for _ in range(n_correct):
            rhs_inner = -safe_gradient(chi_j * cg[:, j], z, axis=0) / G
            chi_j = chi[:, j + 1] - 0.5 * (rhs_outer + rhs_inner) * dr
        chi[:, j] = chi_j

    # Prevent non-physical values while retaining a diagnostic count.
    bad = (~np.isfinite(chi)) | (chi <= 0.0)
    chi_fallback = 1.0 / np.maximum(theta_model, theta_floor)
    chi[bad] = chi_fallback[bad]
    theta_bal = np.maximum(1.0 / np.maximum(chi, 1.0e-12), theta_floor)
    chi = 1.0 / theta_bal

    residual = G * safe_gradient(chi, r, axis=1) + safe_gradient(chi * cg, z, axis=0)
    scale = np.maximum(
        np.abs(G * safe_gradient(chi, r, axis=1))
        + np.abs(safe_gradient(chi * cg, z, axis=0)),
        1.0e-14,
    )
    return theta_bal, {
        "repaired_chi_points": float(np.count_nonzero(bad)),
        "thermal_wind_residual_rms": float(np.sqrt(np.nanmean(residual**2))),
        "thermal_wind_relative_residual_rms": float(
            np.sqrt(np.nanmean((residual / scale) ** 2))
        ),
    }


def build_basic_state(
    vt_zr: np.ndarray,
    theta_bal_zr: np.ndarray,
    rho_zr: np.ndarray,
    r_m: np.ndarray,
    z_m: np.ndarray,
    f: float,
    baroclinic_scale: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Build the three principal coefficients of the Bui general SE operator."""
    vt = np.asarray(vt_zr, dtype=np.float64)
    theta = np.asarray(theta_bal_zr, dtype=np.float64)
    rho = np.asarray(rho_zr, dtype=np.float64)
    r = np.asarray(r_m, dtype=np.float64)
    z = np.asarray(z_m, dtype=np.float64)
    if not (vt.shape == theta.shape == rho.shape == (z.size, r.size)):
        raise ValueError("basic-state fields must share shape (z, r)")

    if r.size > 1:
        r_safe = np.maximum(r, 0.5 * float(np.nanmin(np.diff(r))))
    else:
        r_safe = np.maximum(r, 1.0)
    chi = 1.0 / np.maximum(theta, 1.0)
    cg = vt**2 / r_safe[None, :] + f * vt
    xi = f + 2.0 * vt / r_safe[None, :]
    zeta_rel = safe_gradient(r_safe[None, :] * vt, r, axis=1) / r_safe[None, :]
    zeta_abs = zeta_rel + f
    chi_r = safe_gradient(chi, r, axis=1)

    k1 = -G * safe_gradient(chi, z, axis=0)
    k2 = -safe_gradient(chi * cg, z, axis=0) * float(baroclinic_scale)
    k3 = chi * xi * zeta_abs + cg * chi_r

    return {
        "chi": chi,
        "Cg": cg,
        "xi": xi,
        "zeta_rel": zeta_rel,
        "zeta_abs": zeta_abs,
        "rho": rho,
        "K1_raw": k1,
        "K2_raw": k2,
        "K3_raw": k3,
        "thermal_wind_residual": G * chi_r + safe_gradient(chi * cg, z, axis=0),
    }


def regularize_ellipticity(
    k1_zr: np.ndarray,
    k2_zr: np.ndarray,
    k3_zr: np.ndarray,
    eps_ratio: float = 1.0e-3,
    margin: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Apply the smallest local coefficient adjustment needed for ellipticity.

    Static and generalized inertial stability receive small positive floors;
    then only the magnitude of K2 is reduced where ``K1*K3-K2^2`` is below
    the requested relative/absolute margin.  No height-dependent sponge is
    applied.
    """
    if not 0.0 <= eps_ratio < 1.0:
        raise ValueError("eps_ratio must satisfy 0 <= eps_ratio < 1")
    k1 = np.array(k1_zr, copy=True, dtype=np.float64)
    k2 = np.array(k2_zr, copy=True, dtype=np.float64)
    k3 = np.array(k3_zr, copy=True, dtype=np.float64)
    if not (k1.shape == k2.shape == k3.shape):
        raise ValueError("K1, K2 and K3 must have matching shapes")

    k1_scale = max(float(np.nanmax(np.abs(k1))), 1.0e-14)
    k3_scale = max(float(np.nanmax(np.abs(k3))), 1.0e-14)
    k1_floor = max(k1_scale * eps_ratio, 1.0e-14)
    k3_floor = max(k3_scale * eps_ratio, 1.0e-14)
    bad_k1 = (~np.isfinite(k1)) | (k1 < k1_floor)
    bad_k3 = (~np.isfinite(k3)) | (k3 < k3_floor)
    k1[bad_k1] = k1_floor
    k3[bad_k3] = k3_floor
    k2[~np.isfinite(k2)] = 0.0

    product = k1 * k3
    target = np.maximum(float(margin), eps_ratio * product)
    allowed_sq = np.maximum(product - target, 0.0)
    bad_disc = k2**2 >= allowed_sq
    allowed = np.sqrt(allowed_sq) * (1.0 - 1.0e-10)
    k2[bad_disc] = np.sign(k2[bad_disc]) * allowed[bad_disc]

    disc_before = np.asarray(k1_zr) * np.asarray(k3_zr) - np.asarray(k2_zr) ** 2
    disc_after = k1 * k3 - k2**2
    return k1, k2, k3, {
        "bad_K1_points": float(np.count_nonzero(bad_k1)),
        "bad_K3_points": float(np.count_nonzero(bad_k3)),
        "baroclinic_adjusted_points": float(np.count_nonzero(bad_disc)),
        "min_discriminant_before": float(np.nanmin(disc_before)),
        "min_discriminant_after": float(np.nanmin(disc_after)),
        "eps_ratio": float(eps_ratio),
        "margin": float(margin),
        "upper_sponge_applied": 0.0,
    }


def assemble_operator(
    basic: Dict[str, np.ndarray],
    k1_zr: np.ndarray,
    k2_zr: np.ndarray,
    k3_zr: np.ndarray,
    r_m: np.ndarray,
    z_m: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Expand the flux-form operator into A/B/C/D/E coefficient arrays."""
    rho = np.maximum(np.asarray(basic["rho"], dtype=np.float64), 1.0e-10)
    r = np.asarray(r_m, dtype=np.float64)
    z = np.asarray(z_m, dtype=np.float64)
    if r.size > 1:
        r_safe = np.maximum(r, 0.5 * float(np.nanmin(np.diff(r))))
    else:
        r_safe = np.maximum(r, 1.0)
    mass_metric = 1.0 / (rho * r_safe[None, :])

    a = np.asarray(k1_zr) * mass_metric
    cross = np.asarray(k2_zr) * mass_metric
    c = np.asarray(k3_zr) * mass_metric
    b = 2.0 * cross
    d = safe_gradient(a, r, axis=1) + safe_gradient(cross, z, axis=0)
    e = safe_gradient(cross, r, axis=1) + safe_gradient(c, z, axis=0)
    return {
        "A": a,
        "B": b,
        "C": c,
        "D": d,
        "E": e,
        "discriminant": 4.0 * a * c - b**2,
    }


def build_forcing(
    basic: Dict[str, np.ndarray],
    q_zr: np.ndarray,
    f_lambda_zr: np.ndarray,
    r_m: np.ndarray,
    z_m: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Build the thermal and tangential-momentum RHS of Bui Eq. (14)."""
    chi = np.asarray(basic["chi"], dtype=np.float64)
    cg = np.asarray(basic["Cg"], dtype=np.float64)
    xi = np.asarray(basic["xi"], dtype=np.float64)
    q = np.asarray(q_zr, dtype=np.float64)
    f_lambda = np.asarray(f_lambda_zr, dtype=np.float64)
    if not (chi.shape == q.shape == f_lambda.shape):
        raise ValueError("Q and F_lambda must match the basic-state shape")

    thermal_flux = chi**2 * q
    forcing_thermal = G * safe_gradient(thermal_flux, r_m, axis=1) + safe_gradient(
        cg * thermal_flux, z_m, axis=0
    )
    forcing_momentum = -safe_gradient(chi * xi * f_lambda, z_m, axis=0)
    return {
        "forcing_thermal": forcing_thermal,
        "forcing_momentum": forcing_momentum,
        "forcing_total": forcing_thermal + forcing_momentum,
    }

