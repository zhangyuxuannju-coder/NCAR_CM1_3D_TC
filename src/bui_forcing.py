"""Assemble Bui et al. (2009) heat and tangential-momentum sources.

CM1 outputs signed right-hand-side tendencies.  Advective tendencies are used
only for an independent eddy-budget check; they must not be added again when
the eddy flux convergence is diagnosed directly from the three-dimensional
fields.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np


THERMAL_DIFFUSION_TERMS = (
    "hidiff",
    "vidiff",
    "hediff",
    "vediff",
    "hturb",
    "vturb",
)
THERMAL_DIABATIC_TERMS = ("mp", "rad", "diss", "pbl")
THERMAL_OTHER_MODEL_TERMS = ("div", "rdamp", "nudge", "lsw", "frc", "efall")

MOMENTUM_DIFFUSION_TERMS = (
    "hidiff",
    "vidiff",
    "hediff",
    "vediff",
    "hturb",
    "vturb",
    "pbl",
)
MOMENTUM_OTHER_MODEL_TERMS = ("rdamp", "lsw", "frc")


def _sum_components(
    components: Mapping[str, np.ndarray], names: tuple[str, ...], template: np.ndarray
) -> np.ndarray:
    out = np.zeros_like(np.asarray(template, dtype=np.float64))
    for name in names:
        if name in components:
            out = out + np.asarray(components[name], dtype=np.float64)
    return out


def assemble_bui_forcings(
    q_eddy_zr: np.ndarray,
    f_eddy_zr: np.ndarray,
    thermal_budget_zr: Mapping[str, np.ndarray],
    tangential_budget_zr: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Return transparent components and totals for Bui ``Q``/``F_lambda``.

    All CM1 budget arrays must already be destaggered, rotated where needed,
    and azimuthally averaged.  ``hadv`` and ``vadv`` are intentionally excluded
    from the totals because their eddy parts are represented by ``q_eddy_zr``
    and ``f_eddy_zr``.
    """
    q_eddy = np.asarray(q_eddy_zr, dtype=np.float64)
    f_eddy = np.asarray(f_eddy_zr, dtype=np.float64)
    if q_eddy.shape != f_eddy.shape:
        raise ValueError("Q_eddy and F_lambda_eddy must have the same (z, r) shape")

    q_diff = _sum_components(thermal_budget_zr, THERMAL_DIFFUSION_TERMS, q_eddy)
    q_diab = _sum_components(thermal_budget_zr, THERMAL_DIABATIC_TERMS, q_eddy)
    q_other = _sum_components(thermal_budget_zr, THERMAL_OTHER_MODEL_TERMS, q_eddy)
    f_diff = _sum_components(tangential_budget_zr, MOMENTUM_DIFFUSION_TERMS, f_eddy)
    f_other = _sum_components(tangential_budget_zr, MOMENTUM_OTHER_MODEL_TERMS, f_eddy)

    return {
        "Q_eddy": q_eddy,
        "Q_diffusion": q_diff,
        "Q_diabatic": q_diab,
        "Q_other_model": q_other,
        "Q_total": q_eddy + q_diff + q_diab + q_other,
        "F_lambda_eddy": f_eddy,
        "F_lambda_diffusion": f_diff,
        "F_lambda_other_model": f_other,
        "F_lambda_total": f_eddy + f_diff + f_other,
    }


def reconstruct_eddy_from_advection(
    tangential_hadv_zr: np.ndarray,
    tangential_vadv_zr: np.ndarray,
    ur_bar_zr: np.ndarray,
    ut_bar_zr: np.ndarray,
    w_bar_zr: np.ndarray,
    r_m: np.ndarray,
    z_m: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Independent Reynolds-eddy check using signed CM1 advection tendencies.

    For CM1's RHS tendencies,

    ``Fh_eddy = T_hadv + ubar * (d(vbar)/dr + vbar/r)`` and
    ``Fv_eddy = T_vadv + wbar * d(vbar)/dz``.

    This form contains no additional ``<u'v'>/r`` correction; that geometry is
    already part of the cylindrical eddy-vorticity term.
    """
    thadv = np.asarray(tangential_hadv_zr, dtype=np.float64)
    tvadv = np.asarray(tangential_vadv_zr, dtype=np.float64)
    ur = np.asarray(ur_bar_zr, dtype=np.float64)
    ut = np.asarray(ut_bar_zr, dtype=np.float64)
    w = np.asarray(w_bar_zr, dtype=np.float64)
    r = np.asarray(r_m, dtype=np.float64)
    z = np.asarray(z_m, dtype=np.float64)
    if not (thadv.shape == tvadv.shape == ur.shape == ut.shape == w.shape):
        raise ValueError("all advection-reconstruction fields must share shape (z, r)")

    edge_r = 2 if r.size >= 3 else 1
    edge_z = 2 if z.size >= 3 else 1
    dv_dr = np.gradient(ut, r, axis=1, edge_order=edge_r) if r.size >= 2 else np.zeros_like(ut)
    dv_dz = np.gradient(ut, z, axis=0, edge_order=edge_z) if z.size >= 2 else np.zeros_like(ut)
    r_safe = np.maximum(r, 0.5 * np.min(np.diff(r)) if r.size > 1 else 1.0)
    horizontal = thadv + ur * (dv_dr + ut / r_safe[None, :])
    vertical = tvadv + w * dv_dz
    return {
        "F_lambda_eddy_budget_horizontal": horizontal,
        "F_lambda_eddy_budget_vertical": vertical,
        "F_lambda_eddy_budget": horizontal + vertical,
    }
