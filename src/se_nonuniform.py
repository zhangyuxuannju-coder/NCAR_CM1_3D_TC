"""Non-uniform-grid flux-form solver for the axisymmetric Bui SE operator.

The operator is

    d_r(a psi_r + b psi_z) + d_z(b psi_r + c psi_z) = rhs,

where ``a=K1/(rho*r)``, ``b=K2/(rho*r)``, and ``c=K3/(rho*r)``.
Coordinates are physical metres and arrays use ``(z, r)`` order.  The solver
uses homogeneous Dirichlet conditions on all four finite-domain boundaries;
it is therefore a balanced diagnostic on that stated finite domain, not a
claim about the unbounded atmosphere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class NonuniformSEResult:
    """A solution plus the residual evaluated against its assembled matrix."""

    psi: np.ndarray
    relative_residual: float
    absolute_residual: float
    rhs_l2: float


def _validate_coordinates(name: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 3 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite one-dimensional array with at least three points")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return values


def _derivative_weights(coords: np.ndarray, index: int) -> Tuple[Tuple[int, float], ...]:
    """Second-order three-point first-derivative weights at an interior node."""
    left = coords[index] - coords[index - 1]
    right = coords[index + 1] - coords[index]
    return (
        (index - 1, -right / (left * (left + right))),
        (index, (right - left) / (left * right)),
        (index + 1, left / (right * (left + right))),
    )


def assemble_flux_form_matrix(
    a_zr: np.ndarray,
    b_zr: np.ndarray,
    c_zr: np.ndarray,
    r_m: np.ndarray,
    z_m: np.ndarray,
) -> csr_matrix:
    """Assemble a conservative centred finite-difference matrix.

    Cross derivatives are evaluated by averaging nodal centred derivatives at
    the corresponding face.  Boundary rows impose ``psi=0``.  Coefficients
    are arithmetic averaged to faces, appropriate to this diagnostic's smooth
    axisymmetric basic states; the caller retains raw and regularized fields
    separately before selecting coefficients here.
    """
    r = _validate_coordinates("r_m", r_m)
    z = _validate_coordinates("z_m", z_m)
    a = np.asarray(a_zr, dtype=np.float64)
    b = np.asarray(b_zr, dtype=np.float64)
    c = np.asarray(c_zr, dtype=np.float64)
    expected = (z.size, r.size)
    if not (a.shape == b.shape == c.shape == expected):
        raise ValueError(f"coefficient arrays must have shape {expected} in (z, r) order")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(np.isfinite(c))):
        raise ValueError("SE coefficients must be finite")

    nz, nr = expected
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    def flat(j: int, i: int) -> int:
        return j * nr + i

    def add(terms: Dict[Tuple[int, int], float], j: int, i: int, value: float) -> None:
        terms[(j, i)] = terms.get((j, i), 0.0) + value

    def add_dz(terms: Dict[Tuple[int, int], float], j: int, i: int, factor: float) -> None:
        for jj, weight in _derivative_weights(z, j):
            add(terms, jj, i, factor * weight)

    def add_dr(terms: Dict[Tuple[int, int], float], j: int, i: int, factor: float) -> None:
        for ii, weight in _derivative_weights(r, i):
            add(terms, j, ii, factor * weight)

    for j in range(nz):
        for i in range(nr):
            row = flat(j, i)
            if j in (0, nz - 1) or i in (0, nr - 1):
                rows.append(row)
                cols.append(row)
                vals.append(1.0)
                continue

            terms: Dict[Tuple[int, int], float] = {}
            dr_plus = r[i + 1] - r[i]
            dr_minus = r[i] - r[i - 1]
            dz_plus = z[j + 1] - z[j]
            dz_minus = z[j] - z[j - 1]
            dr_cell = 0.5 * (dr_plus + dr_minus)
            dz_cell = 0.5 * (dz_plus + dz_minus)

            # (F_r[i+1/2]-F_r[i-1/2]) / dr_cell
            a_plus = 0.5 * (a[j, i] + a[j, i + 1]) / dr_plus / dr_cell
            b_plus = 0.5 * (b[j, i] + b[j, i + 1]) / (2.0 * dr_cell)
            add(terms, j, i + 1, a_plus)
            add(terms, j, i, -a_plus)
            add_dz(terms, j, i, b_plus)
            add_dz(terms, j, i + 1, b_plus)

            a_minus = 0.5 * (a[j, i] + a[j, i - 1]) / dr_minus / dr_cell
            b_minus = 0.5 * (b[j, i] + b[j, i - 1]) / (2.0 * dr_cell)
            add(terms, j, i, -a_minus)
            add(terms, j, i - 1, a_minus)
            add_dz(terms, j, i, -b_minus)
            add_dz(terms, j, i - 1, -b_minus)

            # (F_z[j+1/2]-F_z[j-1/2]) / dz_cell
            c_plus = 0.5 * (c[j, i] + c[j + 1, i]) / dz_plus / dz_cell
            b_plus_z = 0.5 * (b[j, i] + b[j + 1, i]) / (2.0 * dz_cell)
            add(terms, j + 1, i, c_plus)
            add(terms, j, i, -c_plus)
            add_dr(terms, j, i, b_plus_z)
            add_dr(terms, j + 1, i, b_plus_z)

            c_minus = 0.5 * (c[j, i] + c[j - 1, i]) / dz_minus / dz_cell
            b_minus_z = 0.5 * (b[j, i] + b[j - 1, i]) / (2.0 * dz_cell)
            add(terms, j, i, -c_minus)
            add(terms, j - 1, i, c_minus)
            add_dr(terms, j, i, -b_minus_z)
            add_dr(terms, j - 1, i, -b_minus_z)

            for (jj, ii), value in terms.items():
                if value:
                    rows.append(row)
                    cols.append(flat(jj, ii))
                    vals.append(value)

    return csr_matrix((vals, (rows, cols)), shape=(nr * nz, nr * nz))


def solve_flux_form_dirichlet(
    a_zr: np.ndarray,
    b_zr: np.ndarray,
    c_zr: np.ndarray,
    rhs_zr: np.ndarray,
    r_m: np.ndarray,
    z_m: np.ndarray,
) -> NonuniformSEResult:
    """Solve the stated operator and return numerical residual diagnostics."""
    rhs = np.asarray(rhs_zr, dtype=np.float64)
    if rhs.shape != np.asarray(a_zr).shape or not np.all(np.isfinite(rhs)):
        raise ValueError("rhs_zr must be finite and match the coefficient shape")
    matrix = assemble_flux_form_matrix(a_zr, b_zr, c_zr, r_m, z_m)
    rhs_flat = rhs.ravel().copy()
    nz, nr = rhs.shape
    for j in range(nz):
        for i in range(nr):
            if j in (0, nz - 1) or i in (0, nr - 1):
                rhs_flat[j * nr + i] = 0.0
    solution_flat = np.asarray(spsolve(matrix, rhs_flat), dtype=np.float64)
    residual = np.asarray(matrix @ solution_flat - rhs_flat, dtype=np.float64)
    absolute = float(np.linalg.norm(residual))
    rhs_l2 = float(np.linalg.norm(rhs_flat))
    relative = absolute / max(rhs_l2, 1.0e-30)
    return NonuniformSEResult(
        psi=solution_flat.reshape(rhs.shape),
        relative_residual=relative,
        absolute_residual=absolute,
        rhs_l2=rhs_l2,
    )
