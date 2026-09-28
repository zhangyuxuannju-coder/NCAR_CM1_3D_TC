import numpy as np

from src.se_nonuniform import assemble_flux_form_matrix, solve_flux_form_dirichlet


def _coords(nr=31, nz=27):
    r = 1_000.0 + 300_000.0 * np.linspace(0.0, 1.0, nr) ** 1.3
    z = 50.0 + 20_000.0 * np.linspace(0.0, 1.0, nz) ** 1.2
    return r, z


def test_zero_rhs_is_zero_with_dirichlet_boundaries():
    r, z = _coords()
    shape = (z.size, r.size)
    out = solve_flux_form_dirichlet(np.ones(shape), np.zeros(shape), np.ones(shape), np.zeros(shape), r, z)
    np.testing.assert_allclose(out.psi, 0.0, atol=1.0e-13)
    assert out.absolute_residual < 1.0e-13


def test_matrix_solution_is_recovered_on_nonuniform_grid():
    r, z = _coords()
    rr, zz = np.meshgrid((r-r[0])/(r[-1]-r[0]), (z-z[0])/(z[-1]-z[0]))
    exact = np.sin(np.pi * rr) * np.sin(np.pi * zz)
    a = 1.0 + 0.2 * zz
    b = 0.05 * np.sin(np.pi * rr) * np.sin(np.pi * zz)
    c = 1.0 + 0.1 * rr
    matrix = assemble_flux_form_matrix(a, b, c, r, z)
    rhs = (matrix @ exact.ravel()).reshape(exact.shape)
    out = solve_flux_form_dirichlet(a, b, c, rhs, r, z)
    np.testing.assert_allclose(out.psi, exact, rtol=1.0e-10, atol=1.0e-10)
    assert out.relative_residual < 1.0e-10


def test_linearity_and_sign_symmetry():
    r, z = _coords()
    rr, zz = np.meshgrid(r, z)
    shape = rr.shape
    a = np.ones(shape)
    b = 0.08 * np.ones(shape)
    c = 1.2 * np.ones(shape)
    rhs_one = np.exp(-((rr-120_000.0)/50_000.0)**2 - ((zz-8_000.0)/3_000.0)**2)
    rhs_two = -0.7 * np.exp(-((rr-210_000.0)/60_000.0)**2 - ((zz-13_000.0)/2_500.0)**2)
    one = solve_flux_form_dirichlet(a, b, c, rhs_one, r, z).psi
    two = solve_flux_form_dirichlet(a, b, c, rhs_two, r, z).psi
    summed = solve_flux_form_dirichlet(a, b, c, rhs_one + rhs_two, r, z).psi
    opposite = solve_flux_form_dirichlet(a, b, c, -rhs_one, r, z).psi
    np.testing.assert_allclose(summed, one + two, rtol=1.0e-11, atol=1.0e-11)
    np.testing.assert_allclose(opposite, -one, rtol=1.0e-11, atol=1.0e-11)
