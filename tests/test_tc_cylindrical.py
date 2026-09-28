import numpy as np

from src.tc_cylindrical import azimuthal_mean, regular_polar_geometry, sample_scalar_to_polar, sample_wind_to_polar


def test_regular_azimuth_mean_of_linear_scalar_field():
    x = np.linspace(-100_000.0, 100_000.0, 101)
    y = np.linspace(-100_000.0, 100_000.0, 101)
    xx, yy = np.meshgrid(x, y)
    field = np.stack((xx + 2.0 * yy, 3.0 * xx - yy))
    geometry = regular_polar_geometry(x, y, 10_000.0, -5_000.0, np.array([5_000.0, 20_000.0]), 360)
    mean, coverage = azimuthal_mean(sample_scalar_to_polar(field, geometry))
    np.testing.assert_allclose(mean[0], 0.0, atol=1.0e-8)
    np.testing.assert_allclose(mean[1], 35_000.0, atol=1.0e-8)
    np.testing.assert_allclose(coverage, 1.0)


def test_solid_body_rotation_has_cyclonic_tangential_wind():
    x = np.linspace(-100_000.0, 100_000.0, 101)
    y = np.linspace(-100_000.0, 100_000.0, 101)
    xx, yy = np.meshgrid(x, y)
    omega = 2.0e-4
    u = np.stack((-omega * yy, -omega * yy))
    v = np.stack((omega * xx, omega * xx))
    radii = np.array([10_000.0, 30_000.0])
    geometry = regular_polar_geometry(x, y, 0.0, 0.0, radii, 360)
    ur, vt = sample_wind_to_polar(u, v, geometry)
    ur_mean, _ = azimuthal_mean(ur)
    vt_mean, _ = azimuthal_mean(vt)
    np.testing.assert_allclose(ur_mean, 0.0, atol=1.0e-10)
    np.testing.assert_allclose(vt_mean, np.broadcast_to(omega * radii[None, :], vt_mean.shape), atol=1.0e-10)
