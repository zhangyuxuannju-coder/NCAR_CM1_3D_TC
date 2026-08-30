import unittest

import numpy as np

from src.environmental_eddy import (
    diagnose_eddy_momentum_forcing,
    eddy_flux_divergence,
    eddy_scalar_flux_divergence,
    environmental_difference,
)


class EnvironmentalEddyTests(unittest.TestCase):
    def test_flux_divergence_respects_cylindrical_r2_metric(self):
        r = np.array([1.0, 2.0, 4.0, 8.0])
        z = np.array([0.0, 1.0, 2.0])
        rho = np.ones((z.size, r.size))
        radial_mass_flux = np.broadcast_to(3.0 / r[None, :] ** 2, rho.shape)
        vertical_mass_flux = np.full_like(rho, 5.0)
        result = eddy_flux_divergence(
            rho, radial_mass_flux, vertical_mass_flux, r, z
        )
        np.testing.assert_allclose(result["forcing"], 0.0, atol=1.0e-12)

    def test_axisymmetric_fields_have_zero_eddy_forcing(self):
        z = np.array([0.0, 1000.0, 2000.0])
        r = np.array([1000.0, 3000.0])
        bins = np.array([0, 0, 1, 1])
        valid = np.ones(4, dtype=bool)
        shape = (z.size, 1, 4)
        ur = np.zeros(shape)
        ut = np.broadcast_to(np.array([10.0, 10.0, 20.0, 20.0]), shape).copy()
        w = np.zeros(shape)
        rho = np.ones(shape)
        theta = np.broadcast_to(np.array([300.0, 300.0, 305.0, 305.0]), shape).copy()
        result = diagnose_eddy_momentum_forcing(
            ur, ut, w, rho, bins, valid, 2, r, z,
            averaging="reynolds", theta_3d=theta,
        )
        np.testing.assert_allclose(result["radial_mass_flux"], 0.0, atol=1.0e-12)
        np.testing.assert_allclose(result["vertical_mass_flux"], 0.0, atol=1.0e-12)
        np.testing.assert_allclose(result["F_lambda_eddy"], 0.0, atol=1.0e-12)
        np.testing.assert_allclose(result["Q_eddy"], 0.0, atol=1.0e-12)
        np.testing.assert_allclose(result["eddy_kinetic_energy"], 0.0, atol=1.0e-12)

    def test_scalar_flux_divergence_respects_cylindrical_r_metric(self):
        r = np.array([1.0, 2.0, 4.0, 8.0])
        z = np.array([0.0, 1.0, 2.0])
        rho = np.ones((z.size, r.size))
        radial_heat_flux = np.broadcast_to(3.0 / r[None, :], rho.shape)
        vertical_heat_flux = np.full_like(rho, 5.0)
        result = eddy_scalar_flux_divergence(
            rho, radial_heat_flux, vertical_heat_flux, r, z
        )
        np.testing.assert_allclose(result["forcing"], 0.0, atol=1.0e-12)

    def test_environmental_definition_is_jet_minus_ctrl(self):
        jet = np.array([[3.0, 5.0]])
        ctrl = np.array([[1.0, 2.0]])
        np.testing.assert_array_equal(
            environmental_difference(jet, ctrl), np.array([[2.0, 3.0]])
        )

    def test_reynolds_flux_retains_density_inside_covariance(self):
        z = np.array([0.0, 1000.0, 2000.0])
        r = np.array([1000.0])
        bins = np.zeros(4, dtype=int)
        valid = np.ones(4, dtype=bool)
        ur_1d = np.array([1.0, -1.0, 1.0, -1.0])
        ut_1d = np.array([1.0, 1.0, -1.0, -1.0])
        rho_1d = np.array([2.0, 1.0, 1.0, 2.0])
        ur = np.broadcast_to(ur_1d, (z.size, 1, ur_1d.size)).copy()
        ut = np.broadcast_to(ut_1d, ur.shape).copy()
        rho = np.broadcast_to(rho_1d, ur.shape).copy()
        result = diagnose_eddy_momentum_forcing(
            ur, ut, np.zeros_like(ur), rho, bins, valid, 1, r, z,
            averaging="reynolds",
        )
        np.testing.assert_allclose(result["radial_mass_flux"], 0.5, atol=1.0e-12)

    def test_pure_two_dimensional_zonal_jet_has_zero_ring_mean_torque(self):
        nphi = 720
        phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
        r = np.array([800.0e3, 888.0e3, 1000.0e3])
        z = np.array([10.0e3, 12.0e3, 14.0e3])
        bins = np.repeat(np.arange(r.size), nphi)
        phi_all = np.tile(phi, r.size)
        r_all = np.repeat(r, nphi)
        y_all = r_all * np.sin(phi_all)
        jet_horizontal = 45.0 * np.exp(-((y_all - 888.0e3) / 444.0e3) ** 2)
        jet_vertical = np.exp(-((z - 12.0e3) / 2.0e3) ** 2)
        cart_u = jet_vertical[:, None] * jet_horizontal[None, :]
        ur = (cart_u * np.cos(phi_all)[None, :])[:, None, :]
        ut = (-cart_u * np.sin(phi_all)[None, :])[:, None, :]
        result = diagnose_eddy_momentum_forcing(
            ur,
            ut,
            np.zeros_like(ur),
            np.ones_like(ur),
            bins,
            np.ones(bins.size, dtype=bool),
            r.size,
            r,
            z,
            averaging="reynolds",
        )
        np.testing.assert_allclose(result["radial_mass_flux"], 0.0, atol=1.0e-10)
        np.testing.assert_allclose(result["F_lambda_eddy"], 0.0, atol=1.0e-15)


if __name__ == "__main__":
    unittest.main()
