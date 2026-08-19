import unittest

import numpy as np

from src.se_bui import (
    G,
    build_forcing,
    invert_balanced_theta,
    regularize_ellipticity,
)


class BuiSECoreTests(unittest.TestCase):
    def test_thermal_forcing_uses_gradient_wind_Cg(self):
        r = np.array([1000.0, 2000.0, 3000.0])
        z = np.array([0.0, 1000.0, 2000.0])
        chi = np.full((3, 3), 1.0 / 300.0)
        cg = np.broadcast_to(np.array([0.0, 1.0, 4.0])[:, None], chi.shape)
        basic = {"chi": chi, "Cg": cg, "xi": np.full_like(chi, 1.0e-4)}
        q = np.ones_like(chi)
        result = build_forcing(basic, q, np.zeros_like(q), r, z)
        expected = np.gradient(cg * chi**2, z, axis=0, edge_order=2)
        np.testing.assert_allclose(result["forcing_thermal"], expected)

    def test_regularization_makes_operator_elliptic(self):
        k1 = np.array([[1.0, -1.0], [1.0, 1.0]])
        k2 = np.array([[2.0, 2.0], [0.5, 3.0]])
        k3 = np.array([[1.0, 1.0], [-1.0, 1.0]])
        a, b, c, info = regularize_ellipticity(k1, k2, k3, eps_ratio=1.0e-3)
        self.assertTrue(np.all(a > 0.0))
        self.assertTrue(np.all(c > 0.0))
        self.assertTrue(np.all(a * c - b**2 > 0.0))
        self.assertGreater(info["baroclinic_adjusted_points"], 0.0)

    def test_barotropic_constant_theta_is_exact_balance(self):
        r = np.array([1000.0, 2000.0, 3000.0, 4000.0])
        z = np.array([0.0, 1000.0, 2000.0])
        vt = np.broadcast_to((1.0e-3 * r)[None, :], (z.size, r.size))
        theta = np.full_like(vt, 300.0)
        theta_bal, info = invert_balanced_theta(vt, theta, r, z, 5.0e-5)
        np.testing.assert_allclose(theta_bal, 300.0, atol=1.0e-10)
        self.assertLess(info["thermal_wind_residual_rms"], 1.0e-12)


if __name__ == "__main__":
    unittest.main()

