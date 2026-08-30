import unittest

import numpy as np

from src.jet_mechanism_diagnostics import (
    angular_momentum_inertial_stability,
    cylindrical_wind,
    lead_lag_correlation,
    match_by_intensity,
    radial_bin_indices,
    radial_mean,
    storm_relative_geometry,
)


class JetMechanismDiagnosticsTests(unittest.TestCase):
    def test_cylindrical_projection_of_pure_radial_flow(self):
        x = np.array([-1.0, 1.0])
        y = np.array([-1.0, 1.0])
        geom = storm_relative_geometry(x, y, 0.0, 0.0)
        u = geom["cos_azimuth"][None] * 5.0
        v = geom["sin_azimuth"][None] * 5.0
        ur, vt = cylindrical_wind(u, v, geom["cos_azimuth"], geom["sin_azimuth"])
        np.testing.assert_allclose(ur, 5.0)
        np.testing.assert_allclose(vt, 0.0, atol=1.0e-14)

    def test_axisymmetric_field_has_exact_radial_mean(self):
        x = np.linspace(-2.0, 2.0, 5)
        y = np.linspace(-2.0, 2.0, 5)
        geom = storm_relative_geometry(x, y, 0.0, 0.0)
        edges = np.array([0.0, 1.1, 2.1, 3.1])
        idx, valid = radial_bin_indices(geom["radius_m"], edges)
        field = np.broadcast_to(3.0 + idx.reshape(5, 5), (2, 5, 5)).astype(float)
        mean = radial_mean(field, idx, valid, 3)
        np.testing.assert_allclose(mean[0], [3.0, 4.0, 5.0])
        np.testing.assert_allclose(mean[1], [3.0, 4.0, 5.0])

    def test_solid_body_inertial_stability(self):
        r = np.linspace(1000.0, 100000.0, 101)
        omega = 2.0e-4
        f = 5.0e-5
        vt = (omega * r)[None, :]
        result = angular_momentum_inertial_stability(vt, r, f)
        expected = (f + 2.0 * omega) ** 2
        np.testing.assert_allclose(result["I2"][:, 2:-2], expected, rtol=1.0e-10)

    def test_strength_matching(self):
        matched = match_by_intensity(
            np.array([995.0, 980.0]), np.array([1000.0, 990.0, 980.0]), np.array([0.0, 6.0, 12.0])
        )
        np.testing.assert_array_equal(matched["index"], [0, 2])
        np.testing.assert_allclose(matched["time_h"], [0.0, 12.0])

    def test_positive_lead_means_predictor_leads(self):
        predictor = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0])
        response = np.r_[np.nan, predictor[:-1]]
        result = lead_lag_correlation(predictor, response, 1.0, [0.0, 1.0])
        self.assertAlmostEqual(result["correlation"][1], 1.0)


if __name__ == "__main__":
    unittest.main()
