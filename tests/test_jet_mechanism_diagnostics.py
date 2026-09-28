import unittest

import numpy as np

from src.jet_mechanism_diagnostics import (
    angular_momentum_inertial_stability,
    centered_tendency,
    moving_block_bootstrap_lead_lag,
    sustained_sign_transitions,
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


    def test_centered_pressure_intensification_rate(self):
        time = np.arange(0.0, 13.0)
        pressure = 1000.0 - 2.0 * time
        rate = centered_tendency(time, pressure, 6.0, stronger_is_larger=False)
        np.testing.assert_allclose(rate[3:-3], 2.0)
        self.assertTrue(np.all(np.isnan(rate[:3])))

    def test_sustained_transition_filters_one_sample_flip(self):
        time = np.arange(8.0)
        values = np.array([1, 1, -1, 1, 1, -1, -1, -1], float)
        transitions = sustained_sign_transitions(time, values, minimum_duration_h=2.0)
        np.testing.assert_allclose(transitions, [5.0])

    def test_block_bootstrap_lead_lag_returns_interval(self):
        predictor = np.sin(np.arange(40.0) / 4.0)
        response = np.r_[np.nan, predictor[:-1]]
        result = moving_block_bootstrap_lead_lag(
            predictor, response, 1.0, [1.0], block_h=4.0, samples=50, seed=4
        )
        self.assertEqual(result["ci_low"].shape, (1,))
        self.assertLessEqual(result["ci_low"][0], result["ci_high"][0])


if __name__ == "__main__":
    unittest.main()
