import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.se_applicability import (
    classify_stability,
    compute_applicability_diagnostics,
    compute_case_stability,
    summarize_case_overlap,
    write_applicability_products,
)
from src.se_bui import build_basic_state


class SEApplicabilityTests(unittest.TestCase):
    def test_four_physical_classes_are_mutually_exclusive(self):
        k1 = np.array([[1.0, 1.0, 1.0, -1.0]])
        i2 = np.array([[1.0, -1.0, 1.0, 1.0]])
        disc = np.array([[1.0, -2.0, -1.0, 1.0]])
        expected = np.array([[0, 1, 2, 3]], dtype=np.int8)
        np.testing.assert_array_equal(classify_stability(k1, i2, disc), expected)

    def test_negative_I2_implies_negative_D_for_static_stability(self):
        k1 = np.array([[2.0, 3.0]])
        i2 = np.array([[-1.0, -0.5]])
        shear = np.array([[0.0, 4.0]])
        disc = k1 * i2 - shear**2
        self.assertTrue(np.all(disc < 0.0))

    def test_forcing_overlap_reports_nonelliptic_fraction(self):
        fields = {
            "I2_raw": np.ones((2, 2)),
            "D_raw": np.array([[1.0, -1.0], [1.0, 1.0]]),
            "stability_class": np.array([[0, 2], [0, 0]], dtype=np.int8),
        }
        forcing = np.array([[0.0, 10.0], [0.0, 0.0]])
        summary = summarize_case_overlap(
            fields,
            forcing,
            r_m=np.array([1000.0, 2000.0]),
            z_m=np.array([0.0, 1000.0]),
        )
        self.assertAlmostEqual(summary["nonelliptic_abs_forcing_fraction"], 1.0)

    def test_raw_fields_use_model_theta_before_balance_projection(self):
        r = np.array([1000.0, 2000.0, 3000.0, 4000.0])
        z = np.array([0.0, 1000.0, 2000.0])
        vt = np.broadcast_to(np.array([5.0, 10.0, 15.0, 20.0]), (z.size, r.size)).copy()
        vt[1] += 2.0
        theta = 300.0 + z[:, None] * 0.003 + r[None, :] * 0.0002
        rho = np.ones_like(theta)
        expected = build_basic_state(vt, theta, rho, r, z, 5.0e-5, baroclinic_scale=1.0)
        fields, _ = compute_case_stability(vt, theta, rho, r, z, 5.0e-5)
        np.testing.assert_allclose(fields["I2_raw"], expected["K3_raw"])
        np.testing.assert_allclose(
            fields["D_raw"],
            expected["K1_raw"] * expected["K3_raw"] - expected["K2_raw"] ** 2,
        )

    def test_ctrl_jet_diagnostic_exposes_raw_and_projection_fields(self):
        r_km = np.array([1.0, 2.0, 3.0, 4.0])
        z_km = np.array([0.0, 1.0, 2.0])
        shape = (z_km.size, r_km.size)
        theta = 300.0 + z_km[:, None] * 3.0 + r_km[None, :] * 0.2
        base = {
            "r_km": r_km,
            "z_km": z_km,
            "ut": np.broadcast_to(np.array([5.0, 10.0, 15.0, 20.0]), shape).copy(),
            "theta": theta,
            "rho": np.ones(shape),
            "ur": np.zeros(shape),
            "F_lambda_eddy": np.zeros(shape),
            "F_lambda_eddy_radial": np.zeros(shape),
            "F_lambda_eddy_vertical": np.zeros(shape),
            "eddy_kinetic_energy": np.zeros(shape),
        }
        jet = {key: np.array(value, copy=True) for key, value in base.items()}
        jet["F_lambda_eddy"][1, 2] = 1.0e-4
        jet["F_lambda_eddy_radial"][1, 2] = 1.0e-4
        arrays, summary = compute_applicability_diagnostics(base, jet, 5.0e-5)
        self.assertIn("I2_raw_ctrl", arrays)
        self.assertIn("I2_balanced_projection_ctrl", arrays)
        self.assertIn("D_regularized_jet", arrays)
        self.assertEqual(summary["maximum_abs_environmental_forcing"]["r_km"], 3.0)
        with TemporaryDirectory() as tmp:
            outputs = write_applicability_products(
                arrays, summary, tmp, write_netcdf=False, make_plots=False
            )
            self.assertTrue(Path(outputs["npz"]).exists())
            self.assertTrue((Path(tmp) / "se_applicability_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
