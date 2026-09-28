import unittest

import numpy as np

from src.isentropic_energetics import (
    assess_cycle_closure,
    build_isentropic_streamfunction,
    classify_outflow_work_hypothesis,
    equivalent_potential_temperature,
    integrate_thermodynamic_cycle,
    moist_entropy_proxy,
)


class IsentropicEnergeticsTests(unittest.TestCase):
    def test_theta_e_increases_with_vapor(self):
        theta = np.array([300.0, 300.0])
        pressure = np.array([90000.0, 90000.0])
        theta_e = equivalent_potential_temperature(theta, pressure, np.array([0.005, 0.015]))
        self.assertGreater(theta_e[1], theta_e[0])

    def test_streamfunction_mass_closure_for_balanced_flux_classes(self):
        theta_e = np.array([
            [[[330.0, 350.0], [330.0, 350.0]]],
            [[[330.0, 350.0], [330.0, 350.0]]],
        ]).reshape(2, 2, 2)
        w = np.where(theta_e < 340.0, 1.0, -1.0)
        rho = np.ones_like(w)
        product = build_isentropic_streamfunction(
            theta_e, w, rho, np.array([1000.0, 2000.0]),
            np.array([320.0, 340.0, 360.0]), 1.0,
        )
        self.assertLess(product["mass_closure_ratio"], 1.0e-12)
        np.testing.assert_allclose(product["streamfunction_kg_s"][:, -1], 0.0)

    def test_closed_cycle_work_and_branch_accounting(self):
        # A synthetic rectangular p-s loop with explicit closure.
        temperature = np.array([300.0, 300.0, 220.0, 220.0, 300.0])
        pressure = np.array([100000.0, 90000.0, 20000.0, 30000.0, 100000.0])
        entropy = np.array([5700.0, 5800.0, 5800.0, 5700.0, 5700.0])
        z = np.array([500.0, 1000.0, 14000.0, 12000.0, 500.0])
        result = integrate_thermodynamic_cycle(temperature, pressure, entropy, z)
        self.assertTrue(np.isfinite(result["pressure_work_j_kg"]))
        self.assertGreater(result["heat_input_j_kg"], 0.0)
        self.assertIn("upper_outflow", result["branches"])
        branch_heat = sum(item["heat_j_kg"] for item in result["branches"].values())
        self.assertAlmostEqual(branch_heat, result["net_tds_j_kg"])


    def test_cycle_closure_gate(self):
        passed = assess_cycle_closure(
            {"mass_closure_ratio": 0.08, "first_law_relative_residual": 0.14}
        )
        self.assertTrue(passed["closure_passed"])
        failed = assess_cycle_closure(
            {"mass_closure_ratio": 0.11, "first_law_relative_residual": 0.01}
        )
        self.assertEqual(failed["interpretation"], "partial_budget_only")

    def test_outflow_work_classification(self):
        self.assertEqual(
            classify_outflow_work_hypothesis([0.01, 0.03], (-0.01, 0.02)),
            "li_like_negligible",
        )
        self.assertEqual(
            classify_outflow_work_hypothesis([0.08, 0.09], (0.01, 0.03)),
            "rappin_like_material",
        )
        self.assertEqual(
            classify_outflow_work_hypothesis([0.01, 0.03], None),
            "insufficient",
        )


if __name__ == "__main__":
    unittest.main()
