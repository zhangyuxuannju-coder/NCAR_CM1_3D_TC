import unittest

import numpy as np

from src.se_equation import (
    build_se_diagnostic_fields,
    regularize_inertial_stability_for_ellipticity,
)


class LegacyInertialStabilityTests(unittest.TestCase):
    def _fields(self, vt, r, f):
        shape = vt.shape
        return build_se_diagnostic_fields(
            ut_2d=vt,
            theta_bal_2d=np.full(shape, 300.0),
            rho_2d=np.ones(shape),
            q_2d=np.zeros(shape),
            fnu_2d=np.zeros(shape),
            r_m=r,
            z_m=np.array([0.0, 1000.0, 2000.0]),
            f=f,
        )

    def test_zero_wind_retains_planetary_inertial_stability(self):
        r = np.array([1000.0, 2000.0, 3000.0, 4000.0])
        f = 5.464e-5
        fields = self._fields(np.zeros((3, r.size)), r, f)
        np.testing.assert_allclose(fields["zeta"], 0.0)
        np.testing.assert_allclose(fields["zeta_abs"], f)
        np.testing.assert_allclose(fields["inertial_stability"], f**2)
        self.assertEqual(fields["f"], f)

    def test_solid_body_rotation_matches_classical_formula(self):
        r = np.array([1000.0, 2000.0, 3000.0, 4000.0])
        f = 5.464e-5
        omega = 1.0e-4
        vt = np.broadcast_to((omega * r)[None, :], (3, r.size))
        fields = self._fields(vt, r, f)
        expected = (f + 2.0 * omega) ** 2
        # The legacy helper intentionally uses first-order one-sided gradients
        # at the two radial boundaries; the centered interior points are exact
        # for this solid-body-rotation check.
        np.testing.assert_allclose(fields["zeta"][:, 1:-1], 2.0 * omega)
        np.testing.assert_allclose(fields["inertial_stability"][:, 1:-1], expected)

    def test_regularization_uses_runtime_f_and_absolute_vorticity(self):
        r = np.array([1000.0, 2000.0, 3000.0, 4000.0])
        z = np.array([0.0, 1000.0, 2000.0])
        f = 5.464e-5
        fields = self._fields(np.zeros((z.size, r.size)), r, f)
        _, _, k3, _ = regularize_inertial_stability_for_ellipticity(
            fields, r, z, margin=0.0, eps_ratio=1.0e-3, max_iter=20
        )
        np.testing.assert_allclose(k3, (1.0 / 300.0) * f**2)


if __name__ == "__main__":
    unittest.main()
