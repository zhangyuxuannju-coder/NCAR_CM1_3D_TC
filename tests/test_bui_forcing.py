import unittest

import numpy as np

from src.bui_forcing import assemble_bui_forcings, reconstruct_eddy_from_advection


class BuiForcingTests(unittest.TestCase):
    def test_total_excludes_resolved_advection(self):
        shape = (3, 4)
        q_eddy = np.full(shape, 1.0)
        f_eddy = np.full(shape, 2.0)
        parts = assemble_bui_forcings(
            q_eddy,
            f_eddy,
            {
                "mp": np.full(shape, 3.0),
                "hidiff": np.full(shape, 4.0),
                "hadv": np.full(shape, 1000.0),
            },
            {
                "pbl": np.full(shape, 5.0),
                "rdamp": np.full(shape, 6.0),
                "hadv": np.full(shape, 1000.0),
                "vadv": np.full(shape, 1000.0),
            },
        )
        np.testing.assert_allclose(parts["Q_total"], 8.0)
        np.testing.assert_allclose(parts["F_lambda_total"], 13.0)

    def test_advection_reconstruction_has_one_curvature_term(self):
        r = np.array([1.0, 2.0, 3.0])
        z = np.array([0.0, 1.0, 2.0])
        shape = (z.size, r.size)
        ur = np.ones(shape)
        ut = np.full(shape, 2.0)
        w = np.zeros(shape)
        # For constant mean tangential wind, mean horizontal advection is
        # -ubar*vbar/r.  Adding it back must yield exactly zero eddy forcing.
        hadv = -ur * ut / r[None, :]
        result = reconstruct_eddy_from_advection(
            hadv, np.zeros(shape), ur, ut, w, r, z
        )
        np.testing.assert_allclose(result["F_lambda_eddy_budget"], 0.0, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
