# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Goemans-Williamson optimizer. """
from test import QiskitOptimizationTestCase, requires_extra_library

import numpy as np

from qiskit_optimization.algorithms.goemans_williamson_optimizer \
    import (GoemansWilliamsonOptimizer, GoemansWilliamsonOptimizationResult)
from qiskit_optimization.applications.max_cut import Maxcut


class TestGoemansWilliamson(QiskitOptimizationTestCase):
    """Test Goemans-Williamson optimizer."""

    @requires_extra_library
    def test_all_cuts(self):
        """Basic test of the Goemans-Williamson optimizer."""
        graph = np.array([[0., 1., 2., 0.],
                          [1., 0., 1., 0.],
                          [2., 1., 0., 1.],
                          [0., 0., 1., 0.]])

        optimizer = GoemansWilliamsonOptimizer(num_cuts=10, seed=0)

        problem = Maxcut(graph).to_quadratic_program()
        self.assertIsNotNone(problem)

        results = optimizer.solve(problem)
        self.assertIsNotNone(results)
        self.assertIsInstance(results, GoemansWilliamsonOptimizationResult)

        self.assertIsNotNone(results.x)
        np.testing.assert_almost_equal([0, 1, 1, 0], results.x, 3)

        self.assertIsNotNone(results.fval)
        np.testing.assert_almost_equal(4, results.fval, 3)

        self.assertIsNotNone(results.samples)
        self.assertEqual(3, len(results.samples))
