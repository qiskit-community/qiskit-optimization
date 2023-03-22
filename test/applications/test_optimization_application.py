# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test OptimizationApplication class"""

import unittest
from test.optimization_test_case import QiskitOptimizationTestCase

import numpy as np
from ddt import data, ddt
from qiskit.opflow import StateFn
from qiskit.result import QuasiDistribution

from qiskit_optimization.applications import OptimizationApplication


@ddt
class TestOptimizationApplication(QiskitOptimizationTestCase):
    """Test OptimizationApplication class"""

    @data(
        np.array([0, 0, 1, 0]),
        StateFn([0, 0, 1, 0]),
        {"10": 0.8, "01": 0.2},
        QuasiDistribution({"10": 0.8, "01": 0.2}),
    )
    def test_sample_most_likely(self, state_vector):
        """Test sample_most_likely"""

        result = OptimizationApplication.sample_most_likely(state_vector)
        np.testing.assert_allclose(result, [0, 1])


if __name__ == "__main__":
    unittest.main()
