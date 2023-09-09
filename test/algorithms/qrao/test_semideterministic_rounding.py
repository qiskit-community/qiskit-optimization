# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for SemideterministicRounding"""
import unittest
from test.optimization_test_case import QiskitOptimizationTestCase

import numpy as np

from qiskit_optimization.algorithms.qrao import (
    QuantumRandomAccessEncoding,
    SemideterministicRounding,
    RoundingResult,
    RoundingContext,
)
from qiskit_optimization.algorithms import SolutionSample
from qiskit_optimization.problems import QuadraticProgram


class TestSemideterministicRounding(QiskitOptimizationTestCase):
    """SemideterministicRounding tests."""

    def setUp(self):
        super().setUp()
        self.problem = QuadraticProgram()
        var_dict = self.problem.binary_var_dict(5)
        self.problem.minimize(linear={name: 1 for name in var_dict})

    def test_semideterministic_rounding(self):
        """Test SemideterministicRounding"""
        encoding = QuantumRandomAccessEncoding()
        encoding.encode(self.problem)
        rounding_scheme = SemideterministicRounding(seed=123)
        expectation_values = [1, -1, 0, 0.7, -0.3]
        result = rounding_scheme.round(
            RoundingContext(expectation_values=expectation_values, encoding=encoding)
        )
        self.assertIsInstance(result, RoundingResult)
        self.assertIsInstance(result.samples[0], SolutionSample)
        self.assertEqual(result.expectation_values, [1, -1, 0, 0.7, -0.3])
        np.testing.assert_array_almost_equal(result.samples[0].x, [0, 1, 1, 0, 1])
        self.assertEqual(result.samples[0].probability, 1.0)


if __name__ == "__main__":
    unittest.main()
