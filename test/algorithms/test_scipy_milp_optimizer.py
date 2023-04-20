# This code is part of Qiskit.
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

""" Test ScipyMilP Optimizer """

import unittest
from test.optimization_test_case import QiskitOptimizationTestCase

from ddt import ddt

import qiskit_optimization.optionals as _optionals
from qiskit_optimization.algorithms import ScipyMilpOptimizer
from qiskit_optimization.problems import QuadraticProgram


@ddt
class TestScipyMilpOptimizer(QiskitOptimizationTestCase):
    """ScipyMilp Optimizer Tests."""

    @unittest.skipIf(not _optionals.HAS_SCIPY_MILP, "Scipy MILP solver not available.")
    def test_scipy_milp_optimizer(self):
        """ScipyMilp Optimizer Test"""
        problem = QuadraticProgram()
        problem.continuous_var(upperbound=4)
        problem.continuous_var(upperbound=4)
        problem.linear_constraint(linear=[1, 1], sense="=", rhs=2)
        problem.minimize(linear=[2, -2])

        optimizer = ScipyMilpOptimizer(disp=False)
        result = optimizer.solve(problem)

        self.assertAlmostEqual(result.fval, -4)


if __name__ == "__main__":
    unittest.main()
