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

""" Test ScipyMilP Optimizer """

import unittest
from test.optimization_test_case import QiskitOptimizationTestCase

import numpy as np
from ddt import ddt

from qiskit_optimization import INFINITY
from qiskit_optimization.algorithms import OptimizationResultStatus, ScipyMilpOptimizer
from qiskit_optimization.exceptions import QiskitOptimizationError
from qiskit_optimization.problems import QuadraticProgram


@ddt
class TestScipyMilpOptimizer(QiskitOptimizationTestCase):
    """ScipyMilp Optimizer Tests."""

    def test_scipy_milp_optimizer(self):
        """ScipyMilp Optimizer Test"""
        problem = QuadraticProgram()
        problem.continuous_var(upperbound=4)
        problem.continuous_var(upperbound=4)
        problem.linear_constraint(linear=[1, 1], sense="=", rhs=2)
        problem.minimize(linear=[2, -2])

        optimizer = ScipyMilpOptimizer()
        result = optimizer.solve(problem)

        self.assertAlmostEqual(result.fval, -4)

    def test_disp(self):
        """Test disp"""
        with self.subTest("default"):
            optimizer = ScipyMilpOptimizer()
            self.assertFalse(optimizer.disp)
        with self.subTest("init True"):
            optimizer = ScipyMilpOptimizer(disp=True)
            self.assertTrue(optimizer.disp)
        with self.subTest("setter / True"):
            optimizer = ScipyMilpOptimizer(disp=False)
            optimizer.disp = True
            self.assertTrue(optimizer.disp)
        with self.subTest("setter / False"):
            optimizer = ScipyMilpOptimizer(disp=True)
            optimizer.disp = False
            self.assertFalse(optimizer.disp)

    def test_compatibility(self):
        """Test compatibility"""
        with self.subTest("quadratic obj min"):
            problem = QuadraticProgram()
            _ = problem.binary_var("x")
            problem.minimize(quadratic={(0, 0): 1})
            optimizer = ScipyMilpOptimizer()
            msg = optimizer.get_compatibility_msg(problem)
            self.assertEqual(msg, "scipy.milp supports only linear objective function")
        with self.subTest("quadratic obj max"):
            problem = QuadraticProgram()
            _ = problem.binary_var("x")
            problem.maximize(quadratic={(0, 0): 1})
            optimizer = ScipyMilpOptimizer()
            msg = optimizer.get_compatibility_msg(problem)
            self.assertEqual(msg, "scipy.milp supports only linear objective function")
        with self.subTest("quadratic constraint"):
            problem = QuadraticProgram()
            _ = problem.binary_var("x")
            problem.quadratic_constraint(quadratic={(0, 0): 1})
            optimizer = ScipyMilpOptimizer()
            msg = optimizer.get_compatibility_msg(problem)
            self.assertEqual(msg, "scipy.milp supports only linear constraints")
        with self.subTest("quadratic obj / quadratic constraint"):
            problem = QuadraticProgram()
            _ = problem.binary_var("x")
            problem.minimize(quadratic={(0, 0): 1})
            problem.quadratic_constraint(quadratic={(0, 0): 1})
            optimizer = ScipyMilpOptimizer()
            msg = optimizer.get_compatibility_msg(problem)
            self.assertEqual(
                msg,
                "scipy.milp supports only linear objective function; "
                "scipy.milp supports only linear constraints",
            )

    def test_error(self):
        """Test errors"""
        with self.subTest("quadratic obj min"):
            problem = QuadraticProgram()
            _ = problem.binary_var("x")
            problem.minimize(quadratic={(0, 0): 1})
            optimizer = ScipyMilpOptimizer()
            with self.assertRaises(QiskitOptimizationError):
                _ = optimizer.solve(problem)
        with self.subTest("quadratic obj max"):
            problem = QuadraticProgram()
            _ = problem.binary_var("x")
            problem.maximize(quadratic={(0, 0): 1})
            optimizer = ScipyMilpOptimizer()
            with self.assertRaises(QiskitOptimizationError):
                _ = optimizer.solve(problem)
        with self.subTest("quadratic constraint"):
            problem = QuadraticProgram()
            _ = problem.binary_var("x")
            problem.quadratic_constraint(quadratic={(0, 0): 1})
            optimizer = ScipyMilpOptimizer()
            with self.assertRaises(QiskitOptimizationError):
                _ = optimizer.solve(problem)

    def test_senses(self):
        """Test constraint senses"""
        with self.subTest("min / <="):
            problem = QuadraticProgram()
            _ = problem.continuous_var(0, 1, "x")
            _ = problem.binary_var("y")
            problem.minimize(linear={"x": 1, "y": -1})
            problem.linear_constraint({"x": 2, "y": 1}, "<=", 1)
            optimizer = ScipyMilpOptimizer()
            result = optimizer.solve(problem)
            self.assertAlmostEqual(result.fval, -1)
            np.testing.assert_allclose(result.x, [0, 1])
            self.assertEqual(result.status, OptimizationResultStatus.SUCCESS)

        with self.subTest("max / <="):
            problem = QuadraticProgram()
            _ = problem.continuous_var(0, 1, "x")
            _ = problem.binary_var("y")
            problem.maximize(linear={"x": 1, "y": -1})
            problem.linear_constraint({"x": 2, "y": 1}, "<=", 1)
            optimizer = ScipyMilpOptimizer()
            result = optimizer.solve(problem)
            self.assertAlmostEqual(result.fval, 0.5)
            np.testing.assert_allclose(result.x, [0.5, 0])
            self.assertEqual(result.status, OptimizationResultStatus.SUCCESS)

        with self.subTest("min / >="):
            problem = QuadraticProgram()
            _ = problem.continuous_var(0, 1, "x")
            _ = problem.binary_var("y")
            problem.minimize(linear={"x": 1, "y": -1})
            problem.linear_constraint({"x": 2, "y": 1}, ">=", 1)
            optimizer = ScipyMilpOptimizer()
            result = optimizer.solve(problem)
            self.assertAlmostEqual(result.fval, -1.0)
            np.testing.assert_allclose(result.x, [0, 1.0])
            self.assertEqual(result.status, OptimizationResultStatus.SUCCESS)

        with self.subTest("max / >="):
            problem = QuadraticProgram()
            _ = problem.continuous_var(0, 1, "x")
            _ = problem.binary_var("y")
            problem.maximize(linear={"x": 1, "y": -1})
            problem.linear_constraint({"x": 2, "y": 1}, ">=", 1)
            optimizer = ScipyMilpOptimizer()
            result = optimizer.solve(problem)
            self.assertAlmostEqual(result.fval, 1)
            np.testing.assert_allclose(result.x, [1, 0])
            self.assertEqual(result.status, OptimizationResultStatus.SUCCESS)

        with self.subTest("min / =="):
            problem = QuadraticProgram()
            _ = problem.continuous_var(0, 1, "x")
            _ = problem.binary_var("y")
            problem.minimize(linear={"x": 1, "y": -1})
            problem.linear_constraint({"x": 2, "y": 1}, "==", 2)
            optimizer = ScipyMilpOptimizer()
            result = optimizer.solve(problem)
            self.assertAlmostEqual(result.fval, -0.5)
            np.testing.assert_allclose(result.x, [0.5, 1.0])
            self.assertEqual(result.status, OptimizationResultStatus.SUCCESS)

        with self.subTest("max / =="):
            problem = QuadraticProgram()
            _ = problem.continuous_var(0, 1, "x")
            _ = problem.binary_var("y")
            problem.maximize(linear={"x": 1, "y": -1})
            problem.linear_constraint({"x": 2, "y": 1}, "==", 2)
            optimizer = ScipyMilpOptimizer()
            result = optimizer.solve(problem)
            self.assertAlmostEqual(result.fval, 1)
            np.testing.assert_allclose(result.x, [1, 0])
            self.assertEqual(result.status, OptimizationResultStatus.SUCCESS)

    def test_infinity(self):
        """Test infinity"""
        with self.subTest("ub = infinity default"):
            problem = QuadraticProgram()
            _ = problem.continuous_var(name="x")
            problem.maximize(linear={0: 1})
            optimizer = ScipyMilpOptimizer()
            with self.assertWarns(UserWarning):
                result = optimizer.solve(problem)
            self.assertEqual(result.status, OptimizationResultStatus.FAILURE)

        with self.subTest("ub = infinity manual"):
            problem = QuadraticProgram()
            _ = problem.continuous_var(upperbound=INFINITY, name="x")
            problem.maximize(linear={0: 1})
            optimizer = ScipyMilpOptimizer()
            with self.assertWarns(UserWarning):
                result = optimizer.solve(problem)
            self.assertEqual(result.status, OptimizationResultStatus.FAILURE)

        with self.subTest("ub > infinity"):
            problem = QuadraticProgram()
            _ = problem.continuous_var(upperbound=1e100, name="x")
            problem.maximize(linear={0: 1})
            optimizer = ScipyMilpOptimizer()
            with self.assertWarns(UserWarning):
                result = optimizer.solve(problem)
            self.assertEqual(result.status, OptimizationResultStatus.FAILURE)

        with self.subTest("ub < infinity"):
            problem = QuadraticProgram()
            _ = problem.continuous_var(upperbound=1e10, name="x")
            problem.maximize(linear={0: 1})
            optimizer = ScipyMilpOptimizer()
            result = optimizer.solve(problem)
            np.testing.assert_allclose(result.x, [1e10])
            self.assertEqual(result.status, OptimizationResultStatus.SUCCESS)

        with self.subTest("lb = -infinity manual"):
            problem = QuadraticProgram()
            _ = problem.continuous_var(lowerbound=-INFINITY, name="x")
            problem.minimize(linear={0: 1})
            optimizer = ScipyMilpOptimizer()
            with self.assertWarns(UserWarning):
                result = optimizer.solve(problem)
            self.assertEqual(result.status, OptimizationResultStatus.FAILURE)

        with self.subTest("lb < -infinity"):
            problem = QuadraticProgram()
            _ = problem.continuous_var(lowerbound=-1e100, name="x")
            problem.minimize(linear={0: 1})
            optimizer = ScipyMilpOptimizer()
            with self.assertWarns(UserWarning):
                result = optimizer.solve(problem)
                print(result)
            self.assertEqual(result.status, OptimizationResultStatus.FAILURE)

        with self.subTest("lb > -infinity"):
            problem = QuadraticProgram()
            _ = problem.continuous_var(lowerbound=-1e10, name="x")
            problem.minimize(linear={0: 1})
            optimizer = ScipyMilpOptimizer()
            result = optimizer.solve(problem)
            np.testing.assert_allclose(result.x, [-1e10])
            self.assertEqual(result.status, OptimizationResultStatus.SUCCESS)


if __name__ == "__main__":
    unittest.main()
