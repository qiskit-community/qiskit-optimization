# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test LinearExpression """

import unittest
from test.optimization_test_case import QiskitOptimizationTestCase

import numpy as np
from scipy.sparse import dok_matrix

from qiskit_optimization import INFINITY, QiskitOptimizationError, QuadraticProgram
from qiskit_optimization.problems import LinearExpression


class TestLinearExpression(QiskitOptimizationTestCase):
    """Test LinearExpression."""

    def test_init(self):
        """test init."""

        quadratic_program = QuadraticProgram()
        for _ in range(5):
            quadratic_program.continuous_var()

        coefficients_list = list(range(5))
        coefficients_array = np.array(coefficients_list)
        coefficients_dok = dok_matrix([coefficients_list])
        coefficients_dict_int = {i: i for i in range(1, 5)}
        coefficients_dict_str = {"x{}".format(i): i for i in range(1, 5)}

        for coeffs in [
            coefficients_list,
            coefficients_array,
            coefficients_dok,
            coefficients_dict_int,
            coefficients_dict_str,
        ]:
            linear = LinearExpression(quadratic_program, coeffs)
            self.assertEqual((linear.coefficients != coefficients_dok).nnz, 0)
            self.assertTrue((linear.to_array() == coefficients_list).all())
            self.assertDictEqual(linear.to_dict(use_name=False), coefficients_dict_int)
            self.assertDictEqual(linear.to_dict(use_name=True), coefficients_dict_str)

    def test_get_item(self):
        """test get_item."""

        quadratic_program = QuadraticProgram()
        for _ in range(5):
            quadratic_program.continuous_var()

        coefficients = list(range(5))
        linear = LinearExpression(quadratic_program, coefficients)
        for i, v in enumerate(coefficients):
            self.assertEqual(linear[i], v)

    def test_setters(self):
        """test setters."""

        quadratic_program = QuadraticProgram()
        for _ in range(5):
            quadratic_program.continuous_var()

        zeros = np.zeros(quadratic_program.get_num_vars())
        linear = LinearExpression(quadratic_program, zeros)

        coefficients_list = list(range(5))
        coefficients_array = np.array(coefficients_list)
        coefficients_dok = dok_matrix([coefficients_list])
        coefficients_dict_int = {i: i for i in range(1, 5)}
        coefficients_dict_str = {"x{}".format(i): i for i in range(1, 5)}

        for coeffs in [
            coefficients_list,
            coefficients_array,
            coefficients_dok,
            coefficients_dict_int,
            coefficients_dict_str,
        ]:
            linear.coefficients = coeffs
            self.assertEqual((linear.coefficients != coefficients_dok).nnz, 0)
            self.assertTrue((linear.to_array() == coefficients_list).all())
            self.assertDictEqual(linear.to_dict(use_name=False), coefficients_dict_int)
            self.assertDictEqual(linear.to_dict(use_name=True), coefficients_dict_str)

    def test_evaluate(self):
        """test evaluate."""

        quadratic_program = QuadraticProgram()
        x = [quadratic_program.continuous_var() for _ in range(5)]

        coefficients_list = list(range(5))
        linear = LinearExpression(quadratic_program, coefficients_list)

        values_list = list(range(len(x)))
        values_array = np.array(values_list)
        values_dict_int = {i: i for i in range(len(x))}
        values_dict_str = {"x{}".format(i): i for i in range(len(x))}

        for values in [values_list, values_array, values_dict_int, values_dict_str]:
            self.assertEqual(linear.evaluate(values), 30)

    def test_evaluate_gradient(self):
        """test evaluate gradient."""

        quadratic_program = QuadraticProgram()
        x = [quadratic_program.continuous_var() for _ in range(5)]

        coefficients_list = list(range(5))
        linear = LinearExpression(quadratic_program, coefficients_list)

        values_list = list(range(len(x)))
        values_array = np.array(values_list)
        values_dict_int = {i: i for i in range(len(x))}
        values_dict_str = {"x{}".format(i): i for i in range(len(x))}

        for values in [values_list, values_array, values_dict_int, values_dict_str]:
            np.testing.assert_almost_equal(linear.evaluate_gradient(values), coefficients_list)

    def test_bounds(self):
        """test lowerbound and upperbound"""

        with self.subTest("bounded"):
            quadratic_program = QuadraticProgram()
            quadratic_program.continuous_var_list(3, lowerbound=-1, upperbound=2)
            coefficients_list = list(range(3))
            bounds = LinearExpression(quadratic_program, coefficients_list).bounds
            self.assertAlmostEqual(bounds.lowerbound, -3)
            self.assertAlmostEqual(bounds.upperbound, 6)

        with self.subTest("bounded2"):
            quadratic_program = QuadraticProgram()
            quadratic_program.integer_var(lowerbound=-2, upperbound=-1, name="x")
            quadratic_program.integer_var(lowerbound=2, upperbound=4, name="y")
            bounds = LinearExpression(quadratic_program, {"x": 1, "y": 10}).bounds
            self.assertAlmostEqual(bounds.lowerbound, 18)
            self.assertAlmostEqual(bounds.upperbound, 39)

            bounds = LinearExpression(quadratic_program, {"x": -1, "y": 10}).bounds
            self.assertAlmostEqual(bounds.lowerbound, 21)
            self.assertAlmostEqual(bounds.upperbound, 42)

            bounds = LinearExpression(quadratic_program, {"x": 1, "y": -10}).bounds
            self.assertAlmostEqual(bounds.lowerbound, -42)
            self.assertAlmostEqual(bounds.upperbound, -21)

            bounds = LinearExpression(quadratic_program, {"x": -1, "y": -10}).bounds
            self.assertAlmostEqual(bounds.lowerbound, -39)
            self.assertAlmostEqual(bounds.upperbound, -18)

            bounds = LinearExpression(quadratic_program, {"x": 0, "y": 0}).bounds
            self.assertAlmostEqual(bounds.lowerbound, 0)
            self.assertAlmostEqual(bounds.upperbound, 0)

        with self.assertRaises(QiskitOptimizationError):
            quadratic_program = QuadraticProgram()
            quadratic_program.continuous_var_list(3, lowerbound=0, upperbound=INFINITY)
            coefficients_list = list(range(3))
            _ = LinearExpression(quadratic_program, coefficients_list).bounds

        with self.assertRaises(QiskitOptimizationError):
            quadratic_program = QuadraticProgram()
            quadratic_program.continuous_var_list(3, lowerbound=-INFINITY, upperbound=0)
            coefficients_list = list(range(3))
            _ = LinearExpression(quadratic_program, coefficients_list).bounds


if __name__ == "__main__":
    unittest.main()
