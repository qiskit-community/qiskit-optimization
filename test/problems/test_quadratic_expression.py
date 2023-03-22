# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test QuadraticExpression """

import unittest
from test.optimization_test_case import QiskitOptimizationTestCase

import numpy as np
from scipy.sparse import dok_matrix

from qiskit_optimization import INFINITY, QiskitOptimizationError, QuadraticProgram
from qiskit_optimization.problems import QuadraticExpression


class TestQuadraticExpression(QiskitOptimizationTestCase):
    """Test QuadraticExpression."""

    def test_init(self):
        """test init."""

        quadratic_program = QuadraticProgram()
        for _ in range(5):
            quadratic_program.continuous_var()

        coefficients_list = [[0 for _ in range(5)] for _ in range(5)]
        for i, v in enumerate(coefficients_list):
            for j, _ in enumerate(v):
                coefficients_list[min(i, j)][max(i, j)] += i * j
        coefficients_array = np.array(coefficients_list)
        coefficients_dok = dok_matrix(coefficients_list)
        coefficients_dict_int = {(i, j): v for (i, j), v in coefficients_dok.items()}
        coefficients_dict_str = {(f"x{i}", f"x{j}"): v for (i, j), v in coefficients_dok.items()}

        for coeffs in [
            coefficients_list,
            coefficients_array,
            coefficients_dok,
            coefficients_dict_int,
            coefficients_dict_str,
        ]:
            quadratic = QuadraticExpression(quadratic_program, coeffs)
            self.assertEqual((quadratic.coefficients != coefficients_dok).nnz, 0)
            self.assertTrue((quadratic.to_array() == coefficients_list).all())
            self.assertDictEqual(quadratic.to_dict(use_name=False), coefficients_dict_int)
            self.assertDictEqual(quadratic.to_dict(use_name=True), coefficients_dict_str)

    def test_get_item(self):
        """test get_item."""

        quadratic_program = QuadraticProgram()
        for _ in range(5):
            quadratic_program.continuous_var()

        coefficients = [[0 for _ in range(5)] for _ in range(5)]
        for i, v in enumerate(coefficients):
            for j, _ in enumerate(v):
                coefficients[min(i, j)][max(i, j)] += i * j
        quadratic = QuadraticExpression(quadratic_program, coefficients)
        for i, j_v in enumerate(coefficients):
            for j, j_v_j in enumerate(j_v):
                if i == j:
                    self.assertEqual(quadratic[i, j], j_v_j)
                else:
                    self.assertEqual(quadratic[i, j], j_v_j + coefficients[j][i])

    def test_setters(self):
        """test setters."""

        quadratic_program = QuadraticProgram()
        for _ in range(5):
            quadratic_program.continuous_var()

        n = quadratic_program.get_num_vars()
        zeros = np.zeros((n, n))
        quadratic = QuadraticExpression(quadratic_program, zeros)

        coefficients_list = [[0 for _ in range(5)] for _ in range(5)]
        for i, v in enumerate(coefficients_list):
            for j, _ in enumerate(v):
                coefficients_list[min(i, j)][max(i, j)] += i * j
        coefficients_array = np.array(coefficients_list)
        coefficients_dok = dok_matrix(coefficients_list)
        coefficients_dict_int = {(i, j): v for (i, j), v in coefficients_dok.items()}
        coefficients_dict_str = {(f"x{i}", f"x{j}"): v for (i, j), v in coefficients_dok.items()}

        for coeffs in [
            coefficients_list,
            coefficients_array,
            coefficients_dok,
            coefficients_dict_int,
            coefficients_dict_str,
        ]:
            quadratic.coefficients = coeffs
            self.assertEqual((quadratic.coefficients != coefficients_dok).nnz, 0)
            self.assertTrue((quadratic.to_array() == coefficients_list).all())
            self.assertDictEqual(quadratic.to_dict(use_name=False), coefficients_dict_int)
            self.assertDictEqual(quadratic.to_dict(use_name=True), coefficients_dict_str)

    def test_evaluate(self):
        """test evaluate."""

        quadratic_program = QuadraticProgram()
        x = [quadratic_program.continuous_var() for _ in range(5)]

        coefficients_list = [[0 for _ in range(5)] for _ in range(5)]
        for i, v in enumerate(coefficients_list):
            for j, _ in enumerate(v):
                coefficients_list[min(i, j)][max(i, j)] += i * j
        quadratic = QuadraticExpression(quadratic_program, coefficients_list)

        values_list = list(range(len(x)))
        values_array = np.array(values_list)
        values_dict_int = {i: i for i in range(len(x))}
        values_dict_str = {f"x{i}": i for i in range(len(x))}

        for values in [values_list, values_array, values_dict_int, values_dict_str]:
            self.assertEqual(quadratic.evaluate(values), 900)

    def test_evaluate_gradient(self):
        """test evaluate gradient."""

        quadratic_program = QuadraticProgram()
        x = [quadratic_program.continuous_var() for _ in range(5)]

        coefficients_list = [[0 for _ in range(5)] for _ in range(5)]
        for i, v in enumerate(coefficients_list):
            for j, _ in enumerate(v):
                coefficients_list[min(i, j)][max(i, j)] += i * j
        quadratic = QuadraticExpression(quadratic_program, coefficients_list)

        values_list = list(range(len(x)))
        values_array = np.array(values_list)
        values_dict_int = {i: i for i in range(len(x))}
        values_dict_str = {f"x{i}": i for i in range(len(x))}

        grad_values = [0.0, 60.0, 120.0, 180.0, 240.0]
        for values in [values_list, values_array, values_dict_int, values_dict_str]:
            np.testing.assert_almost_equal(quadratic.evaluate_gradient(values), grad_values)

    def test_symmetric_set(self):
        """test symmetric set"""
        q_p = QuadraticProgram()
        q_p.binary_var("x")
        q_p.binary_var("y")
        q_p.binary_var("z")
        quad = QuadraticExpression(q_p, {("x", "y"): -1, ("y", "x"): 2, ("z", "x"): 3})
        self.assertDictEqual(quad.to_dict(use_name=True), {("x", "y"): 1, ("x", "z"): 3})
        self.assertDictEqual(
            quad.to_dict(symmetric=True, use_name=True),
            {("x", "y"): 0.5, ("y", "x"): 0.5, ("x", "z"): 1.5, ("z", "x"): 1.5},
        )

    def test_bounds(self):
        """test lowerbound and upperbound"""

        with self.subTest("bounded"):
            q_p = QuadraticProgram()
            q_p.continuous_var(1, 2, "x")
            q_p.continuous_var(-2, -1, "y")
            q_p.continuous_var(-1, 2, "z")
            bounds = QuadraticExpression(q_p, {("x", "y"): 1, ("y", "z"): 2, ("z", "z"): 3}).bounds
            self.assertAlmostEqual(bounds.lowerbound, -12)
            self.assertAlmostEqual(bounds.upperbound, 15)

        with self.subTest("bounded2"):
            q_p = QuadraticProgram()
            q_p.integer_var(-1, 2, "x")
            bounds = QuadraticExpression(q_p, {("x", "x"): 1}).bounds
            self.assertAlmostEqual(bounds.lowerbound, 0)
            self.assertAlmostEqual(bounds.upperbound, 4)

            bounds = QuadraticExpression(q_p, {("x", "x"): -1}).bounds
            self.assertAlmostEqual(bounds.lowerbound, -4)
            self.assertAlmostEqual(bounds.upperbound, 0)

            bounds = QuadraticExpression(q_p, {("x", "x"): 0}).bounds
            self.assertAlmostEqual(bounds.lowerbound, 0)
            self.assertAlmostEqual(bounds.upperbound, 0)

        with self.subTest("bounded3"):
            q_p = QuadraticProgram()
            q_p.integer_var(-2, -1, "x")
            bounds = QuadraticExpression(q_p, {("x", "x"): 1}).bounds
            self.assertAlmostEqual(bounds.lowerbound, 1)
            self.assertAlmostEqual(bounds.upperbound, 4)

            bounds = QuadraticExpression(q_p, {("x", "x"): -1}).bounds
            self.assertAlmostEqual(bounds.lowerbound, -4)
            self.assertAlmostEqual(bounds.upperbound, -1)

            bounds = QuadraticExpression(q_p, {("x", "x"): 0}).bounds
            self.assertAlmostEqual(bounds.lowerbound, 0)
            self.assertAlmostEqual(bounds.upperbound, 0)

        with self.subTest("bounded4"):
            q_p = QuadraticProgram()
            q_p.integer_var(-2, 3, "x")
            q_p.integer_var(-10, 20, "y")
            bounds = QuadraticExpression(q_p, {("x", "y"): 1}).bounds
            self.assertAlmostEqual(bounds.lowerbound, -40)
            self.assertAlmostEqual(bounds.upperbound, 60)

            bounds = QuadraticExpression(q_p, {("x", "y"): -1}).bounds
            self.assertAlmostEqual(bounds.lowerbound, -60)
            self.assertAlmostEqual(bounds.upperbound, 40)

            bounds = QuadraticExpression(q_p, {("x", "y"): 0}).bounds
            self.assertAlmostEqual(bounds.lowerbound, 0)
            self.assertAlmostEqual(bounds.upperbound, 0)

        with self.assertRaises(QiskitOptimizationError):
            q_p = QuadraticProgram()
            q_p.continuous_var(1, 2, "x")
            q_p.continuous_var(-2, -1, "y")
            q_p.continuous_var(-INFINITY, 2, "z")
            _ = QuadraticExpression(q_p, {("x", "y"): 1, ("y", "z"): 2, ("z", "z"): 3}).bounds

        with self.assertRaises(QiskitOptimizationError):
            q_p = QuadraticProgram()
            q_p.continuous_var(1, 2, "x")
            q_p.continuous_var(-2, -1, "y")
            q_p.continuous_var(-1, INFINITY, "z")
            _ = QuadraticExpression(q_p, {("x", "y"): 1, ("y", "z"): 2, ("z", "z"): 3}).bounds

    def test_str_repr(self):
        """Test str and repr"""
        with self.subTest("5 variables"):
            n = 5
            q_p = QuadraticProgram()
            q_p.binary_var_list(n)  # x0,...,x4
            expr = QuadraticExpression(q_p, {(i, i): float(i) for i in range(n)})
            self.assertEqual(str(expr), "x1^2 + 2*x2^2 + 3*x3^2 + 4*x4^2")
            self.assertEqual(repr(expr), "<QuadraticExpression: x1^2 + 2*x2^2 + 3*x3^2 + 4*x4^2>")

            expr = QuadraticExpression(q_p, {(i, (i + 1) % n): float(i) for i in range(n)})
            self.assertEqual(str(expr), "4*x0*x4 + x1*x2 + 2*x2*x3 + 3*x3*x4")
            self.assertEqual(
                repr(expr), "<QuadraticExpression: 4*x0*x4 + x1*x2 + 2*x2*x3 + 3*x3*x4>"
            )

        with self.subTest("50 variables"):
            # pylint: disable=cyclic-import
            from qiskit_optimization.translators.prettyprint import DEFAULT_TRUNCATE

            n = 50
            q_p = QuadraticProgram()
            q_p.binary_var_list(n)  # x0,...,x49
            expr = QuadraticExpression(q_p, {(i, i): float(i) for i in range(n)})
            expected = " ".join(
                ["x1^2"]
                + sorted([f"+ {i}*x{i}^2" for i in range(2, n)], key=lambda e: e.split(" ")[1])
            )
            self.assertEqual(str(expr), expected)
            self.assertEqual(repr(expr), f"<QuadraticExpression: {expected[:DEFAULT_TRUNCATE]}...>")

            expr = QuadraticExpression(q_p, {(i, (i + 1) % n): float(i) for i in range(n)})
            expected = " ".join(
                [f"{n-1}*x0*x{n-1}", "+ x1*x2"]
                + sorted(
                    [f"+ {i}*x{i}*x{i + 1}" for i in range(2, n - 1)], key=lambda e: e.split(" ")[1]
                )
            )
            self.assertEqual(str(expr), expected)
            self.assertEqual(repr(expr), f"<QuadraticExpression: {expected[:DEFAULT_TRUNCATE]}...>")


if __name__ == "__main__":
    unittest.main()
