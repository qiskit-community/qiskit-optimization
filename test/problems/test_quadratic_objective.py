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

""" Test QuadraticObjective """

import unittest
from test.optimization_test_case import QiskitOptimizationTestCase

import numpy as np

from qiskit_optimization.problems import QuadraticProgram, QuadraticObjective


class TestQuadraticObjective(QiskitOptimizationTestCase):
    """Test QuadraticObjective"""

    def test_init(self):
        """test init."""

        quadratic_program = QuadraticProgram()
        for _ in range(5):
            quadratic_program.continuous_var()

        self.assertEqual(quadratic_program.objective.constant, 0.0)
        self.assertEqual(len(quadratic_program.objective.linear.to_dict()), 0)
        self.assertEqual(len(quadratic_program.objective.quadratic.to_dict()), 0)
        self.assertEqual(quadratic_program.objective.sense, QuadraticObjective.Sense.MINIMIZE)

        constant = 1.0
        linear_coeffs = np.array(range(5))
        lst = [[0 for _ in range(5)] for _ in range(5)]
        for i, v in enumerate(lst):
            for j, _ in enumerate(v):
                lst[min(i, j)][max(i, j)] += i * j
        quadratic_coeffs = np.array(lst)

        quadratic_program.minimize(constant, linear_coeffs, quadratic_coeffs)

        self.assertEqual(quadratic_program.objective.constant, constant)
        self.assertTrue((quadratic_program.objective.linear.to_array() == linear_coeffs).all())
        self.assertTrue(
            (quadratic_program.objective.quadratic.to_array() == quadratic_coeffs).all()
        )
        self.assertEqual(quadratic_program.objective.sense, QuadraticObjective.Sense.MINIMIZE)

        quadratic_program.maximize(constant, linear_coeffs, quadratic_coeffs)

        self.assertEqual(quadratic_program.objective.constant, constant)
        self.assertTrue((quadratic_program.objective.linear.to_array() == linear_coeffs).all())
        self.assertTrue(
            (quadratic_program.objective.quadratic.to_array() == quadratic_coeffs).all()
        )
        self.assertEqual(quadratic_program.objective.sense, QuadraticObjective.Sense.MAXIMIZE)

        self.assertEqual(quadratic_program.objective.evaluate(linear_coeffs), 931.0)

        grad_values = [0.0, 61.0, 122.0, 183.0, 244.0]
        np.testing.assert_almost_equal(
            quadratic_program.objective.evaluate_gradient(linear_coeffs), grad_values
        )

    def test_setters(self):
        """test setters."""

        quadratic_program = QuadraticProgram()
        for _ in range(5):
            quadratic_program.continuous_var()

        constant = 1.0
        linear_coeffs = np.array(range(5))
        lst = [[0 for _ in range(5)] for _ in range(5)]
        for i, v in enumerate(lst):
            for j, _ in enumerate(v):
                lst[min(i, j)][max(i, j)] += i * j
        quadratic_coeffs = np.array(lst)

        quadratic_program.objective.constant = constant
        quadratic_program.objective.linear = linear_coeffs
        quadratic_program.objective.quadratic = quadratic_coeffs

        self.assertEqual(quadratic_program.objective.constant, constant)
        self.assertTrue((quadratic_program.objective.linear.to_array() == linear_coeffs).all())
        self.assertTrue(
            (quadratic_program.objective.quadratic.to_array() == quadratic_coeffs).all()
        )

        self.assertEqual(quadratic_program.objective.sense, QuadraticObjective.Sense.MINIMIZE)

        quadratic_program.objective.sense = quadratic_program.objective.Sense.MAXIMIZE
        self.assertEqual(quadratic_program.objective.sense, QuadraticObjective.Sense.MAXIMIZE)

        quadratic_program.objective.sense = quadratic_program.objective.Sense.MINIMIZE
        self.assertEqual(quadratic_program.objective.sense, QuadraticObjective.Sense.MINIMIZE)

    def test_str_repr(self):
        """Test str and repr"""
        with self.subTest("4 variables"):
            n = 4
            q_p = QuadraticProgram()
            q_p.binary_var_list(n)  # x0,...,x3
            quad = {(i, i): float(i) for i in range(n)}
            lin = [float(i) for i in range(n)]

            q_p.maximize(quadratic=quad, linear=lin, constant=n)
            expected = "maximize x1^2 + 2*x2^2 + 3*x3^2 + x1 + 2*x2 + 3*x3 + 4"
            self.assertEqual(str(q_p.objective), expected)
            self.assertEqual(repr(q_p.objective), f"<QuadraticObjective: {expected}>")

            q_p.minimize(quadratic=quad, linear=lin, constant=n)
            expected = "minimize x1^2 + 2*x2^2 + 3*x3^2 + x1 + 2*x2 + 3*x3 + 4"
            self.assertEqual(str(q_p.objective), expected)
            self.assertEqual(repr(q_p.objective), f"<QuadraticObjective: {expected}>")

            quad = {(i, (i + 1) % n): i for i in range(n)}

            q_p.maximize(quadratic=quad, linear=lin, constant=0)
            expected = "maximize 3*x0*x3 + x1*x2 + 2*x2*x3 + x1 + 2*x2 + 3*x3"
            self.assertEqual(str(q_p.objective), expected)
            self.assertEqual(repr(q_p.objective), f"<QuadraticObjective: {expected}>")

            q_p.minimize(quadratic=quad, linear=lin, constant=0)
            expected = "minimize 3*x0*x3 + x1*x2 + 2*x2*x3 + x1 + 2*x2 + 3*x3"
            self.assertEqual(str(q_p.objective), expected)
            self.assertEqual(repr(q_p.objective), f"<QuadraticObjective: {expected}>")

        with self.subTest("50 variables"):
            # pylint: disable=cyclic-import
            from qiskit_optimization.translators.prettyprint import DEFAULT_TRUNCATE

            n = 50
            q_p = QuadraticProgram()
            q_p.binary_var_list(n)
            quad = {(i, i): float(i) for i in range(n)}
            lin = [float(i) for i in range(n)]

            expected = " ".join(
                ["x1^2"]
                + sorted([f"+ {i}*x{i}^2" for i in range(2, n)], key=lambda e: e.split(" ")[1])
                + ["+ x1"]
                + sorted([f"+ {i}*x{i}" for i in range(2, n)], key=lambda e: e.split(" ")[1])
                + [f"+ {n}"]
            )
            q_p.maximize(quadratic=quad, linear=lin, constant=n)
            self.assertEqual(str(q_p.objective), f"maximize {expected}")
            self.assertEqual(
                repr(q_p.objective),
                f"<QuadraticObjective: maximize {expected[:DEFAULT_TRUNCATE]}...>",
            )

            q_p.minimize(quadratic=quad, linear=lin, constant=n)
            self.assertEqual(str(q_p.objective), f"minimize {expected}")
            self.assertEqual(
                repr(q_p.objective),
                f"<QuadraticObjective: minimize {expected[:DEFAULT_TRUNCATE]}...>",
            )

            quad = {(i + 1, i): float(i + 1) for i in range(2)}
            lin = [float(i) for i in range(n)]
            expected = " ".join(
                ["x0*x1 + 2*x1*x2"]
                + ["+ x1"]
                + sorted([f"+ {i}*x{i}" for i in range(2, n)], key=lambda e: e.split(" ")[1])
                + [f"+ {n}"]
            )
            q_p.maximize(quadratic=quad, linear=lin, constant=n)
            self.assertEqual(str(q_p.objective), f"maximize {expected}")
            self.assertEqual(
                repr(q_p.objective),
                f"<QuadraticObjective: maximize {expected[:DEFAULT_TRUNCATE]}...>",
            )

            q_p.minimize(quadratic=quad, linear=lin, constant=n)
            self.assertEqual(str(q_p.objective), f"minimize {expected}")
            self.assertEqual(
                repr(q_p.objective),
                f"<QuadraticObjective: minimize {expected[:DEFAULT_TRUNCATE]}...>",
            )

        with self.subTest("only constant"):
            q_p = QuadraticProgram()

            q_p.maximize()
            expected = "maximize 0"
            self.assertEqual(str(q_p.objective), expected)
            self.assertEqual(repr(q_p.objective), f"<QuadraticObjective: {expected}>")

            q_p.minimize()
            expected = "minimize 0"
            self.assertEqual(str(q_p.objective), expected)
            self.assertEqual(repr(q_p.objective), f"<QuadraticObjective: {expected}>")

            q_p.maximize(constant=-1.23)
            expected = "maximize -1.23"
            self.assertEqual(str(q_p.objective), expected)
            self.assertEqual(repr(q_p.objective), f"<QuadraticObjective: {expected}>")

            q_p.minimize(constant=-1.23)
            expected = "minimize -1.23"
            self.assertEqual(str(q_p.objective), expected)
            self.assertEqual(repr(q_p.objective), f"<QuadraticObjective: {expected}>")

        with self.subTest("1 variable"):
            q_p = QuadraticProgram()
            q_p.binary_var("z")

            q_p.maximize(linear=[1], quadratic={("z", "z"): -1}, constant=1)
            expected = "maximize -z^2 + z + 1"
            self.assertEqual(str(q_p.objective), expected)
            self.assertEqual(repr(q_p.objective), f"<QuadraticObjective: {expected}>")

            q_p.minimize(linear=[1], quadratic={("z", "z"): -1}, constant=1)
            expected = "minimize -z^2 + z + 1"
            self.assertEqual(str(q_p.objective), expected)
            self.assertEqual(repr(q_p.objective), f"<QuadraticObjective: {expected}>")


if __name__ == "__main__":
    unittest.main()
