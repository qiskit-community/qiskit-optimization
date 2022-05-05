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

""" Test LinearConstraint """

import unittest
from test.optimization_test_case import QiskitOptimizationTestCase

import numpy as np

from qiskit_optimization import QuadraticProgram, QiskitOptimizationError
from qiskit_optimization.problems import Constraint


class TestLinearConstraint(QiskitOptimizationTestCase):
    """Test LinearConstraint."""

    def setUp(self) -> None:
        super().setUp()
        self.quadratic_program = QuadraticProgram()
        self.quadratic_program.binary_var_list(3, name="x")
        self.quadratic_program.linear_constraint({"x0": 1, "x1": -2}, "<=", 1)
        self.quadratic_program.linear_constraint({"x0": 1, "x1": -2}, "<", 1)
        self.quadratic_program.linear_constraint({"x0": 1, "x1": -2}, "LE", 1)
        self.quadratic_program.linear_constraint({"x0": 1, "x1": -2}, "L", 1)
        self.quadratic_program.linear_constraint({"x0": -1, "x1": 2}, "==", 2)
        self.quadratic_program.linear_constraint({"x0": -1, "x1": 2}, "=", 2)
        self.quadratic_program.linear_constraint({"x0": -1, "x1": 2}, "EQ", 2)
        self.quadratic_program.linear_constraint({"x0": -1, "x1": 2}, "E", 2)
        self.quadratic_program.linear_constraint({"x1": 2, "x2": -1}, ">=", 3)
        self.quadratic_program.linear_constraint({"x1": 2, "x2": -1}, ">", 3)
        self.quadratic_program.linear_constraint({"x1": 2, "x2": -1}, "GE", 3)
        self.quadratic_program.linear_constraint({"x1": 2, "x2": -1}, "G", 3)

    def test_init(self):
        """test init."""

        quadratic_program = QuadraticProgram()
        for _ in range(5):
            quadratic_program.continuous_var()
        self.assertEqual(quadratic_program.get_num_linear_constraints(), 0)

        coefficients = np.array(range(5))

        # equality constraints
        quadratic_program.linear_constraint(sense="==")
        self.assertEqual(quadratic_program.get_num_linear_constraints(), 1)
        self.assertEqual(quadratic_program.linear_constraints[0].name, "c0")
        self.assertEqual(len(quadratic_program.linear_constraints[0].linear.to_dict()), 0)
        self.assertEqual(quadratic_program.linear_constraints[0].sense, Constraint.Sense.EQ)
        self.assertEqual(quadratic_program.linear_constraints[0].rhs, 0.0)
        self.assertEqual(
            quadratic_program.linear_constraints[0],
            quadratic_program.get_linear_constraint("c0"),
        )
        self.assertEqual(
            quadratic_program.linear_constraints[0],
            quadratic_program.get_linear_constraint(0),
        )

        with self.assertRaises(QiskitOptimizationError):
            quadratic_program.linear_constraint(name="c0")

        quadratic_program.linear_constraint(coefficients, "==", 1.0, "c1")
        self.assertEqual(quadratic_program.get_num_linear_constraints(), 2)
        self.assertEqual(quadratic_program.linear_constraints[1].name, "c1")
        self.assertTrue(
            (quadratic_program.linear_constraints[1].linear.to_array() == coefficients).all()
        )
        self.assertEqual(quadratic_program.linear_constraints[1].sense, Constraint.Sense.EQ)
        self.assertEqual(quadratic_program.linear_constraints[1].rhs, 1.0)
        self.assertEqual(
            quadratic_program.linear_constraints[1],
            quadratic_program.get_linear_constraint("c1"),
        )
        self.assertEqual(
            quadratic_program.linear_constraints[1],
            quadratic_program.get_linear_constraint(1),
        )

        # geq constraints
        quadratic_program.linear_constraint(sense=">=")
        self.assertEqual(quadratic_program.get_num_linear_constraints(), 3)
        self.assertEqual(quadratic_program.linear_constraints[2].name, "c2")
        self.assertEqual(len(quadratic_program.linear_constraints[2].linear.to_dict()), 0)
        self.assertEqual(quadratic_program.linear_constraints[2].sense, Constraint.Sense.GE)
        self.assertEqual(quadratic_program.linear_constraints[2].rhs, 0.0)
        self.assertEqual(
            quadratic_program.linear_constraints[2],
            quadratic_program.get_linear_constraint("c2"),
        )
        self.assertEqual(
            quadratic_program.linear_constraints[2],
            quadratic_program.get_linear_constraint(2),
        )

        with self.assertRaises(QiskitOptimizationError):
            quadratic_program.linear_constraint(name="c2", sense=">=")

        quadratic_program.linear_constraint(coefficients, ">=", 1.0, "c3")
        self.assertEqual(quadratic_program.get_num_linear_constraints(), 4)
        self.assertEqual(quadratic_program.linear_constraints[3].name, "c3")
        self.assertTrue(
            (quadratic_program.linear_constraints[3].linear.to_array() == coefficients).all()
        )
        self.assertEqual(quadratic_program.linear_constraints[3].sense, Constraint.Sense.GE)
        self.assertEqual(quadratic_program.linear_constraints[3].rhs, 1.0)
        self.assertEqual(
            quadratic_program.linear_constraints[3],
            quadratic_program.get_linear_constraint("c3"),
        )
        self.assertEqual(
            quadratic_program.linear_constraints[3],
            quadratic_program.get_linear_constraint(3),
        )

        # leq constraints
        quadratic_program.linear_constraint(sense="<=")
        self.assertEqual(quadratic_program.get_num_linear_constraints(), 5)
        self.assertEqual(quadratic_program.linear_constraints[4].name, "c4")
        self.assertEqual(len(quadratic_program.linear_constraints[4].linear.to_dict()), 0)
        self.assertEqual(quadratic_program.linear_constraints[4].sense, Constraint.Sense.LE)
        self.assertEqual(quadratic_program.linear_constraints[4].rhs, 0.0)
        self.assertEqual(
            quadratic_program.linear_constraints[4],
            quadratic_program.get_linear_constraint("c4"),
        )
        self.assertEqual(
            quadratic_program.linear_constraints[4],
            quadratic_program.get_linear_constraint(4),
        )

        with self.assertRaises(QiskitOptimizationError):
            quadratic_program.linear_constraint(name="c4", sense="<=")

        quadratic_program.linear_constraint(coefficients, "<=", 1.0, "c5")
        self.assertEqual(quadratic_program.get_num_linear_constraints(), 6)
        self.assertEqual(quadratic_program.linear_constraints[5].name, "c5")
        self.assertTrue(
            (quadratic_program.linear_constraints[5].linear.to_array() == coefficients).all()
        )
        self.assertEqual(quadratic_program.linear_constraints[5].sense, Constraint.Sense.LE)
        self.assertEqual(quadratic_program.linear_constraints[5].rhs, 1.0)
        self.assertEqual(
            quadratic_program.linear_constraints[5],
            quadratic_program.get_linear_constraint("c5"),
        )
        self.assertEqual(
            quadratic_program.linear_constraints[5],
            quadratic_program.get_linear_constraint(5),
        )

    def test_str(self):
        """Test str"""
        self.assertEqual(self.quadratic_program.get_num_linear_constraints(), 12)
        for i in range(0, 4):
            self.assertEqual(
                str(self.quadratic_program.get_linear_constraint(i)), f"x0 - 2*x1 <= 1 'c{i}'"
            )
        for i in range(4, 8):
            self.assertEqual(
                str(self.quadratic_program.get_linear_constraint(i)), f"-x0 + 2*x1 == 2 'c{i}'"
            )
        for i in range(8, 12):
            self.assertEqual(
                str(self.quadratic_program.get_linear_constraint(i)), f"2*x1 - x2 >= 3 'c{i}'"
            )

    def test_repr(self):
        """Test repr"""
        self.assertEqual(self.quadratic_program.get_num_linear_constraints(), 12)
        for i in range(0, 4):
            self.assertEqual(
                repr(self.quadratic_program.get_linear_constraint(i)),
                f"<LinearConstraint: x0 - 2*x1 <= 1 'c{i}'>",
            )
        for i in range(4, 8):
            self.assertEqual(
                repr(self.quadratic_program.get_linear_constraint(i)),
                f"<LinearConstraint: -x0 + 2*x1 == 2 'c{i}'>",
            )
        for i in range(8, 12):
            self.assertEqual(
                repr(self.quadratic_program.get_linear_constraint(i)),
                f"<LinearConstraint: 2*x1 - x2 >= 3 'c{i}'>",
            )


if __name__ == "__main__":
    unittest.main()
