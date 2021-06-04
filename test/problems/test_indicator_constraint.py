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

""" Test LinearConstraint """

import unittest
from test.optimization_test_case import QiskitOptimizationTestCase

from qiskit_optimization import QuadraticProgram, QiskitOptimizationError
from qiskit_optimization.problems.constraint import ConstraintSense


class TestIndicatorConstraint(QiskitOptimizationTestCase):
    """Test IndicatorConstraint."""

    def make_problem(self):
        """Make a test problem"""
        op = QuadraticProgram()
        for i in range(3):
            op.binary_var(name="x{}".format(i))
        op.binary_var(name="a")
        return op

    def validate_indicator_constraint(self, op):
        """Validate an indicator constraint"""
        i_const = op.indicator_constraints[0]
        self.assertEqual(i_const.active_value, 1)
        self.assertEqual(i_const.binary_var, op.variables[3])
        self.assertEqual({0: 1.0, 1: 1.0, 2: 1.0}, i_const.linear.to_dict())
        self.assertEqual(i_const.rhs, 1)
        self.assertEqual(i_const.sense, ConstraintSense.LE)
        self.assertEqual(i_const.name, "indicator_const")

    def test_init(self):
        """test init"""
        # use a name for binary_var
        op = self.make_problem()
        op.indicator_constraint("a", {"x0": 1, "x1": 1, "x2": 1}, "<=", 1, 1, "indicator_const")
        self.validate_indicator_constraint(op)
        # use an index for binary_var
        op = self.make_problem()
        op.indicator_constraint(3, {"x0": 1, "x1": 1, "x2": 1}, "<=", 1, 1, "indicator_const")
        self.validate_indicator_constraint(op)
        # use a Variable for binary_var
        op = self.make_problem()
        op.indicator_constraint(
            op.variables[3], {"x0": 1, "x1": 1, "x2": 1}, "<=", 1, 1, "indicator_const"
        )
        self.validate_indicator_constraint(op)
        # test for type check
        with self.assertRaises(QiskitOptimizationError):
            op = self.make_problem()
            # unsupported format for binary_var
            op.indicator_constraint(
                1.5, {"x0": 1, "x1": 1, "x2": 1}, "<=", 1, 1, "indicator_const_float"
            )
        # unsupported vartype for binary_var
        with self.assertRaises(QiskitOptimizationError):
            op = self.make_problem()
            op.integer_var(lowerbound=0, upperbound=3, name="i")
            op.indicator_constraint(
                "i", {"x0": 1, "x1": 1, "x2": 1}, "<=", 1, 1, "indicator_const_int_var"
            )
        # unsupported value for active_value
        with self.assertRaises(QiskitOptimizationError):
            op = self.make_problem()
            op.indicator_constraint(
                "a", {"x0": 1, "x1": 1, "x2": 1}, "<=", 1, 5, "indicator_const_wrong_active_val"
            )

    def test_active_value(self):
        """test active_value"""
        op = self.make_problem()
        op.indicator_constraint("a", {"x0": 1, "x1": 1, "x2": 1}, "<=", 1, 0, "indicator_const")
        self.assertEqual(op.indicator_constraints[0].active_value, 0)
        op.indicator_constraints[0].active_value = 1
        self.assertEqual(op.indicator_constraints[0].active_value, 1)

    def test_binary_var(self):
        """test binary_var"""
        op = self.make_problem()
        op.binary_var(name="b")
        op.indicator_constraint("b", {"x0": 1, "x1": 1, "x2": 1}, "<=", 1, 1, "indicator_const")
        self.assertEqual(op.indicator_constraints[0].binary_var, op.variables[4])
        op.indicator_constraints[0].binary_var = op.variables[3]
        self.assertEqual(op.indicator_constraints[0].binary_var, op.variables[3])

    def test_linear(self):
        """test binary_var"""
        op = self.make_problem()
        op.indicator_constraint("a", {"x0": 1, "x1": 1}, "<=", 1, 1, "indicator_const")
        self.assertEqual(op.indicator_constraints[0].linear.to_dict(), {0: 1, 1: 1})
        op.indicator_constraints[0].linear = {"x0": 1, "x1": 1, "x2": 1}
        self.assertEqual(op.indicator_constraints[0].linear.to_dict(), {0: 1, 1: 1, 2: 1})

    def test_evaluate(self):
        """test evaluate and evaluate_indicator"""
        op = self.make_problem()
        op.indicator_constraint("a", {"x0": 1, "x1": 1, "x2": 1}, "<=", 1, 1, "indicator_const")
        self.assertEqual(2, op.indicator_constraints[0].evaluate([1, 0, 1, 1]))
        self.assertEqual(True, op.indicator_constraints[0].evaluate_indicator([1, 0, 1, 1]))


if __name__ == "__main__":
    unittest.main()
