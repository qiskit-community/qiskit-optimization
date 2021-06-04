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

""" Test a IndicatorToInequality converter """
from test.optimization_test_case import QiskitOptimizationTestCase
import numpy as np
from qiskit_optimization import QuadraticProgram, QiskitOptimizationError
from qiskit_optimization.converters import IndicatorToInequality
from qiskit_optimization.problems.constraint import ConstraintSense


class TestIndicatorToInequality(QiskitOptimizationTestCase):
    """Test an IndicatorToInequality converter"""

    def test_convert_indicator_to_inequality(self):
        """Test convert method"""
        # for an LE constraint
        op = QuadraticProgram()
        for i in range(3):
            op.binary_var(name="x{}".format(i))
        op.binary_var(name="a")
        op.indicator_constraint(
            binary_var="a", linear={"x0": 1, "x1": 1, "x2": 1}, sense="<=", rhs=1, name="i_const"
        )
        indi2ineq = IndicatorToInequality()
        new_op = indi2ineq.convert(op)
        self.assertEqual(new_op.get_num_indicator_constraints(), 0)
        self.assertEqual(new_op.get_num_linear_constraints(), 1)
        l_const = new_op.linear_constraints[0]
        self.assertEqual({0: 1.0, 1: 1.0, 2: 1.0, 3: 2.0}, l_const.linear.to_dict())
        self.assertEqual(l_const.rhs, 3)
        self.assertEqual(l_const.sense, ConstraintSense.LE)
        self.assertEqual(l_const.name, "i_const@indicator")
        # for a GE constraint
        op = QuadraticProgram()
        for i in range(3):
            op.binary_var(name="x{}".format(i))
        op.binary_var(name="a")
        op.indicator_constraint(
            binary_var="a", linear={"x0": 1, "x1": 1, "x2": 1}, sense=">=", rhs=1, name="i_const"
        )
        indi2ineq = IndicatorToInequality()
        new_op = indi2ineq.convert(op)
        self.assertEqual(new_op.get_num_indicator_constraints(), 0)
        self.assertEqual(new_op.get_num_linear_constraints(), 1)
        l_const = new_op.linear_constraints[0]
        self.assertEqual({0: 1.0, 1: 1.0, 2: 1.0, 3: -1.0}, l_const.linear.to_dict())
        self.assertEqual(l_const.rhs, 0)
        self.assertEqual(l_const.sense, ConstraintSense.GE)
        self.assertEqual(l_const.name, "i_const@indicator")
        # for an EQ constraint
        op = QuadraticProgram()
        for i in range(3):
            op.binary_var(name="x{}".format(i))
        op.binary_var(name="a")
        op.indicator_constraint(
            binary_var="a", linear={"x0": 1, "x1": 1, "x2": 1}, sense="==", rhs=1, name="i_const"
        )
        indi2ineq = IndicatorToInequality()
        new_op = indi2ineq.convert(op)
        self.assertEqual(new_op.get_num_indicator_constraints(), 0)
        self.assertEqual(new_op.get_num_linear_constraints(), 2)
        l_const = new_op.linear_constraints[0]
        self.assertEqual({0: 1.0, 1: 1.0, 2: 1.0, 3: 2.0}, l_const.linear.to_dict())
        self.assertEqual(l_const.rhs, 3)
        self.assertEqual(l_const.sense, ConstraintSense.LE)
        self.assertEqual(l_const.name, "i_const@indicator_LE")
        l_const = new_op.linear_constraints[1]
        self.assertEqual({0: 1.0, 1: 1.0, 2: 1.0, 3: -1.0}, l_const.linear.to_dict())
        self.assertEqual(l_const.rhs, 0)
        self.assertEqual(l_const.sense, ConstraintSense.GE)
        self.assertEqual(l_const.name, "i_const@indicator_GE")

    def test_interpret(self):
        """Test interpret method"""
        op = QuadraticProgram()
        for i in range(3):
            op.binary_var(name="x{}".format(i))
        op.binary_var(name="a")
        op.indicator_constraint(
            binary_var="a", linear={"x0": 1, "x1": 1, "x2": 1}, sense=">=", rhs=1, name="i_const"
        )
        indi2ineq = IndicatorToInequality()
        _ = indi2ineq.convert(op)
        np.testing.assert_array_almost_equal([1, 0, 0, 1], indi2ineq.interpret([1, 0, 0, 1]))
        with self.assertRaises(QiskitOptimizationError):
            indi2ineq.interpret([1, 0, 0])
