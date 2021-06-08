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

"""Flip problem sense tests."""
from test import QiskitOptimizationTestCase

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import MaximizeToMinimize, MinimizeToMaximize


class TestFlipProblemSense(QiskitOptimizationTestCase):
    """Tests various flips of problem sense."""

    def test_maximize_to_minimize(self):
        """Test maximization to minimization conversion."""
        op_max = QuadraticProgram()
        op_min = QuadraticProgram()
        for i in range(2):
            op_max.binary_var(name="x{}".format(i))
            op_min.binary_var(name="x{}".format(i))
        op_max.integer_var(name="x{}".format(2), lowerbound=-3, upperbound=3)
        op_min.integer_var(name="x{}".format(2), lowerbound=-3, upperbound=3)
        op_max.maximize(constant=3, linear={"x0": 1}, quadratic={("x1", "x2"): 2})
        op_min.minimize(constant=3, linear={"x0": 1}, quadratic={("x1", "x2"): 2})

        # check conversion of maximization problem
        conv = MaximizeToMinimize()
        op_conv = conv.convert(op_max)
        self.assertEqual(op_conv.objective.sense, op_conv.objective.Sense.MINIMIZE)
        x = [0, 1, 2]
        fval_min = op_conv.objective.evaluate(conv.interpret(x))
        self.assertAlmostEqual(fval_min, -7)
        self.assertAlmostEqual(op_max.objective.evaluate(x), -fval_min)

        # check conversion of minimization problem
        op_conv = conv.convert(op_min)
        self.assertEqual(op_conv.objective.sense, op_min.objective.sense)
        fval_min = op_conv.objective.evaluate(conv.interpret(x))
        self.assertAlmostEqual(op_min.objective.evaluate(x), fval_min)

    def test_minimize_to_maximize(self):
        """Test minimization to maximization conversion."""
        op_max = QuadraticProgram()
        op_min = QuadraticProgram()
        for i in range(2):
            op_max.binary_var(name="x{}".format(i))
            op_min.binary_var(name="x{}".format(i))
        op_max.integer_var(name="x{}".format(2), lowerbound=-3, upperbound=3)
        op_min.integer_var(name="x{}".format(2), lowerbound=-3, upperbound=3)
        op_max.maximize(constant=3, linear={"x0": 1}, quadratic={("x1", "x2"): 2})
        op_min.minimize(constant=3, linear={"x0": 1}, quadratic={("x1", "x2"): 2})

        # check conversion of maximization problem
        conv = MinimizeToMaximize()
        op_conv = conv.convert(op_min)
        self.assertEqual(op_conv.objective.sense, op_conv.objective.Sense.MAXIMIZE)
        x = [0, 1, 2]
        fval_max = op_conv.objective.evaluate(conv.interpret(x))
        self.assertAlmostEqual(fval_max, -7)
        self.assertAlmostEqual(op_max.objective.evaluate(x), -fval_max)

        # check conversion of maximization problem
        op_conv = conv.convert(op_max)
        self.assertEqual(op_conv.objective.sense, op_max.objective.sense)
        fval_max = op_conv.objective.evaluate(conv.interpret(x))
        self.assertAlmostEqual(op_min.objective.evaluate(x), fval_max)
