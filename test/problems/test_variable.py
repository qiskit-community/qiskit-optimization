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

"""Test Variable."""

import unittest
from test.optimization_test_case import QiskitOptimizationTestCase

from qiskit_optimization import INFINITY
from qiskit_optimization.problems import QuadraticProgram, Variable


class TestVariable(QiskitOptimizationTestCase):
    """Test Variable."""

    def test_init(self):
        """test init"""

        quadratic_program = QuadraticProgram()
        name = "variable"
        lowerbound = 0
        upperbound = 10
        vartype = Variable.Type.INTEGER

        variable = Variable(quadratic_program, name, lowerbound, upperbound, vartype)

        self.assertEqual(variable.name, name)
        self.assertEqual(variable.lowerbound, lowerbound)
        self.assertEqual(variable.upperbound, upperbound)
        self.assertEqual(variable.vartype, Variable.Type.INTEGER)

    def test_init_default(self):
        """test init with default values."""

        quadratic_program = QuadraticProgram()
        name = "variable"

        variable = Variable(quadratic_program, name)

        self.assertEqual(variable.name, name)
        self.assertEqual(variable.lowerbound, 0)
        self.assertEqual(variable.upperbound, INFINITY)
        self.assertEqual(variable.vartype, Variable.Type.CONTINUOUS)

    def test_str_repr(self):
        """test str and repr"""
        quadratic_program = QuadraticProgram()

        with self.subTest("binary"):
            bin_var = quadratic_program.binary_var(name="x")
            expected = "x (binary)"
            self.assertEqual(str(bin_var), expected)
            self.assertEqual(repr(bin_var), f"<Variable: {expected}>")

        with self.subTest("integer 1"):
            int_var = quadratic_program.integer_var(name="y1")
            expected = "0 <= y1 (integer)"
            self.assertEqual(str(int_var), expected)
            self.assertEqual(repr(int_var), f"<Variable: {expected}>")

        with self.subTest("integer 2"):
            int_var2 = quadratic_program.integer_var(lowerbound=5, upperbound=10, name="y2")
            expected = "5 <= y2 <= 10 (integer)"
            self.assertEqual(str(int_var2), expected)
            self.assertEqual(repr(int_var2), f"<Variable: {expected}>")

        with self.subTest("continuous 1"):
            con_var = quadratic_program.continuous_var(name="z1")
            expected = "0 <= z1 (continuous)"
            self.assertEqual(str(con_var), expected)
            self.assertEqual(repr(con_var), f"<Variable: {expected}>")

        with self.subTest("continuous 2"):
            con_var2 = quadratic_program.continuous_var(
                lowerbound=-10.0, upperbound=100.0, name="z2"
            )
            expected = "-10.0 <= z2 <= 100.0 (continuous)"
            self.assertEqual(str(con_var2), expected)
            self.assertEqual(repr(con_var2), f"<Variable: {expected}>")

        with self.subTest("continuous 3"):
            con_var3 = quadratic_program.continuous_var(
                lowerbound=-INFINITY, upperbound=100.0, name="z3"
            )
            expected = "z3 <= 100.0 (continuous)"
            self.assertEqual(str(con_var3), expected)
            self.assertEqual(repr(con_var3), f"<Variable: {expected}>")


if __name__ == "__main__":
    unittest.main()
