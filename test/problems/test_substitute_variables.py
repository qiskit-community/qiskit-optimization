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

""" Test substitute_variables """

import unittest
from test.optimization_test_case import QiskitOptimizationTestCase
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.problems.substitute_variables import substitute_variables


class TestSubstituteVariables(QiskitOptimizationTestCase):
    """Test substitute_variables"""

    def test_substitute_variables(self):
        """test substitute variables"""
        q_p = QuadraticProgram("test")
        q_p.binary_var(name="x")
        q_p.integer_var(name="y", lowerbound=-2, upperbound=4)
        q_p.continuous_var(name="z", lowerbound=-1.5, upperbound=3.2)
        q_p.minimize(
            constant=1,
            linear={"x": 1, "y": 2},
            quadratic={("x", "y"): -1, ("z", "z"): 2},
        )
        q_p.linear_constraint({"x": 2, "z": -1}, "==", 1)
        q_p.quadratic_constraint({"x": 2, "z": -1}, {("y", "z"): 3}, "<=", -1)

        with self.subTest("x <- -1"):
            q_p2 = substitute_variables(q_p, constants={"x": -1})
            self.assertEqual(q_p2.status, QuadraticProgram.Status.INFEASIBLE)
            q_p2 = substitute_variables(q_p, constants={"y": -3})
            self.assertEqual(q_p2.status, QuadraticProgram.Status.INFEASIBLE)
            q_p2 = substitute_variables(q_p, constants={"x": 1, "z": 2})
            self.assertEqual(q_p2.status, QuadraticProgram.Status.INFEASIBLE)
            q_p2.clear()
            self.assertEqual(q_p2.status, QuadraticProgram.Status.VALID)

        with self.subTest("x <- 0"):
            q_p2 = substitute_variables(q_p, constants={"x": 0})
            self.assertEqual(q_p2.status, QuadraticProgram.Status.VALID)
            self.assertDictEqual(q_p2.objective.linear.to_dict(use_name=True), {"y": 2})
            self.assertDictEqual(q_p2.objective.quadratic.to_dict(use_name=True), {("z", "z"): 2})
            self.assertEqual(q_p2.objective.constant, 1)
            self.assertEqual(len(q_p2.linear_constraints), 1)
            self.assertEqual(len(q_p2.quadratic_constraints), 1)

            cst = q_p2.linear_constraints[0]
            self.assertDictEqual(cst.linear.to_dict(use_name=True), {"z": -1})
            self.assertEqual(cst.sense.name, "EQ")
            self.assertEqual(cst.rhs, 1)

            cst = q_p2.quadratic_constraints[0]
            self.assertDictEqual(cst.linear.to_dict(use_name=True), {"z": -1})
            self.assertDictEqual(cst.quadratic.to_dict(use_name=True), {("y", "z"): 3})
            self.assertEqual(cst.sense.name, "LE")
            self.assertEqual(cst.rhs, -1)

        with self.subTest("z <- -1"):
            q_p2 = substitute_variables(q_p, constants={"z": -1})
            self.assertEqual(q_p2.status, QuadraticProgram.Status.VALID)
            self.assertDictEqual(q_p2.objective.linear.to_dict(use_name=True), {"x": 1, "y": 2})
            self.assertDictEqual(q_p2.objective.quadratic.to_dict(use_name=True), {("x", "y"): -1})
            self.assertEqual(q_p2.objective.constant, 3)
            self.assertEqual(len(q_p2.linear_constraints), 2)
            self.assertEqual(len(q_p2.quadratic_constraints), 0)

            cst = q_p2.linear_constraints[0]
            self.assertDictEqual(cst.linear.to_dict(use_name=True), {"x": 2})
            self.assertEqual(cst.sense.name, "EQ")
            self.assertEqual(cst.rhs, 0)

            cst = q_p2.linear_constraints[1]
            self.assertDictEqual(cst.linear.to_dict(use_name=True), {"x": 2, "y": -3})
            self.assertEqual(cst.sense.name, "LE")
            self.assertEqual(cst.rhs, -2)

        with self.subTest("y <- -0.5 * x"):
            q_p2 = substitute_variables(q_p, variables={"y": ("x", -0.5)})
            self.assertEqual(q_p2.status, QuadraticProgram.Status.VALID)
            self.assertDictEqual(q_p2.objective.linear.to_dict(use_name=True), {})
            self.assertDictEqual(
                q_p2.objective.quadratic.to_dict(use_name=True),
                {("x", "x"): 0.5, ("z", "z"): 2},
            )
            self.assertEqual(q_p2.objective.constant, 1)
            self.assertEqual(len(q_p2.linear_constraints), 1)
            self.assertEqual(len(q_p2.quadratic_constraints), 1)

            cst = q_p2.linear_constraints[0]
            self.assertDictEqual(cst.linear.to_dict(use_name=True), {"x": 2, "z": -1})
            self.assertEqual(cst.sense.name, "EQ")
            self.assertEqual(cst.rhs, 1)

            cst = q_p2.quadratic_constraints[0]
            self.assertDictEqual(cst.linear.to_dict(use_name=True), {"x": 2, "z": -1})
            self.assertDictEqual(cst.quadratic.to_dict(use_name=True), {("x", "z"): -1.5})
            self.assertEqual(cst.sense.name, "LE")
            self.assertEqual(cst.rhs, -1)


if __name__ == "__main__":
    unittest.main()
