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

""" Test LinearInequalityToPenalty converter """

import unittest
from test.optimization_test_case import QiskitOptimizationTestCase

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import LinearInequalityToPenalty, QuadraticProgramToQubo
from qiskit_optimization.problems import Constraint


class TestLinearInequalityToPenalty(QiskitOptimizationTestCase):
    """Test Converters"""

    def test_linear_inequality_to_penalty1(self):
        """Test special constraint to penalty x+y <= 1 -> P(x*y)"""

        op = QuadraticProgram()
        lip = LinearInequalityToPenalty()

        op.binary_var(name="x")
        op.binary_var(name="y")
        # Linear constraints
        linear_constraint = {"x": 1, "y": 1}
        op.linear_constraint(linear_constraint, Constraint.Sense.LE, 1, "P(xy)")

        # Test with no max/min
        with self.subTest("No max/min"):
            self.assertEqual(op.get_num_linear_constraints(), 1)
            lip.penalty = 1
            quadratic = {("x", "y"): lip.penalty}
            op2 = lip.convert(op)
            qdct = op2.objective.quadratic.to_dict(use_name=True)
            self.assertEqual(qdct, quadratic)
            self.assertEqual(op2.get_num_linear_constraints(), 0)

        # Test maximize
        with self.subTest("Maximize"):
            linear = {"x": 2, "y": 1}
            op.maximize(linear=linear)
            lip.penalty = 5
            quadratic = {("x", "y"): -1 * lip.penalty}
            op2 = lip.convert(op)
            ldct = op2.objective.linear.to_dict(use_name=True)
            qdct = op2.objective.quadratic.to_dict(use_name=True)
            self.assertEqual(ldct, linear)
            self.assertEqual(qdct, quadratic)
            self.assertEqual(op2.get_num_linear_constraints(), 0)

        # Test minimize
        with self.subTest("Minimize"):
            linear = {"x": 2, "y": 1}
            op.minimize(linear=linear)
            lip.penalty = 5
            quadratic = {("x", "y"): lip.penalty}
            op2 = lip.convert(op)
            ldct = op2.objective.linear.to_dict(use_name=True)
            qdct = op2.objective.quadratic.to_dict(use_name=True)
            self.assertEqual(ldct, linear)
            self.assertEqual(qdct, quadratic)
            self.assertEqual(op2.get_num_linear_constraints(), 0)

        # Test combination
        with self.subTest("Combination"):
            op = QuadraticProgram()
            lip = LinearInequalityToPenalty()

            op.binary_var(name="x")
            op.binary_var(name="y")
            op.binary_var(name="z")
            op.binary_var(name="w")
            linear = {"x": 2, "y": 1, "z": -1, "w": 1}
            quadratic = {("y", "z"): -2, ("w", "w"): 1}
            op.minimize(linear=linear, quadratic=quadratic)

            linear_constraint = {"x": 1, "w": 1}
            op.linear_constraint(linear_constraint, Constraint.Sense.LE, 1, "P(xw)")
            linear_constraint = {"y": 1, "z": 1}
            op.linear_constraint(linear_constraint, Constraint.Sense.LE, 1, "P(yz)")
            linear_constraint = {"y": 2, "z": 1}
            op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 1, "None 1")
            quadratic_constraint = {("x", "x"): -2, ("y", "w"): 1}
            op.quadratic_constraint(
                linear_constraint, quadratic_constraint, Constraint.Sense.LE, 1, "None 2"
            )

            lip.penalty = 5
            op2 = lip.convert(op)
            quadratic[("x", "w")] = lip.penalty
            quadratic[("y", "z")] = quadratic[("y", "z")] + lip.penalty
            ldct = op2.objective.linear.to_dict(use_name=True)
            qdct = op2.objective.quadratic.to_dict(use_name=True)
            self.assertEqual(ldct, linear)
            self.assertEqual(qdct, quadratic)
            self.assertEqual(op2.get_num_linear_constraints(), 1)
            self.assertEqual(op2.get_num_quadratic_constraints(), 1)

    def test_linear_inequality_to_penalty2(self):
        """Test special constraint to penalty x+y >= 1 -> P(1-x-y+x*y)"""

        op = QuadraticProgram()
        lip = LinearInequalityToPenalty()

        op.binary_var(name="x")
        op.binary_var(name="y")
        # Linear constraints
        linear_constraint = {"x": 1, "y": 1}
        op.linear_constraint(linear_constraint, Constraint.Sense.GE, 1, "P(1-x-y+xy)")

        # Test with no max/min
        with self.subTest("No max/min"):
            self.assertEqual(op.get_num_linear_constraints(), 1)
            lip.penalty = 1
            constant = 1
            linear = {"x": -1 * lip.penalty, "y": -1 * lip.penalty}
            quadratic = {("x", "y"): lip.penalty}
            op2 = lip.convert(op)
            cnst = op2.objective.constant
            ldct = op2.objective.linear.to_dict(use_name=True)
            qdct = op2.objective.quadratic.to_dict(use_name=True)
            self.assertEqual(cnst, constant)
            self.assertEqual(ldct, linear)
            self.assertEqual(qdct, quadratic)
            self.assertEqual(op2.get_num_linear_constraints(), 0)

        # Test maximize
        with self.subTest("Maximize"):
            linear = {"x": 2, "y": 1}
            op.maximize(linear=linear)
            lip.penalty = 5
            constant = -1 * lip.penalty
            linear["x"] = linear["x"] + lip.penalty
            linear["y"] = linear["y"] + lip.penalty
            quadratic = {("x", "y"): -1 * lip.penalty}
            op2 = lip.convert(op)
            cnst = op2.objective.constant
            ldct = op2.objective.linear.to_dict(use_name=True)
            qdct = op2.objective.quadratic.to_dict(use_name=True)
            self.assertEqual(cnst, constant)
            self.assertEqual(ldct, linear)
            self.assertEqual(qdct, quadratic)
            self.assertEqual(op2.get_num_linear_constraints(), 0)

        # Test minimize
        with self.subTest("Minimize"):
            linear = {"x": 2, "y": 1}
            op.minimize(linear=linear)
            lip.penalty = 5
            constant = lip.penalty
            linear["x"] = linear["x"] - lip.penalty
            linear["y"] = linear["y"] - lip.penalty
            quadratic = {("x", "y"): lip.penalty}
            op2 = lip.convert(op)
            cnst = op2.objective.constant
            ldct = op2.objective.linear.to_dict(use_name=True)
            qdct = op2.objective.quadratic.to_dict(use_name=True)
            self.assertEqual(cnst, constant)
            self.assertEqual(ldct, linear)
            self.assertEqual(qdct, quadratic)
            self.assertEqual(op2.get_num_linear_constraints(), 0)

        # Test combination
        with self.subTest("Combination"):
            op = QuadraticProgram()
            lip = LinearInequalityToPenalty()

            op.binary_var(name="x")
            op.binary_var(name="y")
            op.binary_var(name="z")
            op.binary_var(name="w")
            linear = {"x": 2, "y": 1, "z": -1, "w": 1}
            quadratic = {("y", "z"): -2, ("w", "w"): 1}
            op.minimize(linear=linear, quadratic=quadratic)

            linear_constraint = {"x": 1, "w": 1}
            op.linear_constraint(linear_constraint, Constraint.Sense.GE, 1, "P(1-x-w+xw)")
            linear_constraint = {"y": 1, "z": 1}
            op.linear_constraint(linear_constraint, Constraint.Sense.GE, 1, "P(1-y-z+yz)")
            linear_constraint = {"y": 2, "z": 1}
            op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 1, "None 1")
            quadratic_constraint = {("x", "x"): -2, ("y", "w"): 1}
            op.quadratic_constraint(
                linear_constraint, quadratic_constraint, Constraint.Sense.LE, 1, "None 2"
            )

            lip.penalty = 5
            op2 = lip.convert(op)
            linear["x"] = linear["x"] - lip.penalty
            linear["y"] = linear["y"] - lip.penalty
            linear["z"] = linear["z"] - lip.penalty
            linear["w"] = linear["w"] - lip.penalty
            quadratic[("x", "w")] = lip.penalty
            quadratic[("y", "z")] = quadratic[("y", "z")] + lip.penalty
            constant = lip.penalty
            ldct = op2.objective.linear.to_dict(use_name=True)
            qdct = op2.objective.quadratic.to_dict(use_name=True)
            self.assertEqual(cnst, constant)
            self.assertEqual(ldct, linear)
            self.assertEqual(qdct, quadratic)
            self.assertEqual(op2.get_num_linear_constraints(), 1)
            self.assertEqual(op2.get_num_quadratic_constraints(), 1)

    def test_linear_inequality_to_penalty3(self):
        """Test special constraint to penalty x1+x2+x3+... >= n-1 -> P(x1*x2+x1*x3+...)"""

        op = QuadraticProgram()
        lip = LinearInequalityToPenalty()
        op.binary_var_list(5)

        # Linear constraints
        n = 5
        op.linear_constraint([1, 1, 1, 1, 1], Constraint.Sense.GE, n - 1, "")

        # Test with no max/min
        with self.subTest("No max/min"):
            self.assertEqual(op.get_num_linear_constraints(), 1)
            penalty = 5
            lip.penalty = penalty

            constant = 10
            linear = [n - 1, n - 1, n - 1, n - 1, n - 1]
            quadratic = [
                [0, 1, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
            ]
            op2 = lip.convert(op)
            cnst2 = op2.objective.constant / penalty
            ldct2 = op2.objective.linear.to_array() / penalty * -1
            qdct2 = op2.objective.quadratic.to_array() / penalty
            self.assertEqual(cnst2, constant)
            self.assertEqual(ldct2.tolist(), linear)
            self.assertEqual(qdct2.tolist(), quadratic)
            self.assertEqual(op2.get_num_linear_constraints(), 0)

    def test_linear_inequality_to_penalty4(self):
        """Test special constraint to penalty x1+x2+x3+... <= 1 -> P(x1*x2+x1*x3+...)"""

        op = QuadraticProgram()
        lip = LinearInequalityToPenalty()

        op.binary_var(name="x")
        op.binary_var(name="y")
        op.binary_var(name="z")
        # Linear constraints
        linear_constraint = {"x": 1, "y": 1, "z": 1}
        op.linear_constraint(linear_constraint, Constraint.Sense.LE, 1, "P(xy+yz+zx)")

        # Test with no max/min
        with self.subTest("No max/min"):
            self.assertEqual(op.get_num_linear_constraints(), 1)
            penalty = 1
            quadratic = {("x", "y"): penalty, ("x", "z"): penalty, ("y", "z"): penalty}
            op2 = lip.convert(op)
            qdct = op2.objective.quadratic.to_dict(use_name=True)
            self.assertEqual(qdct, quadratic)
            self.assertEqual(op2.get_num_linear_constraints(), 0)

        # Test maximize
        op = QuadraticProgram()
        op.binary_var_list(5)
        linear2 = [1, 1, 0, 0, 0]
        op.maximize(linear=linear2)
        op.linear_constraint([1, 1, 1, 1, 1], Constraint.Sense.LE, 1, "")

        with self.subTest("Maximum"):
            self.assertEqual(op.get_num_linear_constraints(), 1)
            lip.penalty = 5
            quadratic2 = [
                [0, 1, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
            ]
            op2 = lip.convert(op)
            ldct2 = op2.objective.linear.to_array()
            qdct2 = op2.objective.quadratic.to_array() / lip.penalty * -1
            self.assertEqual(ldct2.tolist(), linear2)
            self.assertEqual(qdct2.tolist(), quadratic2)
            self.assertEqual(op2.get_num_linear_constraints(), 0)

    def test_linear_inequality_to_penalty6(self):
        """Test special constraint to penalty 6 x-y <= 0 -> P(x-x*y)"""

        op = QuadraticProgram()
        lip = LinearInequalityToPenalty()

        op.binary_var(name="x")
        op.binary_var(name="y")
        # Linear constraints
        linear_constraint = {"x": 1, "y": -1}
        op.linear_constraint(linear_constraint, Constraint.Sense.LE, 0, "P(x-xy)")

        # Test with no max/min
        with self.subTest("No max/min"):
            self.assertEqual(op.get_num_linear_constraints(), 1)
            penalty = 1
            linear = {"x": penalty}
            quadratic = {("x", "y"): -1 * penalty}
            op2 = lip.convert(op)
            ldct = op2.objective.linear.to_dict(use_name=True)
            qdct = op2.objective.quadratic.to_dict(use_name=True)
            self.assertEqual(ldct, linear)
            self.assertEqual(qdct, quadratic)
            self.assertEqual(op2.get_num_linear_constraints(), 0)

        # Test maximize
        with self.subTest("Maximize"):
            linear = {"x": 2, "y": 1}
            op.maximize(linear=linear)
            penalty = 4
            linear["x"] = linear["x"] - penalty
            quadratic = {("x", "y"): penalty}
            op2 = lip.convert(op)
            ldct = op2.objective.linear.to_dict(use_name=True)
            qdct = op2.objective.quadratic.to_dict(use_name=True)
            self.assertEqual(ldct, linear)
            self.assertEqual(qdct, quadratic)
            self.assertEqual(op2.get_num_linear_constraints(), 0)

        # Test minimize
        with self.subTest("Minimize"):
            linear = {"x": 2, "y": 1}
            op.minimize(linear={"x": 2, "y": 1})
            penalty = 4
            linear["x"] = linear["x"] + penalty
            quadratic = {("x", "y"): -1 * penalty}
            op2 = lip.convert(op)
            ldct = op2.objective.linear.to_dict(use_name=True)
            qdct = op2.objective.quadratic.to_dict(use_name=True)
            self.assertEqual(ldct, linear)
            self.assertEqual(qdct, quadratic)
            self.assertEqual(op2.get_num_linear_constraints(), 0)

    def test_linear_inequality_to_penalty7(self):
        """Test special constraint to penalty 6 x-y >= 0 -> P(y-x*y)"""

        op = QuadraticProgram()
        lip = LinearInequalityToPenalty()

        op.binary_var(name="x")
        op.binary_var(name="y")
        # Linear constraints
        linear_constraint = {"x": 1, "y": -1}
        op.linear_constraint(linear_constraint, Constraint.Sense.GE, 0, "P(y-xy)")

        # Test with no max/min
        with self.subTest("No max/min"):
            self.assertEqual(op.get_num_linear_constraints(), 1)
            penalty = 1
            linear = {"y": penalty}
            quadratic = {("x", "y"): -1 * penalty}
            op2 = lip.convert(op)
            ldct = op2.objective.linear.to_dict(use_name=True)
            qdct = op2.objective.quadratic.to_dict(use_name=True)
            self.assertEqual(ldct, linear)
            self.assertEqual(qdct, quadratic)
            self.assertEqual(op2.get_num_linear_constraints(), 0)

        # Test maximize
        with self.subTest("Maximize"):
            linear = {"x": 2, "y": 1}
            op.maximize(linear=linear)
            penalty = 4
            linear["y"] = linear["y"] - penalty
            quadratic = {("x", "y"): penalty}
            op2 = lip.convert(op)
            ldct = op2.objective.linear.to_dict(use_name=True)
            qdct = op2.objective.quadratic.to_dict(use_name=True)
            self.assertEqual(ldct, linear)
            self.assertEqual(qdct, quadratic)
            self.assertEqual(op2.get_num_linear_constraints(), 0)

        # Test minimize
        with self.subTest("Minimize"):
            linear = {"x": 2, "y": 1}
            op.minimize(linear={"x": 2, "y": 1})
            penalty = 4
            linear["y"] = linear["y"] + penalty
            quadratic = {("x", "y"): -1 * penalty}
            op2 = lip.convert(op)
            ldct = op2.objective.linear.to_dict(use_name=True)
            qdct = op2.objective.quadratic.to_dict(use_name=True)
            self.assertEqual(ldct, linear)
            self.assertEqual(qdct, quadratic)
            self.assertEqual(op2.get_num_linear_constraints(), 0)

    def test_quadratic_program_to_qubo_inequality_to_penalty(self):
        """Test QuadraticProgramToQubo, passing inequality pattern"""

        op = QuadraticProgram()
        conv = QuadraticProgramToQubo()
        op.binary_var(name="x")
        op.binary_var(name="y")

        # Linear constraints
        linear_constraint = {"x": 1, "y": 1}
        op.linear_constraint(linear_constraint, Constraint.Sense.GE, 1, "P(1-x-y+xy)")

        conv.penalty = 1
        constant = 1
        linear = {"x": -conv.penalty, "y": -conv.penalty}
        quadratic = {("x", "y"): conv.penalty}
        op2 = conv.convert(op)
        cnst = op2.objective.constant
        ldct = op2.objective.linear.to_dict(use_name=True)
        qdct = op2.objective.quadratic.to_dict(use_name=True)
        self.assertEqual(cnst, constant)
        self.assertEqual(ldct, linear)
        self.assertEqual(qdct, quadratic)
        self.assertEqual(op2.get_num_linear_constraints(), 0)

    def test_inequality_to_penalty_auto_define_penalty(self):
        """Test _auto_define_penalty() in InequalityToPenalty"""
        op = QuadraticProgram()
        op.integer_var(name="x", lowerbound=1, upperbound=3)
        op.integer_var(name="y", lowerbound=-1, upperbound=4)
        op.integer_var(name="z", lowerbound=-5, upperbound=-1)
        op.maximize(linear={"x": 1, "y": 1, "z": 1})
        lip = LinearInequalityToPenalty()
        self.assertEqual(lip._auto_define_penalty(op), 12)
        op = QuadraticProgram()
        op.integer_var(name="x", lowerbound=1, upperbound=3)
        op.integer_var(name="y", lowerbound=-1, upperbound=4)
        op.integer_var(name="z", lowerbound=-5, upperbound=-1)
        op.maximize(linear={"x": -1, "y": -1, "z": -1})
        lip = LinearInequalityToPenalty()
        self.assertEqual(lip._auto_define_penalty(op), 12)
        op = QuadraticProgram()
        op.integer_var(name="x", lowerbound=1, upperbound=3)
        op.integer_var(name="y", lowerbound=-1, upperbound=4)
        op.integer_var(name="z", lowerbound=-5, upperbound=-1)
        op.maximize(quadratic={(0, 0): 1, (0, 1): 1, (0, 2): 1, (1, 1): 1, (1, 2): 1, (2, 2): 1})
        lip = LinearInequalityToPenalty()
        self.assertEqual(lip._auto_define_penalty(op), 103)
        op = QuadraticProgram()
        op.integer_var(name="x", lowerbound=1, upperbound=3)
        op.integer_var(name="y", lowerbound=-1, upperbound=4)
        op.integer_var(name="z", lowerbound=-5, upperbound=-1)
        op.maximize(
            quadratic={(0, 0): -1, (0, 1): -1, (0, 2): -1, (1, 1): -1, (1, 2): -1, (2, 2): -1}
        )
        lip = LinearInequalityToPenalty()
        self.assertEqual(lip._auto_define_penalty(op), 103)
        op = QuadraticProgram()
        op.integer_var(lowerbound=-2, upperbound=1, name="x")
        op.minimize(quadratic={("x", "x"): 1})
        lip = LinearInequalityToPenalty()
        self.assertEqual(lip._auto_define_penalty(op), 5)

    def test_linear_inequality_to_penalty8(self):
        """Test combinations of inequality constraints"""

        with self.subTest("minimize 1"):
            op = QuadraticProgram()
            op.binary_var("x")
            op.binary_var("y")
            op.binary_var("z")
            op.integer_var(-1, 4, "q")
            op.minimize(linear={"x": 1, "y": 1, "z": 1}, quadratic={("q", "q"): -1})
            op.linear_constraint({"x": 1, "y": -1}, "<=", 0)
            op.linear_constraint({"x": 1, "y": 1, "z": 1}, "<=", 1)
            op2 = LinearInequalityToPenalty().convert(op)
            self.assertEqual(op2.get_num_vars(), 4)
            self.assertEqual(op2.get_num_binary_vars(), 3)
            self.assertEqual(op2.get_num_integer_vars(), 1)
            self.assertEqual(op2.get_num_continuous_vars(), 0)
            self.assertEqual(op2.get_num_linear_constraints(), 0)
            self.assertEqual(op2.get_num_quadratic_constraints(), 0)
            obj = op2.objective
            self.assertEqual(obj.constant, 0)
            self.assertDictEqual(obj.linear.to_dict(use_name=True), {"x": 21, "y": 1, "z": 1})
            self.assertDictEqual(
                obj.quadratic.to_dict(use_name=True),
                {("x", "z"): 20, ("y", "z"): 20, ("q", "q"): -1},
            )

        with self.subTest("maximize 1"):
            op = QuadraticProgram()
            op.binary_var("x")
            op.binary_var("y")
            op.binary_var("z")
            op.integer_var(-1, 4, "q")
            op.maximize(linear={"x": 1, "y": 1, "z": 1}, quadratic={("q", "q"): -1})
            op.linear_constraint({"x": 1, "y": -1}, "<=", 0)
            op.linear_constraint({"x": 1, "y": 1, "z": 1}, "<=", 1)
            op2 = LinearInequalityToPenalty().convert(op)
            self.assertEqual(op2.get_num_vars(), 4)
            self.assertEqual(op2.get_num_binary_vars(), 3)
            self.assertEqual(op2.get_num_integer_vars(), 1)
            self.assertEqual(op2.get_num_continuous_vars(), 0)
            self.assertEqual(op2.get_num_linear_constraints(), 0)
            self.assertEqual(op2.get_num_quadratic_constraints(), 0)
            obj = op2.objective
            self.assertEqual(obj.constant, 0)
            self.assertDictEqual(obj.linear.to_dict(use_name=True), {"x": -19, "y": 1, "z": 1})
            self.assertDictEqual(
                obj.quadratic.to_dict(use_name=True),
                {("x", "z"): -20, ("y", "z"): -20, ("q", "q"): -1},
            )

        with self.subTest("minimize 2"):
            op = QuadraticProgram()
            op.binary_var("x")
            op.binary_var("y")
            op.binary_var("z")
            op.integer_var(-1, 4, "q")
            op.minimize(linear={"x": 1, "y": 1, "z": 1}, quadratic={("q", "q"): -1})
            op.linear_constraint({"x": 1, "y": -1}, ">=", 0)
            op.linear_constraint({"x": 1, "y": 1, "z": 1}, ">=", 2)
            op2 = LinearInequalityToPenalty().convert(op)
            self.assertEqual(op2.get_num_vars(), 4)
            self.assertEqual(op2.get_num_binary_vars(), 3)
            self.assertEqual(op2.get_num_integer_vars(), 1)
            self.assertEqual(op2.get_num_continuous_vars(), 0)
            self.assertEqual(op2.get_num_linear_constraints(), 0)
            self.assertEqual(op2.get_num_quadratic_constraints(), 0)
            obj = op2.objective
            self.assertEqual(obj.constant, 60)
            self.assertDictEqual(obj.linear.to_dict(use_name=True), {"x": -39, "y": -19, "z": -39})
            self.assertDictEqual(
                obj.quadratic.to_dict(use_name=True),
                {("x", "z"): 20, ("y", "z"): 20, ("q", "q"): -1},
            )

        with self.subTest("maximize 2"):
            op = QuadraticProgram()
            op.binary_var("x")
            op.binary_var("y")
            op.binary_var("z")
            op.integer_var(-1, 4, "q")
            op.maximize(linear={"x": 1, "y": 1, "z": 1}, quadratic={("q", "q"): -1})
            op.linear_constraint({"x": 1, "y": -1}, ">=", 0)
            op.linear_constraint({"x": 1, "y": 1, "z": 1}, ">=", 2)
            op2 = LinearInequalityToPenalty().convert(op)
            self.assertEqual(op2.get_num_vars(), 4)
            self.assertEqual(op2.get_num_binary_vars(), 3)
            self.assertEqual(op2.get_num_integer_vars(), 1)
            self.assertEqual(op2.get_num_continuous_vars(), 0)
            self.assertEqual(op2.get_num_linear_constraints(), 0)
            self.assertEqual(op2.get_num_quadratic_constraints(), 0)
            obj = op2.objective
            self.assertEqual(obj.constant, -60)
            self.assertDictEqual(obj.linear.to_dict(use_name=True), {"x": 41, "y": 21, "z": 41})
            self.assertDictEqual(
                obj.quadratic.to_dict(use_name=True),
                {("x", "z"): -20, ("y", "z"): -20, ("q", "q"): -1},
            )


if __name__ == "__main__":
    unittest.main()
