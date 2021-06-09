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

"""Test from_docplex_mp and to_docplex_mp"""

from test.optimization_test_case import QiskitOptimizationTestCase

from docplex.mp.model import Model

from qiskit_optimization.exceptions import QiskitOptimizationError
from qiskit_optimization.problems import Constraint, QuadraticProgram
from qiskit_optimization.translators.docplex_mp import from_docplex_mp, to_docplex_mp


class TestDocplexMpTranslator(QiskitOptimizationTestCase):
    """Test from_docplex_mp and to_docplex_mp"""

    def test_from_and_to(self):
        """test from_docplex_mp and to_docplex_mp"""
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
        q_p.quadratic_constraint({"x": 2, "z": -1}, {("y", "z"): 3}, "==", 1)
        q_p2 = from_docplex_mp(to_docplex_mp(q_p))
        self.assertEqual(q_p.export_as_lp_string(), q_p2.export_as_lp_string())

        mod = Model("test")
        x = mod.binary_var("x")
        y = mod.integer_var(-2, 4, "y")
        z = mod.continuous_var(-1.5, 3.2, "z")
        mod.minimize(1 + x + 2 * y - x * y + 2 * z * z)
        mod.add(2 * x - z == 1, "c0")
        mod.add(2 * x - z + 3 * y * z == 1, "q0")
        self.assertEqual(q_p.export_as_lp_string(), mod.export_as_lp_string())

        with self.assertRaises(QiskitOptimizationError):
            mod = Model()
            mod.semiinteger_var(lb=1, name="x")
            _ = from_docplex_mp(mod)

        with self.assertRaises(QiskitOptimizationError):
            mod = Model()
            x = mod.binary_var("x")
            mod.add_range(0, 2 * x, 1)
            _ = from_docplex_mp(mod)

        with self.assertRaises(QiskitOptimizationError):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.binary_var("y")
            mod.add_indicator(x, x + y <= 1, 1)
            _ = from_docplex_mp(mod)

        with self.assertRaises(QiskitOptimizationError):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.binary_var("y")
            mod.add_equivalence(x, x + y <= 1, 1)
            _ = from_docplex_mp(mod)

        with self.assertRaises(QiskitOptimizationError):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.binary_var("y")
            mod.add(mod.not_equal_constraint(x, y + 1))
            _ = from_docplex_mp(mod)

        # test from_docplex without explicit variable names
        mod = Model()
        x = mod.binary_var()
        y = mod.continuous_var()
        z = mod.integer_var()
        mod.minimize(x + y + z + x * y + y * z + x * z)
        mod.add_constraint(x + y == z)  # linear EQ
        mod.add_constraint(x + y >= z)  # linear GE
        mod.add_constraint(x + y <= z)  # linear LE
        mod.add_constraint(x * y == z)  # quadratic EQ
        mod.add_constraint(x * y >= z)  # quadratic GE
        mod.add_constraint(x * y <= z)  # quadratic LE
        q_p = from_docplex_mp(mod)
        var_names = [v.name for v in q_p.variables]
        self.assertListEqual(var_names, ["x0", "x1", "x2"])
        senses = [Constraint.Sense.EQ, Constraint.Sense.GE, Constraint.Sense.LE]
        for i, c in enumerate(q_p.linear_constraints):
            self.assertDictEqual(c.linear.to_dict(use_name=True), {"x0": 1, "x1": 1, "x2": -1})
            self.assertEqual(c.rhs, 0)
            self.assertEqual(c.sense, senses[i])
        for i, c in enumerate(q_p.quadratic_constraints):
            self.assertEqual(c.rhs, 0)
            self.assertDictEqual(c.linear.to_dict(use_name=True), {"x2": -1})
            self.assertDictEqual(c.quadratic.to_dict(use_name=True), {("x0", "x1"): 1})
            self.assertEqual(c.sense, senses[i])
