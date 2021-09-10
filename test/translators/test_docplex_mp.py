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

    def test_from_without_variable_names(self):
        """test from_docplex_mp without explicit variable names"""
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

    def test_unsupported_features(self):
        """Test unsupported features"""
        with self.subTest("semiinteget_var"), self.assertRaises(QiskitOptimizationError):
            mod = Model()
            mod.semiinteger_var(lb=1, name="x")
            _ = from_docplex_mp(mod)

        with self.subTest("range constraint"), self.assertRaises(QiskitOptimizationError):
            mod = Model()
            x = mod.binary_var("x")
            mod.add_range(0, 2 * x, 1)
            _ = from_docplex_mp(mod)

        with self.subTest("equivalence constraint"), self.assertRaises(QiskitOptimizationError):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.binary_var("y")
            mod.add_equivalence(x, x + y <= 1, 1)
            _ = from_docplex_mp(mod)

        with self.subTest("not equal constraint"), self.assertRaises(QiskitOptimizationError):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.binary_var("y")
            mod.add(x != y)
            _ = from_docplex_mp(mod)

        with self.subTest("PWL constraint"), self.assertRaises(QiskitOptimizationError):
            mod = Model()
            x = mod.binary_var("x")
            mod.add(mod.piecewise(-1, [(0, 0)], 1)(x) <= 1)
            _ = from_docplex_mp(mod)

        with self.subTest("lazy constraint"), self.assertRaises(QiskitOptimizationError):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.binary_var("y")
            mod.add_lazy_constraint(x + y <= 1)
            _ = from_docplex_mp(mod)

        with self.subTest("user cut constraint"), self.assertRaises(QiskitOptimizationError):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.binary_var("y")
            mod.add_user_cut_constraint(x + y <= 1)
            _ = from_docplex_mp(mod)

        with self.subTest("sos1"), self.assertRaises(QiskitOptimizationError):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.binary_var("y")
            mod.add_sos1([x, y])
            _ = from_docplex_mp(mod)

        with self.subTest("sos2"), self.assertRaises(QiskitOptimizationError):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.binary_var("y")
            z = mod.binary_var("z")
            mod.add_sos2([x, y, z])
            _ = from_docplex_mp(mod)

    def test_indicator_constraints(self):
        """Test indicator constraints"""
        with self.subTest("active 0, sense <="):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.integer_var(lb=-1, ub=2, name="y")
            z = mod.continuous_var(lb=-1, ub=2, name="z")
            mod.add_indicator(binary_var=x, active_value=0, linear_ct=(y + 2 * z <= 1), name="ind")
            quad_prog = from_docplex_mp(mod)
            self.assertEqual(quad_prog.get_num_linear_constraints(), 1)
            ind = quad_prog.get_linear_constraint(0)
            self.assertEqual(ind.name, "ind")
            self.assertEqual(ind.sense, Constraint.Sense.LE)
            self.assertDictEqual(ind.linear.to_dict(use_name=True), {"x": -5.0, "y": 1.0, "z": 2.0})
            self.assertEqual(ind.rhs, 1)

        with self.subTest("active 0, sense >="):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.integer_var(lb=-1, ub=2, name="y")
            z = mod.continuous_var(lb=-1, ub=2, name="z")
            mod.add_indicator(binary_var=x, active_value=0, linear_ct=(y + 2 * z >= 1), name="ind")
            quad_prog = from_docplex_mp(mod)
            self.assertEqual(quad_prog.get_num_linear_constraints(), 1)
            ind = quad_prog.get_linear_constraint(0)
            self.assertEqual(ind.name, "ind")
            self.assertEqual(ind.sense, Constraint.Sense.GE)
            self.assertDictEqual(ind.linear.to_dict(use_name=True), {"x": 4.0, "y": 1.0, "z": 2.0})
            self.assertEqual(ind.rhs, 1)

        with self.subTest("active 1, sense <="):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.integer_var(lb=-1, ub=2, name="y")
            z = mod.continuous_var(lb=-1, ub=2, name="z")
            mod.add_indicator(binary_var=x, active_value=1, linear_ct=(y + 2 * z <= 1), name="ind")
            quad_prog = from_docplex_mp(mod)
            self.assertEqual(quad_prog.get_num_linear_constraints(), 1)
            ind = quad_prog.get_linear_constraint(0)
            self.assertEqual(ind.name, "ind")
            self.assertEqual(ind.sense, Constraint.Sense.LE)
            self.assertDictEqual(ind.linear.to_dict(use_name=True), {"x": 5.0, "y": 1.0, "z": 2.0})
            self.assertEqual(ind.rhs, 6)

        with self.subTest("active 1, sense >="):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.integer_var(lb=-1, ub=2, name="y")
            z = mod.continuous_var(lb=-1, ub=2, name="z")
            mod.add_indicator(binary_var=x, active_value=1, linear_ct=(y + 2 * z >= 1), name="ind")
            quad_prog = from_docplex_mp(mod)
            self.assertEqual(quad_prog.get_num_linear_constraints(), 1)
            ind = quad_prog.get_linear_constraint(0)
            self.assertEqual(ind.name, "ind")
            self.assertEqual(ind.sense, Constraint.Sense.GE)
            self.assertDictEqual(ind.linear.to_dict(use_name=True), {"x": -4.0, "y": 1.0, "z": 2.0})
            self.assertEqual(ind.rhs, -3)

        with self.subTest("active 0, sense =="):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.integer_var(lb=-1, ub=2, name="y")
            z = mod.continuous_var(lb=-1, ub=2, name="z")
            mod.add_indicator(binary_var=x, active_value=0, linear_ct=(y + 2 * z == 1), name="ind")
            quad_prog = from_docplex_mp(mod)
            self.assertEqual(quad_prog.get_num_linear_constraints(), 2)

            ind = quad_prog.get_linear_constraint(0)
            self.assertEqual(ind.name, "ind_LE")
            self.assertEqual(ind.sense, Constraint.Sense.LE)
            self.assertDictEqual(ind.linear.to_dict(use_name=True), {"x": -5.0, "y": 1.0, "z": 2.0})
            self.assertEqual(ind.rhs, 1)

            ind = quad_prog.get_linear_constraint(1)
            self.assertEqual(ind.name, "ind_GE")
            self.assertEqual(ind.sense, Constraint.Sense.GE)
            self.assertDictEqual(ind.linear.to_dict(use_name=True), {"x": 4.0, "y": 1.0, "z": 2.0})
            self.assertEqual(ind.rhs, 1)

        with self.subTest("active 1, sense =="):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.integer_var(lb=-1, ub=2, name="y")
            z = mod.continuous_var(lb=-1, ub=2, name="z")
            mod.add_indicator(binary_var=x, active_value=1, linear_ct=(y + 2 * z == 1), name="ind")
            quad_prog = from_docplex_mp(mod)
            self.assertEqual(quad_prog.get_num_linear_constraints(), 2)

            ind = quad_prog.get_linear_constraint(0)
            self.assertEqual(ind.name, "ind_LE")
            self.assertEqual(ind.sense, Constraint.Sense.LE)
            self.assertDictEqual(ind.linear.to_dict(use_name=True), {"x": 5.0, "y": 1.0, "z": 2.0})
            self.assertEqual(ind.rhs, 6)

            ind = quad_prog.get_linear_constraint(1)
            self.assertEqual(ind.name, "ind_GE")
            self.assertEqual(ind.sense, Constraint.Sense.GE)
            self.assertDictEqual(ind.linear.to_dict(use_name=True), {"x": -4.0, "y": 1.0, "z": 2.0})
            self.assertEqual(ind.rhs, -3)

        with self.subTest("active 0, sense <=, indicator_big_m"):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.integer_var(lb=-1, ub=2, name="y")
            z = mod.continuous_var(lb=-1, ub=2, name="z")
            mod.add_indicator(binary_var=x, active_value=0, linear_ct=(y + 2 * z <= 1), name="ind")
            quad_prog = from_docplex_mp(mod, indicator_big_m=100)
            self.assertEqual(quad_prog.get_num_linear_constraints(), 1)
            ind = quad_prog.get_linear_constraint(0)
            self.assertEqual(ind.name, "ind")
            self.assertEqual(ind.sense, Constraint.Sense.LE)
            self.assertDictEqual(
                ind.linear.to_dict(use_name=True), {"x": -100.0, "y": 1.0, "z": 2.0}
            )
            self.assertEqual(ind.rhs, 1)

        with self.subTest("active 0, sense >=, indicator_big_m"):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.integer_var(lb=-1, ub=2, name="y")
            z = mod.continuous_var(lb=-1, ub=2, name="z")
            mod.add_indicator(binary_var=x, active_value=0, linear_ct=(y + 2 * z >= 1), name="ind")
            quad_prog = from_docplex_mp(mod, indicator_big_m=100)
            self.assertEqual(quad_prog.get_num_linear_constraints(), 1)
            ind = quad_prog.get_linear_constraint(0)
            self.assertEqual(ind.name, "ind")
            self.assertEqual(ind.sense, Constraint.Sense.GE)
            self.assertDictEqual(
                ind.linear.to_dict(use_name=True), {"x": 100.0, "y": 1.0, "z": 2.0}
            )
            self.assertEqual(ind.rhs, 1)

        with self.subTest("active 1, sense <=, indicator_big_m"):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.integer_var(lb=-1, ub=2, name="y")
            z = mod.continuous_var(lb=-1, ub=2, name="z")
            mod.add_indicator(binary_var=x, active_value=1, linear_ct=(y + 2 * z <= 1), name="ind")
            quad_prog = from_docplex_mp(mod, indicator_big_m=100)
            self.assertEqual(quad_prog.get_num_linear_constraints(), 1)
            ind = quad_prog.get_linear_constraint(0)
            self.assertEqual(ind.name, "ind")
            self.assertEqual(ind.sense, Constraint.Sense.LE)
            self.assertDictEqual(
                ind.linear.to_dict(use_name=True), {"x": 100.0, "y": 1.0, "z": 2.0}
            )
            self.assertEqual(ind.rhs, 101)

        with self.subTest("active 1, sense >=, indicator_big_m"):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.integer_var(lb=-1, ub=2, name="y")
            z = mod.continuous_var(lb=-1, ub=2, name="z")
            mod.add_indicator(binary_var=x, active_value=1, linear_ct=(y + 2 * z >= 1), name="ind")
            quad_prog = from_docplex_mp(mod, indicator_big_m=100)
            self.assertEqual(quad_prog.get_num_linear_constraints(), 1)
            ind = quad_prog.get_linear_constraint(0)
            self.assertEqual(ind.name, "ind")
            self.assertEqual(ind.sense, Constraint.Sense.GE)
            self.assertDictEqual(
                ind.linear.to_dict(use_name=True), {"x": -100.0, "y": 1.0, "z": 2.0}
            )
            self.assertEqual(ind.rhs, -99)

        with self.subTest("active 0, sense ==, indicator_big_m"):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.integer_var(lb=-1, ub=2, name="y")
            z = mod.continuous_var(lb=-1, ub=2, name="z")
            mod.add_indicator(binary_var=x, active_value=0, linear_ct=(y + 2 * z == 1), name="ind")
            quad_prog = from_docplex_mp(mod, indicator_big_m=100)
            self.assertEqual(quad_prog.get_num_linear_constraints(), 2)

            ind = quad_prog.get_linear_constraint(0)
            self.assertEqual(ind.name, "ind_LE")
            self.assertEqual(ind.sense, Constraint.Sense.LE)
            self.assertDictEqual(
                ind.linear.to_dict(use_name=True), {"x": -100.0, "y": 1.0, "z": 2.0}
            )
            self.assertEqual(ind.rhs, 1)

            ind = quad_prog.get_linear_constraint(1)
            self.assertEqual(ind.name, "ind_GE")
            self.assertEqual(ind.sense, Constraint.Sense.GE)
            self.assertDictEqual(
                ind.linear.to_dict(use_name=True), {"x": 100.0, "y": 1.0, "z": 2.0}
            )
            self.assertEqual(ind.rhs, 1)

        with self.subTest("active 1, sense ==, indicator_big_m"):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.integer_var(lb=-1, ub=2, name="y")
            z = mod.continuous_var(lb=-1, ub=2, name="z")
            mod.add_indicator(binary_var=x, active_value=1, linear_ct=(y + 2 * z == 1), name="ind")
            quad_prog = from_docplex_mp(mod, indicator_big_m=100)
            self.assertEqual(quad_prog.get_num_linear_constraints(), 2)

            ind = quad_prog.get_linear_constraint(0)
            self.assertEqual(ind.name, "ind_LE")
            self.assertEqual(ind.sense, Constraint.Sense.LE)
            self.assertDictEqual(
                ind.linear.to_dict(use_name=True), {"x": 100.0, "y": 1.0, "z": 2.0}
            )
            self.assertEqual(ind.rhs, 101)

            ind = quad_prog.get_linear_constraint(1)
            self.assertEqual(ind.name, "ind_GE")
            self.assertEqual(ind.sense, Constraint.Sense.GE)
            self.assertDictEqual(
                ind.linear.to_dict(use_name=True), {"x": -100.0, "y": 1.0, "z": 2.0}
            )
            self.assertEqual(ind.rhs, -99)

        with self.subTest("active 0, sense <=, obvious bound"):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.integer_var(lb=-1, ub=2, name="y")
            z = mod.continuous_var(lb=-1, ub=2, name="z")
            mod.add_indicator(binary_var=x, active_value=0, linear_ct=(y + 2 * z <= 10), name="ind")
            quad_prog = from_docplex_mp(mod)
            self.assertEqual(quad_prog.get_num_linear_constraints(), 1)
            ind = quad_prog.get_linear_constraint(0)
            self.assertEqual(ind.name, "ind")
            self.assertEqual(ind.sense, Constraint.Sense.LE)
            self.assertDictEqual(ind.linear.to_dict(use_name=True), {"y": 1.0, "z": 2.0})
            self.assertEqual(ind.rhs, 10)

        with self.subTest("active 0, sense >=, obvious bound"):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.integer_var(lb=-1, ub=2, name="y")
            z = mod.continuous_var(lb=-1, ub=2, name="z")
            mod.add_indicator(
                binary_var=x, active_value=0, linear_ct=(y + 2 * z >= -10), name="ind"
            )
            quad_prog = from_docplex_mp(mod)
            self.assertEqual(quad_prog.get_num_linear_constraints(), 1)
            ind = quad_prog.get_linear_constraint(0)
            self.assertEqual(ind.name, "ind")
            self.assertEqual(ind.sense, Constraint.Sense.GE)
            self.assertDictEqual(ind.linear.to_dict(use_name=True), {"y": 1.0, "z": 2.0})
            self.assertEqual(ind.rhs, -10)

        with self.subTest("active 1, sense <=, obvious bound"):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.integer_var(lb=-1, ub=2, name="y")
            z = mod.continuous_var(lb=-1, ub=2, name="z")
            mod.add_indicator(binary_var=x, active_value=1, linear_ct=(y + 2 * z <= 10), name="ind")
            quad_prog = from_docplex_mp(mod)
            self.assertEqual(quad_prog.get_num_linear_constraints(), 1)
            ind = quad_prog.get_linear_constraint(0)
            self.assertEqual(ind.name, "ind")
            self.assertEqual(ind.sense, Constraint.Sense.LE)
            self.assertDictEqual(ind.linear.to_dict(use_name=True), {"y": 1.0, "z": 2.0})
            self.assertEqual(ind.rhs, 10)

        with self.subTest("active 1, sense >=, obvious bound"):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.integer_var(lb=-1, ub=2, name="y")
            z = mod.continuous_var(lb=-1, ub=2, name="z")
            mod.add_indicator(
                binary_var=x, active_value=1, linear_ct=(y + 2 * z >= -10), name="ind"
            )
            quad_prog = from_docplex_mp(mod)
            self.assertEqual(quad_prog.get_num_linear_constraints(), 1)
            ind = quad_prog.get_linear_constraint(0)
            self.assertEqual(ind.name, "ind")
            self.assertEqual(ind.sense, Constraint.Sense.GE)
            self.assertDictEqual(ind.linear.to_dict(use_name=True), {"y": 1.0, "z": 2.0})
            self.assertEqual(ind.rhs, -10)

        with self.subTest("active 0, sense ==, too small rhs"):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.integer_var(lb=-1, ub=2, name="y")
            z = mod.continuous_var(lb=-1, ub=2, name="z")
            mod.add_indicator(
                binary_var=x, active_value=0, linear_ct=(y + 2 * z == -10), name="ind"
            )
            quad_prog = from_docplex_mp(mod)
            self.assertEqual(quad_prog.get_num_linear_constraints(), 2)

            ind = quad_prog.get_linear_constraint(0)
            self.assertEqual(ind.name, "ind_LE")
            self.assertEqual(ind.sense, Constraint.Sense.LE)
            self.assertDictEqual(
                ind.linear.to_dict(use_name=True), {"x": -16.0, "y": 1.0, "z": 2.0}
            )
            self.assertEqual(ind.rhs, -10)

            ind = quad_prog.get_linear_constraint(1)
            self.assertEqual(ind.name, "ind_GE")
            self.assertEqual(ind.sense, Constraint.Sense.GE)
            self.assertDictEqual(ind.linear.to_dict(use_name=True), {"y": 1.0, "z": 2.0})
            self.assertEqual(ind.rhs, -10)

        with self.subTest("active 0, sense ==, too large rhs"):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.integer_var(lb=-1, ub=2, name="y")
            z = mod.continuous_var(lb=-1, ub=2, name="z")
            mod.add_indicator(binary_var=x, active_value=0, linear_ct=(y + 2 * z == 10), name="ind")
            quad_prog = from_docplex_mp(mod)
            self.assertEqual(quad_prog.get_num_linear_constraints(), 2)

            ind = quad_prog.get_linear_constraint(0)
            self.assertEqual(ind.name, "ind_LE")
            self.assertEqual(ind.sense, Constraint.Sense.LE)
            self.assertDictEqual(ind.linear.to_dict(use_name=True), {"y": 1.0, "z": 2.0})
            self.assertEqual(ind.rhs, 10)

            ind = quad_prog.get_linear_constraint(1)
            self.assertEqual(ind.name, "ind_GE")
            self.assertEqual(ind.sense, Constraint.Sense.GE)
            self.assertDictEqual(ind.linear.to_dict(use_name=True), {"x": 13, "y": 1.0, "z": 2.0})
            self.assertEqual(ind.rhs, 10)

        with self.subTest("active 1, sense ==, too small rhs"):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.integer_var(lb=-1, ub=2, name="y")
            z = mod.continuous_var(lb=-1, ub=2, name="z")
            mod.add_indicator(
                binary_var=x, active_value=1, linear_ct=(y + 2 * z == -10), name="ind"
            )
            quad_prog = from_docplex_mp(mod)
            self.assertEqual(quad_prog.get_num_linear_constraints(), 2)

            ind = quad_prog.get_linear_constraint(0)
            self.assertEqual(ind.name, "ind_LE")
            self.assertEqual(ind.sense, Constraint.Sense.LE)
            self.assertDictEqual(ind.linear.to_dict(use_name=True), {"x": 16.0, "y": 1.0, "z": 2.0})
            self.assertEqual(ind.rhs, 6)

            ind = quad_prog.get_linear_constraint(1)
            self.assertEqual(ind.name, "ind_GE")
            self.assertEqual(ind.sense, Constraint.Sense.GE)
            self.assertDictEqual(ind.linear.to_dict(use_name=True), {"y": 1.0, "z": 2.0})
            self.assertEqual(ind.rhs, -10)

        with self.subTest("active 1, sense ==, too large rhs"):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.integer_var(lb=-1, ub=2, name="y")
            z = mod.continuous_var(lb=-1, ub=2, name="z")
            mod.add_indicator(binary_var=x, active_value=1, linear_ct=(y + 2 * z == 10), name="ind")
            quad_prog = from_docplex_mp(mod)
            self.assertEqual(quad_prog.get_num_linear_constraints(), 2)

            ind = quad_prog.get_linear_constraint(0)
            self.assertEqual(ind.name, "ind_LE")
            self.assertEqual(ind.sense, Constraint.Sense.LE)
            self.assertDictEqual(ind.linear.to_dict(use_name=True), {"y": 1.0, "z": 2.0})
            self.assertEqual(ind.rhs, 10)

            ind = quad_prog.get_linear_constraint(1)
            self.assertEqual(ind.name, "ind_GE")
            self.assertEqual(ind.sense, Constraint.Sense.GE)
            self.assertDictEqual(
                ind.linear.to_dict(use_name=True), {"x": -13.0, "y": 1.0, "z": 2.0}
            )
            self.assertEqual(ind.rhs, -3)

        with self.subTest("no name"):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.integer_var(lb=-1, ub=2, name="y")
            mod.add_indicator(binary_var=x, active_value=1, linear_ct=(y == 1))
            mod.add_indicator(binary_var=x, active_value=1, linear_ct=(y <= 1))
            mod.add_indicator(binary_var=x, active_value=1, linear_ct=(y >= 1))
            quad_prog = from_docplex_mp(mod)
            self.assertEqual(quad_prog.get_num_linear_constraints(), 4)

            ind = quad_prog.get_linear_constraint(0)
            self.assertEqual(ind.name, "ind0_LE")
            ind = quad_prog.get_linear_constraint(1)
            self.assertEqual(ind.name, "ind0_GE")
            ind = quad_prog.get_linear_constraint(2)
            self.assertEqual(ind.name, "ind1")
            ind = quad_prog.get_linear_constraint(3)
            self.assertEqual(ind.name, "ind2")

        with self.subTest("sense <=, binary_var is included as part of linear_ct too"):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.integer_var(lb=-1, ub=2, name="y")
            z = mod.continuous_var(lb=-1, ub=2, name="z")
            mod.add_indicator(binary_var=x, linear_ct=(x + y + 2 * z <= -10))
            quad_prog = from_docplex_mp(mod)
            self.assertEqual(quad_prog.get_num_linear_constraints(), 1)

            ind = quad_prog.get_linear_constraint(0)
            self.assertEqual(ind.name, "ind0")
            self.assertEqual(ind.sense, Constraint.Sense.LE)
            self.assertDictEqual(ind.linear.to_dict(use_name=True), {"x": 18.0, "y": 1.0, "z": 2.0})
            self.assertEqual(ind.rhs, 7)

        with self.subTest("sense >=, binary_var is included as part of linear_ct too"):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.integer_var(lb=-1, ub=2, name="y")
            z = mod.continuous_var(lb=-1, ub=2, name="z")
            mod.add_indicator(binary_var=x, linear_ct=(x + y + 2 * z >= 10))
            quad_prog = from_docplex_mp(mod)
            self.assertEqual(quad_prog.get_num_linear_constraints(), 1)

            ind = quad_prog.get_linear_constraint(0)
            self.assertEqual(ind.name, "ind0")
            self.assertEqual(ind.sense, Constraint.Sense.GE)
            self.assertDictEqual(
                ind.linear.to_dict(use_name=True), {"x": -12.0, "y": 1.0, "z": 2.0}
            )
            self.assertEqual(ind.rhs, -3)

        with self.subTest("sense ==, binary_var is included as part of linear_ct too"):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.integer_var(lb=-1, ub=2, name="y")
            z = mod.continuous_var(lb=-1, ub=2, name="z")
            mod.add_indicator(binary_var=x, linear_ct=(x + y + 2 * z == 0))
            quad_prog = from_docplex_mp(mod)
            self.assertEqual(quad_prog.get_num_linear_constraints(), 2)

            ind = quad_prog.get_linear_constraint(0)
            self.assertEqual(ind.name, "ind0_LE")
            self.assertEqual(ind.sense, Constraint.Sense.LE)
            self.assertDictEqual(ind.linear.to_dict(use_name=True), {"x": 8.0, "y": 1.0, "z": 2.0})
            self.assertEqual(ind.rhs, 7)

            ind = quad_prog.get_linear_constraint(1)
            self.assertEqual(ind.name, "ind0_GE")
            self.assertEqual(ind.sense, Constraint.Sense.GE)
            self.assertDictEqual(ind.linear.to_dict(use_name=True), {"x": -2.0, "y": 1.0, "z": 2.0})
            self.assertEqual(ind.rhs, -3)

    def test_logical_expressions(self):
        """test from_docplex_mp with logical expressions"""

        with self.subTest("logical NOT"):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.logical_not(x)
            mod.add_constraint(y <= 1)
            mod.add_constraint(y ** 2 == 2)
            q_p = from_docplex_mp(mod)
            self.assertListEqual([v.name for v in q_p.variables], ["x", "_not1"])
            self.assertEqual(q_p.get_num_linear_constraints(), 2)

            lin = q_p.get_linear_constraint(0)
            self.assertEqual(lin.name, "c0")
            self.assertEqual(lin.sense, Constraint.Sense.EQ)
            self.assertDictEqual(lin.linear.to_dict(use_name=True), {"x": 1, "_not1": 1})
            self.assertAlmostEqual(lin.rhs, 1)

            lin = q_p.get_linear_constraint(1)
            self.assertEqual(lin.name, "c1")
            self.assertEqual(lin.sense, Constraint.Sense.LE)
            self.assertDictEqual(lin.linear.to_dict(use_name=True), {"_not1": 1})
            self.assertAlmostEqual(lin.rhs, 1)

            self.assertEqual(q_p.get_num_quadratic_constraints(), 1)
            quad = q_p.get_quadratic_constraint(0)
            self.assertEqual(quad.name, "q0")
            self.assertEqual(quad.sense, Constraint.Sense.EQ)
            self.assertDictEqual(quad.linear.to_dict(), {})
            self.assertDictEqual(quad.quadratic.to_dict(use_name=True), {("_not1", "_not1"): 1})
            self.assertAlmostEqual(quad.rhs, 2)

        with self.subTest("logical AND"):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.binary_var("y")
            z = mod.logical_and(x, y)
            mod.add_constraint(z <= 1)
            mod.add_constraint(z ** 2 == 2)
            q_p = from_docplex_mp(mod)
            self.assertListEqual([v.name for v in q_p.variables], ["x", "y", "_and2"])
            self.assertEqual(q_p.get_num_linear_constraints(), 4)

            lin = q_p.get_linear_constraint(0)
            self.assertEqual(lin.name, "c0")
            self.assertEqual(lin.sense, Constraint.Sense.LE)
            self.assertDictEqual(lin.linear.to_dict(use_name=True), {"x": -1, "_and2": 1})
            self.assertAlmostEqual(lin.rhs, 0)

            lin = q_p.get_linear_constraint(1)
            self.assertEqual(lin.name, "c1")
            self.assertEqual(lin.sense, Constraint.Sense.LE)
            self.assertDictEqual(lin.linear.to_dict(use_name=True), {"y": -1, "_and2": 1})
            self.assertAlmostEqual(lin.rhs, 0)

            lin = q_p.get_linear_constraint(2)
            self.assertEqual(lin.name, "c2")
            self.assertEqual(lin.sense, Constraint.Sense.GE)
            self.assertDictEqual(lin.linear.to_dict(use_name=True), {"x": -1, "y": -1, "_and2": 1})
            self.assertAlmostEqual(lin.rhs, -1)

            lin = q_p.get_linear_constraint(3)
            self.assertEqual(lin.name, "c3")
            self.assertEqual(lin.sense, Constraint.Sense.LE)
            self.assertDictEqual(lin.linear.to_dict(use_name=True), {"_and2": 1})
            self.assertAlmostEqual(lin.rhs, 1)

            self.assertEqual(q_p.get_num_quadratic_constraints(), 1)
            quad = q_p.get_quadratic_constraint(0)
            self.assertEqual(quad.name, "q0")
            self.assertEqual(quad.sense, Constraint.Sense.EQ)
            self.assertDictEqual(quad.linear.to_dict(), {})
            self.assertDictEqual(quad.quadratic.to_dict(use_name=True), {("_and2", "_and2"): 1})
            self.assertAlmostEqual(quad.rhs, 2)

        with self.subTest("logical OR"):
            mod = Model()
            x = mod.binary_var("x")
            y = mod.binary_var("y")
            z = mod.logical_or(x, y)
            mod.add_constraint(z <= 1)
            mod.add_constraint(z ** 2 == 2)
            q_p = from_docplex_mp(mod)
            self.assertListEqual([v.name for v in q_p.variables], ["x", "y", "_or2"])
            self.assertEqual(q_p.get_num_linear_constraints(), 4)

            lin = q_p.get_linear_constraint(0)
            self.assertEqual(lin.name, "c0")
            self.assertEqual(lin.sense, Constraint.Sense.LE)
            self.assertDictEqual(lin.linear.to_dict(use_name=True), {"x": 1, "_or2": -1})
            self.assertAlmostEqual(lin.rhs, 0)

            lin = q_p.get_linear_constraint(1)
            self.assertEqual(lin.name, "c1")
            self.assertEqual(lin.sense, Constraint.Sense.LE)
            self.assertDictEqual(lin.linear.to_dict(use_name=True), {"y": 1, "_or2": -1})
            self.assertAlmostEqual(lin.rhs, 0)

            lin = q_p.get_linear_constraint(2)
            self.assertEqual(lin.name, "c2")
            self.assertEqual(lin.sense, Constraint.Sense.LE)
            self.assertDictEqual(lin.linear.to_dict(use_name=True), {"x": -1, "y": -1, "_or2": 1})
            self.assertAlmostEqual(lin.rhs, 0)

            lin = q_p.get_linear_constraint(3)
            self.assertEqual(lin.name, "c3")
            self.assertEqual(lin.sense, Constraint.Sense.LE)
            self.assertDictEqual(lin.linear.to_dict(use_name=True), {"_or2": 1})
            self.assertAlmostEqual(lin.rhs, 1)

            self.assertEqual(q_p.get_num_quadratic_constraints(), 1)
            quad = q_p.get_quadratic_constraint(0)
            self.assertEqual(quad.name, "q0")
            self.assertEqual(quad.sense, Constraint.Sense.EQ)
            self.assertDictEqual(quad.linear.to_dict(), {})
            self.assertDictEqual(quad.quadratic.to_dict(use_name=True), {("_or2", "_or2"): 1})
            self.assertAlmostEqual(quad.rhs, 2)

    def test_trivial_constraints_from_docplex_mp(self):
        """test trivial constraints of from_docplex_mp"""

        with self.subTest("trivial linear constraint"), self.assertWarns(UserWarning):
            mod = Model()
            x = mod.binary_var("x")
            mod.add_constraint(x + 1 <= x + 1)
            q_p = from_docplex_mp(mod)
            self.assertListEqual([v.name for v in q_p.variables], ["x"])
            self.assertEqual(q_p.get_num_linear_constraints(), 1)
            self.assertEqual(q_p.get_num_quadratic_constraints(), 0)
            lin = q_p.get_linear_constraint(0)
            self.assertEqual(lin.name, "c0")
            self.assertEqual(lin.sense, Constraint.Sense.LE)
            self.assertDictEqual(lin.linear.to_dict(), {})
            self.assertAlmostEqual(lin.rhs, 0)

        with self.subTest("trivial quadratic constraint"), self.assertWarns(UserWarning):
            mod = Model()
            x = mod.binary_var("x")
            mod.add_constraint(x * x == x * x)
            q_p = from_docplex_mp(mod)
            self.assertListEqual([v.name for v in q_p.variables], ["x"])
            self.assertEqual(q_p.get_num_linear_constraints(), 0)
            self.assertEqual(q_p.get_num_quadratic_constraints(), 1)
            quad = q_p.get_quadratic_constraint(0)
            self.assertEqual(quad.name, "q0")
            self.assertEqual(quad.sense, Constraint.Sense.EQ)
            self.assertDictEqual(quad.linear.to_dict(), {})
            self.assertDictEqual(quad.quadratic.to_dict(), {})
            self.assertAlmostEqual(quad.rhs, 0)

        with self.subTest("trivial indicator constraint"), self.assertWarns(UserWarning):
            mod = Model()
            x = mod.binary_var("x")
            mod.add_indicator(x, x + 1 >= x + 1)
            q_p = from_docplex_mp(mod)
            self.assertListEqual([v.name for v in q_p.variables], ["x"])
            self.assertEqual(q_p.get_num_linear_constraints(), 1)
            self.assertEqual(q_p.get_num_quadratic_constraints(), 0)
            ind = q_p.get_linear_constraint(0)
            self.assertEqual(ind.name, "ind0")
            self.assertEqual(ind.sense, Constraint.Sense.GE)
            self.assertDictEqual(ind.linear.to_dict(), {})
            self.assertAlmostEqual(ind.rhs, 0)

    def test_trivial_constraints_to_docplex_mp(self):
        """test trivial constraints of to_docplex_mp"""

        with self.subTest("trivial linear constraint"):
            q_p = QuadraticProgram()
            q_p.linear_constraint(sense="==", rhs=1.0)
            mod = to_docplex_mp(q_p)
            self.assertEqual(mod.number_of_variables, 0)
            self.assertEqual(mod.number_of_constraints, 1)
            self.assertEqual(mod.number_of_linear_constraints, 1)
            cst = mod.get_constraint_by_index(0)
            left = cst.get_left_expr()
            self.assertTrue(left.is_constant())
            self.assertAlmostEqual(left.constant, 0)
            right = cst.get_right_expr()
            self.assertTrue(right.is_constant())
            self.assertAlmostEqual(right.constant, 1)

        with self.subTest("trivial quadratic constraint"):
            q_p = QuadraticProgram()
            q_p.quadratic_constraint(sense="==", rhs=1.0)
            mod = to_docplex_mp(q_p)
            self.assertEqual(mod.number_of_variables, 0)
            self.assertEqual(mod.number_of_constraints, 1)
            self.assertEqual(mod.number_of_linear_constraints, 1)
            cst = mod.get_constraint_by_index(0)
            left = cst.get_left_expr()
            self.assertTrue(left.is_constant())
            self.assertAlmostEqual(left.constant, 0)
            right = cst.get_right_expr()
            self.assertTrue(right.is_constant())
            self.assertAlmostEqual(right.constant, 1)
