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

"""Test from_gurobipy and to_gurobipy"""

from test.optimization_test_case import QiskitOptimizationTestCase, requires_extra_library
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit_optimization.exceptions import QiskitOptimizationError
from qiskit_optimization.problems import Constraint, QuadraticProgram
from qiskit_optimization.translators.gurobipy import from_gurobipy, to_gurobipy


class TestGurobiTranslator(QiskitOptimizationTestCase):
    """Test from_gurobipy and to_gurobipy"""

    @requires_extra_library
    def test_from_and_to(self):
        """test from_gurobipy and to_gurobipy"""
        q_p = QuadraticProgram("test")
        q_p.binary_var(name="x")
        q_p.integer_var(name="y", lowerbound=-2, upperbound=4)
        q_p.continuous_var(name="z", lowerbound=-1.5, upperbound=3.2)
        q_p.minimize(constant=1, linear={"x": 1, "y": 2}, quadratic={("x", "y"): -1, ("z", "z"): 2})
        q_p.linear_constraint({"x": 2, "z": -1}, "==", 1)
        q_p.quadratic_constraint({"x": 2, "z": -1}, {("y", "z"): 3}, "==", 1)
        q_p2 = from_gurobipy(to_gurobipy(q_p))
        self.assertEqual(q_p.export_as_lp_string(), q_p2.export_as_lp_string())

        try:
            import gurobipy as gp
        except ImportError as ex:
            raise MissingOptionalLibraryError(
                libname="GUROBI",
                name="GurobiOptimizer",
                pip_install="pip install qiskit-optimization[gurobi]",
            ) from ex

        mod = gp.Model("test")
        x = mod.addVar(vtype=gp.GRB.BINARY, name="x")
        y = mod.addVar(vtype=gp.GRB.INTEGER, lb=-2, ub=4, name="y")
        z = mod.addVar(vtype=gp.GRB.CONTINUOUS, lb=-1.5, ub=3.2, name="z")
        mod.setObjective(1 + x + 2 * y - x * y + 2 * z * z)
        mod.addConstr(2 * x - z == 1, name="c0")
        mod.addConstr(2 * x - z + 3 * y * z == 1, name="q0")

        # Here I am unsure what to do, let's come back to it later
        # self.assertEqual(q_p.export_as_lp_string(), mod.export_as_lp_string())

        with self.assertRaises(QiskitOptimizationError):
            mod = gp.Model()
            mod.addVar(vtype=gp.GRB.SEMIINT, lb=1, name="x")
            _ = from_gurobipy(mod)

        with self.assertRaises(QiskitOptimizationError):
            mod = gp.Model()
            x = mod.addVar(vtype=gp.GRB.BINARY, name="x")
            y = mod.addVar(vtype=gp.GRB.BINARY, name="y")
            mod.addConstr((x == 1) >> (x + y <= 1))
            _ = from_gurobipy(mod)

        # test from_gurobipy without explicit variable names
        mod = gp.Model()
        x = mod.addVar(vtype=gp.GRB.BINARY)
        y = mod.addVar(vtype=gp.GRB.CONTINUOUS)
        z = mod.addVar(vtype=gp.GRB.INTEGER)
        mod.setObjective(x + y + z + x * y + y * z + x * z)
        mod.addConstr(x + y == z)  # linear EQ
        mod.addConstr(x + y >= z)  # linear GE
        mod.addConstr(x + y <= z)  # linear LE
        mod.addConstr(x * y == z)  # quadratic EQ
        mod.addConstr(x * y >= z)  # quadratic GE
        mod.addConstr(x * y <= z)  # quadratic LE
        q_p = from_gurobipy(mod)
        var_names = [v.name for v in q_p.variables]
        self.assertListEqual(var_names, ["C0", "C1", "C2"])
        senses = [Constraint.Sense.EQ, Constraint.Sense.GE, Constraint.Sense.LE]
        for i, c in enumerate(q_p.linear_constraints):
            self.assertDictEqual(c.linear.to_dict(use_name=True), {"C0": 1, "C1": 1, "C2": -1})
            self.assertEqual(c.rhs, 0)
            self.assertEqual(c.sense, senses[i])
        for i, c in enumerate(q_p.quadratic_constraints):
            self.assertEqual(c.rhs, 0)
            self.assertDictEqual(c.linear.to_dict(use_name=True), {"C2": -1})
            self.assertDictEqual(c.quadratic.to_dict(use_name=True), {("C0", "C1"): 1})
            self.assertEqual(c.sense, senses[i])
