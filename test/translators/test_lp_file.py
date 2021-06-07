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

""" Test DocplexMpTranslator """

import tempfile
from os import path
from test.optimization_test_case import QiskitOptimizationTestCase, requires_extra_library

from docplex.mp.model import DOcplexException

from qiskit_optimization.problems import Constraint, QuadraticObjective, QuadraticProgram, Variable
from qiskit_optimization.translators import LPFileTranslator


class TestDocplexMpTranslator(QiskitOptimizationTestCase):
    """DocplexMpTranslator tests"""

    @requires_extra_library
    def test_read_from_lp_file(self):
        """test read lp file"""
        try:
            translator = LPFileTranslator()
            with self.assertRaises(FileNotFoundError):
                translator.to_qp("")
            with self.assertRaises(FileNotFoundError):
                translator.to_qp("no_file.txt")
            lp_file = self.get_resource_path("test_quadratic_program.lp", "translators/resources")
            q_p = translator.to_qp(lp_file)
            self.assertEqual(q_p.name, "my problem")
            self.assertEqual(q_p.get_num_vars(), 3)
            self.assertEqual(q_p.get_num_binary_vars(), 1)
            self.assertEqual(q_p.get_num_integer_vars(), 1)
            self.assertEqual(q_p.get_num_continuous_vars(), 1)
            self.assertEqual(q_p.get_num_linear_constraints(), 3)
            self.assertEqual(q_p.get_num_quadratic_constraints(), 3)

            self.assertEqual(q_p.variables[0].name, "x")
            self.assertEqual(q_p.variables[0].vartype, Variable.Type.BINARY)
            self.assertEqual(q_p.variables[0].lowerbound, 0)
            self.assertEqual(q_p.variables[0].upperbound, 1)
            self.assertEqual(q_p.variables[1].name, "y")
            self.assertEqual(q_p.variables[1].vartype, Variable.Type.INTEGER)
            self.assertEqual(q_p.variables[1].lowerbound, -1)
            self.assertEqual(q_p.variables[1].upperbound, 5)
            self.assertEqual(q_p.variables[2].name, "z")
            self.assertEqual(q_p.variables[2].vartype, Variable.Type.CONTINUOUS)
            self.assertEqual(q_p.variables[2].lowerbound, -1)
            self.assertEqual(q_p.variables[2].upperbound, 5)

            self.assertEqual(q_p.objective.sense, QuadraticObjective.Sense.MINIMIZE)
            self.assertEqual(q_p.objective.constant, 1)
            self.assertDictEqual(
                q_p.objective.linear.to_dict(use_name=True), {"x": 1, "y": -1, "z": 10}
            )
            self.assertDictEqual(
                q_p.objective.quadratic.to_dict(use_name=True),
                {("x", "x"): 0.5, ("y", "z"): -1},
            )

            cst = q_p.linear_constraints
            self.assertEqual(cst[0].name, "lin_eq")
            self.assertDictEqual(cst[0].linear.to_dict(use_name=True), {"x": 1, "y": 2})
            self.assertEqual(cst[0].sense, Constraint.Sense.EQ)
            self.assertEqual(cst[0].rhs, 1)
            self.assertEqual(cst[1].name, "lin_leq")
            self.assertDictEqual(cst[1].linear.to_dict(use_name=True), {"x": 1, "y": 2})
            self.assertEqual(cst[1].sense, Constraint.Sense.LE)
            self.assertEqual(cst[1].rhs, 1)
            self.assertEqual(cst[2].name, "lin_geq")
            self.assertDictEqual(cst[2].linear.to_dict(use_name=True), {"x": 1, "y": 2})
            self.assertEqual(cst[2].sense, Constraint.Sense.GE)
            self.assertEqual(cst[2].rhs, 1)

            cst = q_p.quadratic_constraints
            self.assertEqual(cst[0].name, "quad_eq")
            self.assertDictEqual(cst[0].linear.to_dict(use_name=True), {"x": 1, "y": 1})
            self.assertDictEqual(
                cst[0].quadratic.to_dict(use_name=True),
                {("x", "x"): 1, ("y", "z"): -1, ("z", "z"): 2},
            )
            self.assertEqual(cst[0].sense, Constraint.Sense.EQ)
            self.assertEqual(cst[0].rhs, 1)
            self.assertEqual(cst[1].name, "quad_leq")
            self.assertDictEqual(cst[1].linear.to_dict(use_name=True), {"x": 1, "y": 1})
            self.assertDictEqual(
                cst[1].quadratic.to_dict(use_name=True),
                {("x", "x"): 1, ("y", "z"): -1, ("z", "z"): 2},
            )
            self.assertEqual(cst[1].sense, Constraint.Sense.LE)
            self.assertEqual(cst[1].rhs, 1)
            self.assertEqual(cst[2].name, "quad_geq")
            self.assertDictEqual(cst[2].linear.to_dict(use_name=True), {"x": 1, "y": 1})
            self.assertDictEqual(
                cst[2].quadratic.to_dict(use_name=True),
                {("x", "x"): 1, ("y", "z"): -1, ("z", "z"): 2},
            )
            self.assertEqual(cst[2].sense, Constraint.Sense.GE)
            self.assertEqual(cst[2].rhs, 1)
        except RuntimeError as ex:
            self.fail(str(ex))

    def test_write_to_lp_file(self):
        """test write problem"""
        q_p = QuadraticProgram("my problem")
        q_p.binary_var("x")
        q_p.integer_var(-1, 5, "y")
        q_p.continuous_var(-1, 5, "z")
        q_p.minimize(1, {"x": 1, "y": -1, "z": 10}, {("x", "x"): 0.5, ("y", "z"): -1})
        q_p.linear_constraint({"x": 1, "y": 2}, "==", 1, "lin_eq")
        q_p.linear_constraint({"x": 1, "y": 2}, "<=", 1, "lin_leq")
        q_p.linear_constraint({"x": 1, "y": 2}, ">=", 1, "lin_geq")
        q_p.quadratic_constraint(
            {"x": 1, "y": 1},
            {("x", "x"): 1, ("y", "z"): -1, ("z", "z"): 2},
            "==",
            1,
            "quad_eq",
        )
        q_p.quadratic_constraint(
            {"x": 1, "y": 1},
            {("x", "x"): 1, ("y", "z"): -1, ("z", "z"): 2},
            "<=",
            1,
            "quad_leq",
        )
        q_p.quadratic_constraint(
            {"x": 1, "y": 1},
            {("x", "x"): 1, ("y", "z"): -1, ("z", "z"): 2},
            ">=",
            1,
            "quad_geq",
        )

        reference_file_name = self.get_resource_path(
            "test_quadratic_program.lp", "translators/resources"
        )
        with tempfile.NamedTemporaryFile(mode="w+t", suffix=".lp") as temp_output_file:
            translator = LPFileTranslator(temp_output_file.name)
            translator.from_qp(q_p)
            with open(reference_file_name) as reference:
                lines1 = temp_output_file.readlines()
                lines2 = reference.readlines()
                self.assertListEqual(lines1, lines2)

        with tempfile.TemporaryDirectory() as temp_problem_dir:
            translator = LPFileTranslator(temp_problem_dir)
            translator.from_qp(q_p)
            with open(path.join(temp_problem_dir, "my_problem.lp")) as file1, open(
                reference_file_name
            ) as file2:
                lines1 = file1.readlines()
                lines2 = file2.readlines()
                self.assertListEqual(lines1, lines2)

        with self.assertRaises(OSError):
            translator = LPFileTranslator("/cannot/write/this/file.lp")
            translator.from_qp(q_p)

        with self.assertRaises(DOcplexException):
            translator = LPFileTranslator("")
            translator.from_qp(q_p)
