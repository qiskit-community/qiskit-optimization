# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test file_io's LP/MPS import/export functions"""

import tempfile
import unittest
from os import path

from test.optimization_test_case import QiskitOptimizationTestCase

from docplex.mp.model import DOcplexException

import qiskit_optimization.optionals as _optionals
from qiskit_optimization.problems import QuadraticProgram, Constraint, QuadraticObjective, Variable
from qiskit_optimization.translators import (
    read_from_lp_file,
    read_from_mps_file,
    write_to_lp_file,
    write_to_mps_file,
)


class TestFileIOTranslator(QiskitOptimizationTestCase):
    """Test Ex/Importers for LP and MPS files (including compressed forms)."""

    def helper_test_read_problem_file(self, q_p: QuadraticProgram):
        """evaluates the quadratic program read in file reading tests"""
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

        qst = q_p.quadratic_constraints
        self.assertEqual(qst[0].name, "quad_eq")
        self.assertDictEqual(qst[0].linear.to_dict(use_name=True), {"x": 1, "y": 1})
        self.assertDictEqual(
            qst[0].quadratic.to_dict(use_name=True),
            {("x", "x"): 1, ("y", "z"): -1, ("z", "z"): 2},
        )
        self.assertEqual(qst[0].sense, Constraint.Sense.EQ)
        self.assertEqual(qst[0].rhs, 1)
        self.assertEqual(qst[1].name, "quad_leq")
        self.assertDictEqual(qst[1].linear.to_dict(use_name=True), {"x": 1, "y": 1})
        self.assertDictEqual(
            qst[1].quadratic.to_dict(use_name=True),
            {("x", "x"): 1, ("y", "z"): -1, ("z", "z"): 2},
        )
        self.assertEqual(qst[1].sense, Constraint.Sense.LE)
        self.assertEqual(qst[1].rhs, 1)
        self.assertEqual(qst[2].name, "quad_geq")
        self.assertDictEqual(qst[2].linear.to_dict(use_name=True), {"x": 1, "y": 1})
        self.assertDictEqual(
            qst[2].quadratic.to_dict(use_name=True),
            {("x", "x"): 1, ("y", "z"): -1, ("z", "z"): 2},
        )
        self.assertEqual(qst[2].sense, Constraint.Sense.GE)
        self.assertEqual(qst[2].rhs, 1)

    def helper_test_write_problem_file(self):
        """creates the quadratic program used in file write tests"""
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

        return q_p

    @unittest.skipIf(not _optionals.HAS_CPLEX, "CPLEX not available.")
    def test_read_from_lp_file(self):
        """test read lp file"""
        try:
            with self.assertRaises(IOError):
                read_from_lp_file("")
            with self.assertRaises(IOError):
                read_from_lp_file("no_file.txt")
            with self.assertRaises(IOError):
                read_from_lp_file("no_file.lp")
            lp_file = self.get_resource_path("test_quadratic_program.lp", "problems/resources")
            q_p = read_from_lp_file(lp_file)

            self.helper_test_read_problem_file(q_p)
        except RuntimeError as ex:
            self.fail(str(ex))

    @unittest.skipIf(not _optionals.HAS_CPLEX, "CPLEX not available.")
    def test_read_from_mps_file(self):
        """test read mps file"""
        try:
            with self.assertRaises(IOError):
                read_from_mps_file("")
            with self.assertRaises(IOError):
                read_from_mps_file("no_file.txt")
            with self.assertRaises(IOError):
                read_from_mps_file("no_file.mps")
            mps_file = self.get_resource_path("test_quadratic_program.mps", "problems/resources")
            q_p = read_from_mps_file(mps_file)

            self.helper_test_read_problem_file(q_p)
        except RuntimeError as ex:
            self.fail(str(ex))

    @unittest.skipIf(not _optionals.HAS_CPLEX, "CPLEX not available.")
    def test_read_from_lp_gz_file(self):
        """test read compressed lp file"""
        try:
            with self.assertRaises(IOError):
                read_from_mps_file("")
            with self.assertRaises(IOError):
                read_from_mps_file("no_file.txt")
            with self.assertRaises(IOError):
                read_from_mps_file("no_file.lp.gz")
            mps_file = self.get_resource_path("test_quadratic_program.lp.gz", "problems/resources")
            q_p = read_from_lp_file(mps_file)

            self.helper_test_read_problem_file(q_p)
        except RuntimeError as ex:
            self.fail(str(ex))

    @unittest.skipIf(not _optionals.HAS_CPLEX, "CPLEX not available.")
    def test_read_from_mps_gz_file(self):
        """test read compressed mps file"""
        try:
            with self.assertRaises(IOError):
                read_from_mps_file("")
            with self.assertRaises(IOError):
                read_from_mps_file("no_file.txt")
            with self.assertRaises(IOError):
                read_from_mps_file("no_file.mps.gz")
            mps_file = self.get_resource_path("test_quadratic_program.mps.gz", "problems/resources")
            q_p = read_from_mps_file(mps_file)

            self.helper_test_read_problem_file(q_p)
        except RuntimeError as ex:
            self.fail(str(ex))

    def test_write_to_lp_file(self):
        """test write problem to lp file"""
        q_p = self.helper_test_write_problem_file()

        reference_file_name = self.get_resource_path(
            "test_quadratic_program.lp", "problems/resources"
        )
        with tempfile.TemporaryDirectory() as tmp:
            temp_output_path = path.join(tmp, "temp.lp")
            write_to_lp_file(q_p, temp_output_path)
            with open(reference_file_name, encoding="utf8") as reference, open(
                temp_output_path, encoding="utf8"
            ) as temp_output_file:
                lines1 = temp_output_file.readlines()
                lines2 = reference.readlines()
                self.assertListEqual(lines1, lines2)

        with tempfile.TemporaryDirectory() as temp_problem_dir:
            write_to_lp_file(q_p, temp_problem_dir)
            with open(path.join(temp_problem_dir, "my_problem.lp"), encoding="utf8") as file1, open(
                reference_file_name, encoding="utf8"
            ) as file2:
                lines1 = file1.readlines()
                lines2 = file2.readlines()
                self.assertListEqual(lines1, lines2)

        with self.assertRaises(OSError):
            write_to_lp_file(q_p, "/cannot/write/this/file.lp")

        with self.assertRaises(DOcplexException):
            write_to_lp_file(q_p, "")

    @unittest.skipIf(not _optionals.HAS_CPLEX, "CPLEX not available.")
    def test_write_to_mps_file(self):
        """test write problem to mps file"""
        q_p = self.helper_test_write_problem_file()

        reference_file_name = self.get_resource_path(
            "test_quadratic_program.mps", "problems/resources"
        )
        with tempfile.TemporaryDirectory() as tmp:
            temp_output_path = path.join(tmp, "temp.mps")
            write_to_mps_file(q_p, temp_output_path)
            with open(reference_file_name, encoding="utf8") as reference, open(
                temp_output_path, encoding="utf8"
            ) as temp_output_file:
                lines1 = temp_output_file.readlines()
                lines2 = reference.readlines()
                self.assertListEqual(lines1, lines2)

        with tempfile.TemporaryDirectory() as temp_problem_dir:
            write_to_mps_file(q_p, temp_problem_dir)
            with open(
                path.join(temp_problem_dir, "my_problem.mps"), encoding="utf8"
            ) as file1, open(reference_file_name, encoding="utf8") as file2:
                lines1 = file1.readlines()
                lines2 = file2.readlines()
                self.assertListEqual(lines1, lines2)

        with self.assertRaises(OSError):
            write_to_mps_file(q_p, "/cannot/write/this/file.mps")

        with self.assertRaises(DOcplexException):
            write_to_mps_file(q_p, "")
