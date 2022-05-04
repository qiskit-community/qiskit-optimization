# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Constraint """

from test.optimization_test_case import QiskitOptimizationTestCase

from qiskit_optimization.problems import Constraint, QuadraticProgram
from qiskit_optimization.problems.constraint import ConstraintSense


class TestConstraintSense(QiskitOptimizationTestCase):
    """Test ConstraintSense."""

    def setUp(self) -> None:
        super().setUp()
        self._sense = {
            ConstraintSense.EQ: ["==", "=", "E", "EQ"],
            ConstraintSense.LE: ["<=", "<", "L", "LE"],
            ConstraintSense.GE: [">=", ">", "G", "GE"],
        }

    def test_convert(self):
        """test convert."""
        for sense, lst in self._sense.items():
            for label in lst:
                self.assertEqual(ConstraintSense.convert(label), sense)

    def test_label(self):
        """test label."""
        for sense, lst in self._sense.items():
            self.assertEqual(sense.label, lst[0])


class TestConstraint(QiskitOptimizationTestCase):
    """Test Constraint"""

    def test_init(self):
        """test init"""
        quadratic_program = QuadraticProgram("test")
        for sense in [ConstraintSense.GE, ConstraintSense.LE, ConstraintSense.EQ]:
            cst = Constraint(quadratic_program, "name", sense, 1.0)
            self.assertEqual(cst.name, "name")
            self.assertEqual(cst.sense, sense)
            self.assertAlmostEqual(cst.rhs, 1.0)
            self.assertEqual(cst.quadratic_program.name, "test")
