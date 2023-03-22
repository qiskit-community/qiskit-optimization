# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test bin packing class"""

import unittest
from test.optimization_test_case import QiskitOptimizationTestCase

import qiskit_optimization.optionals as _optionals
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import OptimizationResult, OptimizationResultStatus
from qiskit_optimization.applications.bin_packing import BinPacking
from qiskit_optimization.problems import Constraint, QuadraticObjective, VarType


class TestBinPacking(QiskitOptimizationTestCase):
    """Test Bin packing class"""

    def setUp(self):
        """Set up for the tests"""
        super().setUp()
        self.weights = [16, 9, 23]
        self.max_weight = 40
        self.max_number_of_bins = 2
        op = QuadraticProgram()
        for _ in range(12):
            op.binary_var()
        self.result = OptimizationResult(
            x=[1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1],
            fval=2.0,
            variables=op.variables,
            status=OptimizationResultStatus.SUCCESS,
        )

    def test_to_quadratic_program(self):
        """Test to_quadratic_program"""
        bin_packing = BinPacking(weights=self.weights, max_weight=self.max_weight)
        op = bin_packing.to_quadratic_program()
        # Test name
        self.assertEqual(op.name, "BinPacking")
        # Test variables
        self.assertEqual(op.get_num_vars(), 12)
        for var in op.variables:
            self.assertEqual(var.vartype, VarType.BINARY)
        # Test objective
        obj = op.objective
        self.assertEqual(obj.sense, QuadraticObjective.Sense.MINIMIZE)
        self.assertEqual(obj.constant, 0)
        self.assertDictEqual(obj.linear.to_dict(), {0: 1.0, 1: 1.0, 2: 1.0})
        self.assertEqual(obj.quadratic.to_dict(), {})
        # Test constraint
        lin = op.linear_constraints
        self.assertEqual(len(lin), 6)
        self.assertEqual(lin[0].sense, Constraint.Sense.EQ)
        self.assertEqual(lin[0].rhs, 1)
        self.assertEqual(lin[0].linear.to_dict(), {3: 1.0, 4: 1.0, 5: 1.0})
        self.assertEqual(lin[3].sense, Constraint.Sense.LE)
        self.assertEqual(lin[3].rhs, 0)
        self.assertEqual(lin[3].linear.to_dict(), {3: 16.0, 6: 9.0, 9: 23.0, 0: -40.0})

    def test_interpret(self):
        """Test interpret"""
        bin_packing = BinPacking(weights=self.weights, max_weight=self.max_weight)
        self.assertEqual(bin_packing.interpret(self.result), [[0, 1], [], [2]])

    def test_max_number_of_bins(self):
        """Test a non-default value of max number of bins."""
        bin_packing = BinPacking(
            weights=self.weights,
            max_weight=self.max_weight,
            max_number_of_bins=self.max_number_of_bins,
        )
        op = bin_packing.to_quadratic_program()
        self.assertEqual(op.get_num_vars(), 8)
        # Test objective
        obj = op.objective
        self.assertDictEqual(obj.linear.to_dict(), {0: 1.0, 1: 1.0})
        # Test constraint
        lin = op.linear_constraints
        self.assertEqual(len(lin), 5)
        self.assertEqual(lin[0].linear.to_dict(), {2: 1.0, 3: 1.0})
        self.assertEqual(lin[1].linear.to_dict(), {4: 1.0, 5: 1.0})
        self.assertEqual(lin[2].linear.to_dict(), {6: 1.0, 7: 1.0})
        self.assertEqual(lin[3].linear.to_dict(), {2: 16.0, 4: 9.0, 6: 23.0, 0: -40.0})
        self.assertEqual(lin[4].linear.to_dict(), {3: 16.0, 5: 9.0, 7: 23.0, 1: -40.0})

    @unittest.skipIf(not _optionals.HAS_MATPLOTLIB, "Matplotlib not available.")
    def test_figure(self):
        """Test the plot of the Bin Packing Problem is properly generated."""
        from matplotlib.pyplot import Figure

        bin_packing = BinPacking(
            weights=self.weights,
            max_weight=self.max_weight,
        )
        self.assertIsInstance(bin_packing.get_figure(self.result), Figure)


if __name__ == "__main__":
    unittest.main()
