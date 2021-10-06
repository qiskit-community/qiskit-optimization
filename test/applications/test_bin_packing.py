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

"""Test bin packing class"""
from test.optimization_test_case import QiskitOptimizationTestCase

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
        op = QuadraticProgram()
        for _ in range(12):
            op.binary_var()
        self.result = OptimizationResult(
            x=[1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
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
        self.assertEqual(lin[0].linear.to_dict(), {3: 1.0, 6: 1.0, 9: 1.0})
        self.assertEqual(lin[3].sense, Constraint.Sense.LE)
        self.assertEqual(lin[3].rhs, 0)
        self.assertEqual(lin[3].linear.to_dict(), {3: 16.0, 4: 9.0, 5: 23.0, 0: -40.0})

    def test_interpret(self):
        """Test interpret"""
        bin_packing = BinPacking(weights=self.weights, max_weight=self.max_weight)
        self.assertEqual(bin_packing.interpret(self.result), [0, 2, 3, 4, 11])

    def test_max_weight(self):
        """Test max_weight"""
        bin_packing = BinPacking(weights=self.weights, max_weight=self.max_weight)
        bin_packing.max_weight = 5
        self.assertEqual(bin_packing.max_weight, 5)
