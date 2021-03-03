# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test SetPacking class"""

import random

import networkx as nx

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import (OptimizationResult,
                                            OptimizationResultStatus)
from qiskit_optimization.applications.ising.set_packing import SetPacking
from qiskit_optimization.problems import (Constraint, QuadraticObjective, VarType)
from test.optimization_test_case import QiskitOptimizationTestCase


class TestSetPacking(QiskitOptimizationTestCase):
    """ Test SetPacking class"""

    def setUp(self):
        super().setUp()
        self.total_set = [1, 2, 3, 4, 5]
        self.list_of_subsets = [[1, 2, 3], [2, 3, 4], [4, 5], [1, 3], [2]]
        qp = QuadraticProgram()
        for i in range(5):
            qp.binary_var()
        self.result = OptimizationResult(
            x=[0, 0, 1, 1, 1], fval=3, variables=qp.variables,
            status=OptimizationResultStatus.SUCCESS)

    def test_to_quadratic_program(self):
        """Test to_quadratic_program"""
        set_packing = SetPacking(self.list_of_subsets)
        qp = set_packing.to_quadratic_program()
        # Test name
        self.assertEqual(qp.name, "Set packing")
        # Test variables
        self.assertEqual(qp.get_num_vars(), 5)
        for var in qp.variables:
            self.assertEqual(var.vartype, VarType.BINARY)
        # Test objective
        obj = qp.objective
        self.assertEqual(obj.sense, QuadraticObjective.Sense.MAXIMIZE)
        self.assertEqual(obj.constant, 0)
        self.assertDictEqual(obj.linear.to_dict(), {0: 1, 1: 1, 2: 1, 3: 1, 4: 1})
        self.assertEqual(obj.quadratic.to_dict(), {})
        # Test constraint
        lin = qp.linear_constraints
        self.assertEqual(len(lin), len(self.total_set))
        for i in range(len(lin)):
            self.assertEqual(lin[i].sense, Constraint.Sense.LE)
            self.assertEqual(lin[i].rhs, 1)
            self.assertEqual(lin[i].linear.to_dict(), {
                             j: 1 for j, subset in enumerate(self.list_of_subsets)
                             if i+1 in subset})

    def test_interpret(self):
        """Test interpret"""
        set_packing = SetPacking(self.list_of_subsets)
        self.assertEqual(set_packing.interpret(self.result), [[4, 5], [1, 3], [2]])
