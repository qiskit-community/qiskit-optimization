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

""" Test Knapsack class"""

import random

import networkx as nx

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import (OptimizationResult,
                                            OptimizationResultStatus)
from qiskit_optimization.applications.ising.knapsack import Knapsack
from qiskit_optimization.problems import (Constraint, QuadraticObjective, VarType)
from test.optimization_test_case import QiskitOptimizationTestCase


class TestKnapsack(QiskitOptimizationTestCase):
    """ Test Knapsack class"""

    def setUp(self):
        super().setUp()
        self.values = [10, 40, 30, 50]
        self.weights = [5, 4, 6, 3]
        self.max_weight = 10
        qp = QuadraticProgram()
        for i in range(4):
            qp.binary_var()
        self.result = OptimizationResult(
            x=[0, 1, 0, 1], fval=90, variables=qp.variables,
            status=OptimizationResultStatus.SUCCESS)

    def test_to_quadratic_program(self):
        """Test to_quadratic_program"""
        knapsack = Knapsack(values=self.values, weights=self.weights, max_weight=self.max_weight)
        qp = knapsack.to_quadratic_program()
        # Test name
        self.assertEqual(qp.name, "K napsack")
        # Test variables
        self.assertEqual(qp.get_num_vars(), 4)
        for var in qp.variables:
            self.assertEqual(var.vartype, VarType.BINARY)
        # Test objective
        obj = qp.objective
        self.assertEqual(obj.sense, QuadraticObjective.Sense.MAXIMIZE)
        self.assertEqual(obj.constant, 0)
        self.assertDictEqual(obj.linear.to_dict(), {0: 10, 1: 40, 2: 30, 3: 50})
        self.assertEqual(obj.quadratic.to_dict(), {})
        # Test constraint
        lin = qp.linear_constraints
        self.assertEqual(len(lin), 1)
        self.assertEqual(lin[0].sense, Constraint.Sense.LE)
        self.assertEqual(lin[0].rhs, self.max_weight)
        self.assertEqual(lin[0].linear.to_dict(), {
                         i: weight for i, weight in enumerate(self.weights)})

    def test_interpret(self):
        """Test interpret"""
        knapsack = Knapsack(values=self.values, weights=self.weights, max_weight=self.max_weight)
        self.assertEqual(knapsack.interpret(self.result), [1, 3])
