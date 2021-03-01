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

""" Test StableSet class"""

import random

import networkx as nx

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import (OptimizationResult,
                                            OptimizationResultStatus)
from qiskit_optimization.applications.ising.stable_set import StableSet
from qiskit_optimization.problems import (Constraint, QuadraticObjective, VarType)
from test.optimization_test_case import QiskitOptimizationTestCase


class TestStableSet(QiskitOptimizationTestCase):
    """ Test StableSet class"""

    def setUp(self):
        super().setUp()
        self.graph = nx.gnm_random_graph(5, 4, 3)
        qp = QuadraticProgram()
        for i in range(5):
            qp.binary_var()
        self.result = OptimizationResult(
            x=[1, 1, 1, 1, 0], fval=4, variables=qp.variables,
            status=OptimizationResultStatus.SUCCESS)

    def test_to_quadratic_program(self):
        """Test to_quadratic_program"""
        stable_set = StableSet(self.graph)
        qp = stable_set.to_quadratic_program()
        # Test name
        self.assertEqual(qp.name, "Stable set")
        # Test variables
        self.assertEqual(qp.get_num_vars(), 5)
        for var in qp.variables:
            self.assertEqual(var.vartype, VarType.BINARY)
        # Test objective
        obj = qp.objective
        self.assertEqual(obj.sense, QuadraticObjective.Sense.MAXIMIZE)
        self.assertEqual(obj.constant, 0)
        self.assertDictEqual(obj.linear.to_dict(), {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})
        # Test constraint
        lin = qp.linear_constraints
        self.assertEqual(len(lin), len(self.graph.edges))
        for i, edge in enumerate(self.graph.edges):
            self.assertEqual(lin[i].sense, Constraint.Sense.LE)
            self.assertEqual(lin[i].rhs, 1)
            self.assertEqual(lin[i].linear.to_dict(), {edge[0]: 1, edge[1]: 1})

    def test_interpret(self):
        """Test interpret"""
        stable_set = StableSet(self.graph)
        self.assertEqual(stable_set.interpret(self.result), [0, 1, 2, 3])

    def test_node_colors(self):
        stable_set = StableSet(self.graph)
        self.assertEqual(stable_set._node_colors(self.result), ['r', 'r', 'r', 'r', 'darkgrey'])
