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

""" Test VertexCover class"""

import random

import networkx as nx

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import (OptimizationResult,
                                            OptimizationResultStatus)
from qiskit_optimization.applications.ising.vertex_cover import VertexCover
from qiskit_optimization.problems import (Constraint, QuadraticObjective, VarType)
from test.optimization_test_case import QiskitOptimizationTestCase


class TestVertexCover(QiskitOptimizationTestCase):
    """ Test VertexCover class"""

    def setUp(self):
        super().setUp()
        self.graph = nx.gnm_random_graph(5, 4, 3)
        qp = QuadraticProgram()
        for i in range(5):
            qp.binary_var()
        self.result = OptimizationResult(
            x=[0, 0, 0, 0, 1], fval=1, variables=qp.variables,
            status=OptimizationResultStatus.SUCCESS)

    def test_to_quadratic_program(self):
        """Test to_quadratic_program"""
        vertex_cover = VertexCover(self.graph)
        qp = vertex_cover.to_quadratic_program()
        # Test name
        self.assertEqual(qp.name, "Vertex cover")
        # Test variables
        self.assertEqual(qp.get_num_vars(), 5)
        for var in qp.variables:
            self.assertEqual(var.vartype, VarType.BINARY)
        # Test objective
        obj = qp.objective
        self.assertEqual(obj.sense, QuadraticObjective.Sense.MINIMIZE)
        self.assertEqual(obj.constant, 0)
        self.assertDictEqual(obj.linear.to_dict(), {0: 1, 1: 1, 2: 1, 3: 1, 4: 1})
        self.assertEqual(obj.quadratic.to_dict(), {})
        # Test constraint
        lin = qp.linear_constraints
        self.assertEqual(len(lin), len(self.graph.edges))
        for i, edge in enumerate(self.graph.edges):
            self.assertEqual(lin[i].sense, Constraint.Sense.GE)
            self.assertEqual(lin[i].rhs, 1)
            self.assertEqual(lin[i].linear.to_dict(), {edge[0]: 1, edge[1]: 1})

    def test_interpret(self):
        """Test interpret"""
        vertex_cover = VertexCover(self.graph)
        self.assertEqual(vertex_cover.interpret(self.result), [4])

    def test_node_colors(self):
        vertex_cover = VertexCover(self.graph)
        self.assertEqual(vertex_cover._node_colors(self.result),
                         ['darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'r'])
