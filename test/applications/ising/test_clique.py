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

""" Test Clique class"""

import random

import networkx as nx

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import (OptimizationResult,
                                            OptimizationResultStatus)
from qiskit_optimization.applications.ising.clique import Clique
from qiskit_optimization.problems import (Constraint, QuadraticObjective, VarType)
from test.optimization_test_case import QiskitOptimizationTestCase


class TestClique(QiskitOptimizationTestCase):
    """ Test Clique class"""

    def setUp(self):
        super().setUp()
        self.graph = nx.gnm_random_graph(5, 8, 123)
        qp = QuadraticProgram()
        for i in range(5):
            qp.binary_var()
        self.result = OptimizationResult(
            x=[1, 0, 1, 1, 0], fval=0, variables=qp.variables,
            status=OptimizationResultStatus.SUCCESS)

    def test_to_quadratic_program(self):
        """Test to_quadratic_program"""
        clique = Clique(self.graph)
        qp = clique.to_quadratic_program()
        # Test name
        self.assertEqual(qp.name, "Clique")
        # Test variables
        self.assertEqual(qp.get_num_vars(), 5)
        for var in qp.variables:
            self.assertEqual(var.vartype, VarType.BINARY)
        # Test objective
        obj = qp.objective
        self.assertEqual(obj.sense, QuadraticObjective.Sense.MINIMIZE)
        self.assertEqual(obj.constant, 0)
        self.assertDictEqual(obj.linear.to_dict(), {})
        self.assertDictEqual(obj.quadratic.to_dict(), {})
        # Test constraint
        lin = qp.linear_constraints
        self.assertEqual(len(lin), 3)
        self.assertEqual(lin[0].sense, Constraint.Sense.LE)
        self.assertEqual(lin[0].rhs, 1)
        self.assertEqual(lin[0].linear.to_dict(), {1: 1, 3: 1})
        self.assertEqual(lin[1].sense, Constraint.Sense.LE)
        self.assertEqual(lin[1].rhs, 1)
        self.assertEqual(lin[1].linear.to_dict(), {1: 1.0, 4: 1.0})
        self.assertEqual(lin[2].sense, Constraint.Sense.EQ)
        self.assertEqual(lin[2].rhs, 3)
        self.assertEqual(lin[2].linear.to_dict(), {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})

    def test_interpret(self):
        """Test interpret"""
        clique = Clique(self.graph)
        self.assertEqual(clique.interpret(self.result), [0, 2, 3])

    def test_node_colors(self):
        clique = Clique(self.graph)
        self.assertEqual(clique._node_colors(self.result), ['r', 'darkgrey', 'r', 'r', 'darkgrey'])
