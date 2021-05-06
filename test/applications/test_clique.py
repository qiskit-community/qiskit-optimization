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

from test.optimization_test_case import QiskitOptimizationTestCase
import networkx as nx

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import OptimizationResult, OptimizationResultStatus
from qiskit_optimization.applications.clique import Clique
from qiskit_optimization.problems import Constraint, QuadraticObjective, VarType


class TestClique(QiskitOptimizationTestCase):
    """Test Clique class"""

    def setUp(self):
        """Set up for the tests"""
        super().setUp()
        self.graph = nx.gnm_random_graph(5, 8, 123)
        op = QuadraticProgram()
        for _ in range(5):
            op.binary_var()
        self.result = OptimizationResult(
            x=[1, 0, 1, 1, 1],
            fval=4,
            variables=op.variables,
            status=OptimizationResultStatus.SUCCESS,
        )
        self.result_c3 = OptimizationResult(
            x=[1, 0, 1, 1, 0],
            fval=0,
            variables=op.variables,
            status=OptimizationResultStatus.SUCCESS,
        )

    def test_to_quadratic_program(self):
        """Test to_quadratic_program"""
        clique = Clique(self.graph)
        op = clique.to_quadratic_program()
        # Test name
        self.assertEqual(op.name, "Clique")
        # Test variables
        self.assertEqual(op.get_num_vars(), 5)
        for var in op.variables:
            self.assertEqual(var.vartype, VarType.BINARY)
        # Test objective
        obj = op.objective
        self.assertEqual(obj.sense, QuadraticObjective.Sense.MAXIMIZE)
        self.assertEqual(obj.constant, 0)
        self.assertDictEqual(obj.linear.to_dict(), {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})
        self.assertDictEqual(obj.quadratic.to_dict(), {})
        # Test constraint
        lin = op.linear_constraints
        self.assertEqual(len(lin), 2)
        self.assertEqual(lin[0].sense, Constraint.Sense.LE)
        self.assertEqual(lin[0].rhs, 1)
        self.assertEqual(lin[0].linear.to_dict(), {1: 1, 3: 1})
        self.assertEqual(lin[1].sense, Constraint.Sense.LE)
        self.assertEqual(lin[1].rhs, 1)
        self.assertEqual(lin[1].linear.to_dict(), {1: 1.0, 4: 1.0})

    def test_interpret(self):
        """Test interpret"""
        clique = Clique(self.graph)
        self.assertEqual(clique.interpret(self.result), [0, 2, 3, 4])

    def test_node_colors(self):
        """Test _node_colors"""
        clique = Clique(self.graph)
        self.assertEqual(clique._node_colors(self.result), ["r", "darkgrey", "r", "r", "r"])

    def test_size(self):
        """Test size"""
        clique = Clique(self.graph)
        clique.size = 3
        self.assertEqual(clique.size, 3)

    def test_to_quadratic_program_c3(self):
        """Test to_quadratic_program for the clique size 3"""
        clique = Clique(self.graph, 3)
        op = clique.to_quadratic_program()
        # Test name
        self.assertEqual(op.name, "Clique")
        # Test variables
        self.assertEqual(op.get_num_vars(), 5)
        for var in op.variables:
            self.assertEqual(var.vartype, VarType.BINARY)
        # Test objective
        obj = op.objective
        self.assertEqual(obj.sense, QuadraticObjective.Sense.MINIMIZE)
        self.assertEqual(obj.constant, 0)
        self.assertDictEqual(obj.linear.to_dict(), {})
        self.assertDictEqual(obj.quadratic.to_dict(), {})
        # Test constraint
        lin = op.linear_constraints
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

    def test_interpret_c3(self):
        """Test interpret for the clique size 3"""
        clique = Clique(self.graph, 3)
        self.assertEqual(clique.interpret(self.result_c3), [0, 2, 3])

    def test_node_colors_c3(self):
        """Test _node_colors for the clique size 3"""
        clique = Clique(self.graph, 3)
        self.assertEqual(
            clique._node_colors(self.result_c3), ["r", "darkgrey", "r", "r", "darkgrey"]
        )
