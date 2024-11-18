# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Tsp class"""
import unittest
import random
from test.optimization_test_case import QiskitOptimizationTestCase

import numpy as np
import networkx as nx

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import OptimizationResult, OptimizationResultStatus
from qiskit_optimization.applications.tsp import Tsp
from qiskit_optimization.problems import Constraint, QuadraticObjective, VarType


class TestTsp(QiskitOptimizationTestCase):
    """Test Tsp class"""

    def setUp(self):
        super().setUp()
        random.seed(123)
        low = 0
        high = 100
        pos = {i: (random.randint(low, high), random.randint(low, high)) for i in range(4)}
        self.graph = nx.random_geometric_graph(4, np.hypot(high - low, high - low) + 1, pos=pos)
        for w, v in self.graph.edges:
            delta = [
                self.graph.nodes[w]["pos"][i] - self.graph.nodes[v]["pos"][i] for i in range(2)
            ]
            self.graph.edges[w, v]["weight"] = np.rint(np.hypot(delta[0], delta[1]))

        op = QuadraticProgram()
        for i in range(16):
            op.binary_var()
        self.result = OptimizationResult(
            x=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            fval=272,
            variables=op.variables,
            status=OptimizationResultStatus.SUCCESS,
        )

    def test_to_quadratic_program(self):
        """Test to_quadratic_program"""
        tsp = Tsp(self.graph)
        op = tsp.to_quadratic_program()
        # Test name
        self.assertEqual(op.name, "TSP")
        # Test variables
        self.assertEqual(op.get_num_vars(), 16)
        for var in op.variables:
            self.assertEqual(var.vartype, VarType.BINARY)
        # Test objective
        obj = op.objective
        self.assertEqual(obj.sense, QuadraticObjective.Sense.MINIMIZE)
        self.assertEqual(obj.constant, 0)
        self.assertDictEqual(obj.linear.to_dict(), {})
        for edge, val in obj.quadratic.to_dict().items():
            self.assertEqual(val, self.graph.edges[edge[0] // 4, edge[1] // 4]["weight"])

        # Test constraint
        lin = op.linear_constraints
        self.assertEqual(len(lin), 8)
        for i in range(4):
            self.assertEqual(lin[i].sense, Constraint.Sense.EQ)
            self.assertEqual(lin[i].rhs, 1)
            self.assertEqual(
                lin[i].linear.to_dict(),
                {4 * i: 1, 4 * i + 1: 1, 4 * i + 2: 1, 4 * i + 3: 1},
            )
        for i in range(4):
            self.assertEqual(lin[4 + i].sense, Constraint.Sense.EQ)
            self.assertEqual(lin[4 + i].rhs, 1)
            self.assertEqual(lin[4 + i].linear.to_dict(), {i: 1, 4 + i: 1, 8 + i: 1, 12 + i: 1})

    def test_interpret(self):
        """Test interpret"""
        tsp = Tsp(self.graph)
        self.assertEqual(tsp.interpret(self.result), [0, 1, 2, 3])

    def test_edgelist(self):
        """Test _edgelist"""
        tsp = Tsp(self.graph)
        self.assertEqual(tsp._edgelist(self.result), [(0, 1), (1, 2), (2, 3), (3, 0)])

    def test_create_random_instance(self):
        """Test create_random_instance"""
        tsp = Tsp.create_random_instance(n=3, seed=123)
        graph = tsp.graph
        edge_weight = [graph.edges[edge]["weight"] for edge in graph.edges]
        expected_weight = [48, 91, 63]
        self.assertEqual(edge_weight, expected_weight)

    def test_parse_tsplib_format(self):
        """Test tsplib format parser"""
        # test_tsplib is eli51.tsp from http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
        reference_file_name = self.get_resource_path("test_tsplib.tsp", "applications/resources")
        tsp = Tsp.parse_tsplib_format(reference_file_name)
        graph = tsp.graph
        self.assertEqual(graph.number_of_nodes(), 51)
        self.assertEqual(graph.number_of_edges(), 51 * 50 / 2)  # fully connected graph


class TestTspCustomGraph(QiskitOptimizationTestCase):
    """Test Tsp class with a custom non-geometric graph"""

    def setUp(self):
        """Set up test cases."""
        super().setUp()
        self.graph = nx.Graph()
        self.edges_with_weights = [
            (0, 1, 5),
            (1, 2, 5),
            (1, 3, 15),
            (2, 3, 15),
            (2, 4, 5),
            (3, 4, 5),
            (3, 0, 5),
        ]

        self.graph.add_nodes_from(range(5))
        for source, target, weight in self.edges_with_weights:
            self.graph.add_edge(source, target, weight=weight)

        op = QuadraticProgram()
        for _ in range(25):
            op.binary_var()

        result_vector = np.zeros(25)
        result_vector[0] = 1
        result_vector[6] = 1
        result_vector[12] = 1
        result_vector[23] = 1
        result_vector[19] = 1

        self.optimal_path = [0, 1, 2, 4, 3]
        self.optimal_edges = [(0, 1), (1, 2), (2, 4), (4, 3), (3, 0)]
        self.optimal_cost = 25

        self.result = OptimizationResult(
            x=result_vector,
            fval=self.optimal_cost,
            variables=op.variables,
            status=OptimizationResultStatus.SUCCESS,
        )

    def test_to_quadratic_program(self):
        """Test to_quadratic_program with custom graph"""
        tsp = Tsp(self.graph)
        quadratic_program = tsp.to_quadratic_program()

        self.assertEqual(quadratic_program.name, "TSP")
        self.assertEqual(quadratic_program.get_num_vars(), 25)

        for variable in quadratic_program.variables:
            self.assertEqual(variable.vartype, VarType.BINARY)

        objective = quadratic_program.objective
        self.assertEqual(objective.constant, 0)
        self.assertEqual(objective.sense, QuadraticObjective.Sense.MINIMIZE)

        # Test objective quadratic terms
        quadratic_terms = objective.quadratic.to_dict()
        for source, target, weight in self.edges_with_weights:
            for position in range(5):
                next_position = (position + 1) % 5
                key = (
                    min(source * 5 + position, target * 5 + next_position),
                    max(source * 5 + position, target * 5 + next_position),
                )
                self.assertIn(key, quadratic_terms)
                self.assertEqual(quadratic_terms[key], weight)

        linear_constraints = quadratic_program.linear_constraints

        # Test node constraints (each node appears once)
        for node in range(5):
            self.assertEqual(linear_constraints[node].sense, Constraint.Sense.EQ)
            self.assertEqual(linear_constraints[node].rhs, 1)
            self.assertEqual(
                linear_constraints[node].linear.to_dict(),
                {5 * node + pos: 1 for pos in range(5)},
            )

        # Test position constraints (each position filled once)
        for position in range(5):
            self.assertEqual(linear_constraints[5 + position].sense, Constraint.Sense.EQ)
            self.assertEqual(linear_constraints[5 + position].rhs, 1)
            self.assertEqual(
                linear_constraints[5 + position].linear.to_dict(),
                {5 * node + position: 1 for node in range(5)},
            )

        # Test non-edge constraints
        non_edges = list(nx.non_edges(self.graph))
        constraint_idx = 10  # Start after node and position constraints

        for i, j in non_edges:
            for k in range(5):
                next_k = (k + 1) % 5

                # Check forward constraint: x[i,k] + x[j,(k+1)%n] <= 1
                constraint = linear_constraints[constraint_idx]
                self.assertEqual(constraint.sense, Constraint.Sense.LE)
                self.assertEqual(constraint.rhs, 1)
                linear_dict = constraint.linear.to_dict()
                self.assertEqual(len(linear_dict), 2)
                self.assertEqual(linear_dict[i * 5 + k], 1)
                self.assertEqual(linear_dict[j * 5 + next_k], 1)
                constraint_idx += 1

                # Check backward constraint: x[j,k] + x[i,(k+1)%n] <= 1
                constraint = linear_constraints[constraint_idx]
                self.assertEqual(constraint.sense, Constraint.Sense.LE)
                self.assertEqual(constraint.rhs, 1)
                linear_dict = constraint.linear.to_dict()
                self.assertEqual(len(linear_dict), 2)
                self.assertEqual(linear_dict[j * 5 + k], 1)
                self.assertEqual(linear_dict[i * 5 + next_k], 1)
                constraint_idx += 1

        # Verify total number of constraints
        expected_constraints = (
            5  # node constraints
            + 5  # position constraints
            + len(non_edges) * 2 * 5  # non-edge constraints (2 per non-edge per position)
        )
        self.assertEqual(len(linear_constraints), expected_constraints)

    def test_interpret(self):
        """Test interpret with custom graph"""
        tsp = Tsp(self.graph)
        self.assertEqual(tsp.interpret(self.result), self.optimal_path)

    def test_edgelist(self):
        """Test _edgelist with custom graph"""
        tsp = Tsp(self.graph)
        self.assertEqual(tsp._edgelist(self.result), self.optimal_edges)


if __name__ == "__main__":
    unittest.main()
