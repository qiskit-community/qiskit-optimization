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
""" Test VehicleRouting class"""

import random
from test.optimization_test_case import QiskitOptimizationTestCase

import networkx as nx
import numpy as np

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import OptimizationResult, OptimizationResultStatus
from qiskit_optimization.applications.vehicle_routing import VehicleRouting
from qiskit_optimization.problems import Constraint, QuadraticObjective, VarType


class TestVehicleRouting(QiskitOptimizationTestCase):
    """Test VehicleRouting class"""

    def setUp(self):
        super().setUp()
        random.seed(600)
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
        for i in range(12):
            op.binary_var()
        self.result = OptimizationResult(
            x=[1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
            fval=184,
            variables=op.variables,
            status=OptimizationResultStatus.SUCCESS,
        )
        self.result_d2 = OptimizationResult(
            x=[1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1],
            fval=208.0,
            variables=op.variables,
            status=OptimizationResultStatus.SUCCESS,
        )
        self.result_nv3 = OptimizationResult(
            x=[1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            fval=212.0,
            variables=op.variables,
            status=OptimizationResultStatus.SUCCESS,
        )

    def test_to_quadratic_program(self):
        """Test to_quadratic_program"""
        vehicle_routing = VehicleRouting(self.graph)
        op = vehicle_routing.to_quadratic_program()
        # Test name
        self.assertEqual(op.name, "Vehicle routing")
        # Test variables
        self.assertEqual(op.get_num_vars(), 12)
        for var in op.variables:
            self.assertEqual(var.vartype, VarType.BINARY)
        # Test objective
        obj = op.objective
        self.assertEqual(obj.sense, QuadraticObjective.Sense.MINIMIZE)
        self.assertEqual(obj.constant, 0)
        self.assertDictEqual(
            obj.linear.to_dict(),
            {
                0: 49.0,
                1: 36.0,
                2: 21.0,
                3: 49.0,
                4: 65.0,
                5: 67.0,
                6: 36.0,
                7: 65.0,
                8: 29.0,
                9: 21.0,
                10: 67.0,
                11: 29.0,
            },
        )
        self.assertEqual(obj.quadratic.to_dict(), {})
        # Test constraint
        lin = op.linear_constraints
        self.assertEqual(len(lin), 12)
        for i in range(3):
            self.assertEqual(lin[i].sense, Constraint.Sense.EQ)
            self.assertEqual(lin[i].rhs, 1)
            self.assertEqual(
                lin[i].linear.to_dict(),
                {3 * (i + 1): 1, 3 * (i + 1) + 1: 1, 3 * (i + 1) + 2: 1},
            )
        self.assertEqual(lin[3].sense, Constraint.Sense.EQ)
        self.assertEqual(lin[3].rhs, 1)
        self.assertEqual(lin[3].linear.to_dict(), {0: 1, 7: 1, 10: 1})
        self.assertEqual(lin[4].sense, Constraint.Sense.EQ)
        self.assertEqual(lin[4].rhs, 1)
        self.assertEqual(lin[4].linear.to_dict(), {1: 1, 4: 1, 11: 1})
        self.assertEqual(lin[5].sense, Constraint.Sense.EQ)
        self.assertEqual(lin[5].rhs, 1)
        self.assertEqual(lin[5].linear.to_dict(), {2: 1, 5: 1, 8: 1})
        self.assertEqual(lin[6].sense, Constraint.Sense.EQ)
        self.assertEqual(lin[6].rhs, 2)
        self.assertEqual(lin[6].linear.to_dict(), {3: 1, 6: 1, 9: 1})
        self.assertEqual(lin[7].sense, Constraint.Sense.EQ)
        self.assertEqual(lin[7].rhs, 2)
        self.assertEqual(lin[7].linear.to_dict(), {0: 1, 1: 1, 2: 1})
        self.assertEqual(lin[8].sense, Constraint.Sense.LE)
        self.assertEqual(lin[8].rhs, 1)
        self.assertEqual(lin[8].linear.to_dict(), {4: 1, 7: 1})
        self.assertEqual(lin[9].sense, Constraint.Sense.LE)
        self.assertEqual(lin[9].rhs, 1)
        self.assertEqual(lin[9].linear.to_dict(), {5: 1, 10: 1})
        self.assertEqual(lin[10].sense, Constraint.Sense.LE)
        self.assertEqual(lin[10].rhs, 1)
        self.assertEqual(lin[10].linear.to_dict(), {8: 1.0, 11: 1.0})
        self.assertEqual(lin[11].sense, Constraint.Sense.LE)
        self.assertEqual(lin[11].rhs, 2)
        self.assertEqual(lin[11].linear.to_dict(), {4: 1, 5: 1, 7: 1, 8: 1, 10: 1, 11: 1})

    def test_interpret(self):
        """Test interpret"""
        vehicle_routing = VehicleRouting(self.graph)
        self.assertEqual(
            vehicle_routing.interpret(self.result),
            [[[0, 1], [1, 0]], [[0, 2], [2, 3], [3, 0]]],
        )

    def test_edgelist(self):
        """Test _edgelist"""
        vehicle_routing = VehicleRouting(self.graph)
        self.assertEqual(
            vehicle_routing._edgelist(vehicle_routing.interpret(self.result)),
            [[0, 1], [1, 0], [0, 2], [2, 3], [3, 0]],
        )

    def test_edge_color(self):
        """Test _edge_color"""
        vehicle_routing = VehicleRouting(self.graph)
        self.assertEqual(
            vehicle_routing._edge_color(vehicle_routing.interpret(self.result)),
            [0.0, 0.0, 0.5, 0.5, 0.5],
        )

    def test_to_quadratic_program_d2(self):
        """Test to_quadratic_program for depot=2"""
        vehicle_routing = VehicleRouting(self.graph, depot=2)
        op = vehicle_routing.to_quadratic_program()
        # Test name
        self.assertEqual(op.name, "Vehicle routing")
        # Test variables
        self.assertEqual(op.get_num_vars(), 12)
        for var in op.variables:
            self.assertEqual(var.vartype, VarType.BINARY)
        # Test objective
        obj = op.objective
        self.assertEqual(obj.sense, QuadraticObjective.Sense.MINIMIZE)
        self.assertEqual(obj.constant, 0)
        self.assertDictEqual(
            obj.linear.to_dict(),
            {
                0: 49.0,
                1: 36.0,
                2: 21.0,
                3: 49.0,
                4: 65.0,
                5: 67.0,
                6: 36.0,
                7: 65.0,
                8: 29.0,
                9: 21.0,
                10: 67.0,
                11: 29.0,
            },
        )
        self.assertEqual(obj.quadratic.to_dict(), {})
        # Test constraint
        lin = op.linear_constraints
        self.assertEqual(len(lin), 12)
        c012 = [-1, 0, 2]
        for i in range(3):
            j = c012[i]
            self.assertEqual(lin[i].sense, Constraint.Sense.EQ)
            self.assertEqual(lin[i].rhs, 1)
            self.assertEqual(
                lin[i].linear.to_dict(),
                {3 * (j + 1): 1, 3 * (j + 1) + 1: 1, 3 * (j + 1) + 2: 1},
            )
        self.assertEqual(lin[3].sense, Constraint.Sense.EQ)
        self.assertEqual(lin[3].rhs, 1)
        self.assertEqual(lin[3].linear.to_dict(), {3: 1, 6: 1, 9: 1})
        self.assertEqual(lin[4].sense, Constraint.Sense.EQ)
        self.assertEqual(lin[4].rhs, 1)
        self.assertEqual(lin[4].linear.to_dict(), {0: 1, 7: 1, 10: 1})
        self.assertEqual(lin[5].sense, Constraint.Sense.EQ)
        self.assertEqual(lin[5].rhs, 1)
        self.assertEqual(lin[5].linear.to_dict(), {2: 1, 5: 1, 8: 1})
        self.assertEqual(lin[6].sense, Constraint.Sense.EQ)
        self.assertEqual(lin[6].rhs, 2)
        self.assertEqual(lin[6].linear.to_dict(), {1: 1, 4: 1, 11: 1})
        self.assertEqual(lin[7].sense, Constraint.Sense.EQ)
        self.assertEqual(lin[7].rhs, 2)
        self.assertEqual(lin[7].linear.to_dict(), {6: 1, 7: 1, 8: 1})
        self.assertEqual(lin[8].sense, Constraint.Sense.LE)
        self.assertEqual(lin[8].rhs, 1)
        self.assertEqual(lin[8].linear.to_dict(), {0: 1, 3: 1})
        self.assertEqual(lin[9].sense, Constraint.Sense.LE)
        self.assertEqual(lin[9].rhs, 1)
        self.assertEqual(lin[9].linear.to_dict(), {2: 1, 9: 1})
        self.assertEqual(lin[10].sense, Constraint.Sense.LE)
        self.assertEqual(lin[10].rhs, 1)
        self.assertEqual(lin[10].linear.to_dict(), {5: 1.0, 10: 1.0})
        self.assertEqual(lin[11].sense, Constraint.Sense.LE)
        self.assertEqual(lin[11].rhs, 2)
        self.assertEqual(lin[11].linear.to_dict(), {0: 1, 2: 1, 3: 1, 5: 1, 9: 1, 10: 1})

    def test_interpret_d2(self):
        """Test interpret for depot=2"""
        vehicle_routing = VehicleRouting(self.graph, depot=2)
        self.assertEqual(
            vehicle_routing.interpret(self.result_d2),
            [[[2, 0], [0, 1], [1, 2]], [[2, 3], [3, 2]]],
        )

    def test_edgelist_d2(self):
        """Test _edgelist for depot=2"""
        vehicle_routing = VehicleRouting(self.graph, depot=2)
        self.assertEqual(
            vehicle_routing._edgelist(vehicle_routing.interpret(self.result_d2)),
            [[2, 0], [0, 1], [1, 2], [2, 3], [3, 2]],
        )

    def test_edge_color_d2(self):
        """Test _edge_color for depot=2"""
        vehicle_routing = VehicleRouting(self.graph, depot=2)
        self.assertEqual(
            vehicle_routing._edge_color(vehicle_routing.interpret(self.result_d2)),
            [0.0, 0.0, 0.0, 0.5, 0.5],
        )

    def test_to_quadratic_program_nv3(self):
        """Test to_quadratic_program for num_vehicles=3"""
        vehicle_routing = VehicleRouting(self.graph, num_vehicles=3)
        op = vehicle_routing.to_quadratic_program()
        # Test name
        self.assertEqual(op.name, "Vehicle routing")
        # Test variables
        self.assertEqual(op.get_num_vars(), 12)
        for var in op.variables:
            self.assertEqual(var.vartype, VarType.BINARY)
        # Test objective
        obj = op.objective
        self.assertEqual(obj.sense, QuadraticObjective.Sense.MINIMIZE)
        self.assertEqual(obj.constant, 0)
        self.assertDictEqual(
            obj.linear.to_dict(),
            {
                0: 49.0,
                1: 36.0,
                2: 21.0,
                3: 49.0,
                4: 65.0,
                5: 67.0,
                6: 36.0,
                7: 65.0,
                8: 29.0,
                9: 21.0,
                10: 67.0,
                11: 29.0,
            },
        )
        self.assertEqual(obj.quadratic.to_dict(), {})
        # Test constraint
        lin = op.linear_constraints
        self.assertEqual(len(lin), 12)
        for i in range(3):
            self.assertEqual(lin[i].sense, Constraint.Sense.EQ)
            self.assertEqual(lin[i].rhs, 1)
            self.assertEqual(
                lin[i].linear.to_dict(),
                {3 * (i + 1): 1, 3 * (i + 1) + 1: 1, 3 * (i + 1) + 2: 1},
            )
        self.assertEqual(lin[3].sense, Constraint.Sense.EQ)
        self.assertEqual(lin[3].rhs, 1)
        self.assertEqual(lin[3].linear.to_dict(), {0: 1, 7: 1, 10: 1})
        self.assertEqual(lin[4].sense, Constraint.Sense.EQ)
        self.assertEqual(lin[4].rhs, 1)
        self.assertEqual(lin[4].linear.to_dict(), {1: 1, 4: 1, 11: 1})
        self.assertEqual(lin[5].sense, Constraint.Sense.EQ)
        self.assertEqual(lin[5].rhs, 1)
        self.assertEqual(lin[5].linear.to_dict(), {2: 1, 5: 1, 8: 1})
        self.assertEqual(lin[6].sense, Constraint.Sense.EQ)
        self.assertEqual(lin[6].rhs, 3)
        self.assertEqual(lin[6].linear.to_dict(), {3: 1, 6: 1, 9: 1})
        self.assertEqual(lin[7].sense, Constraint.Sense.EQ)
        self.assertEqual(lin[7].rhs, 3)
        self.assertEqual(lin[7].linear.to_dict(), {0: 1, 1: 1, 2: 1})
        self.assertEqual(lin[8].sense, Constraint.Sense.LE)
        self.assertEqual(lin[8].rhs, 1)
        self.assertEqual(lin[8].linear.to_dict(), {4: 1, 7: 1})
        self.assertEqual(lin[9].sense, Constraint.Sense.LE)
        self.assertEqual(lin[9].rhs, 1)
        self.assertEqual(lin[9].linear.to_dict(), {5: 1, 10: 1})
        self.assertEqual(lin[10].sense, Constraint.Sense.LE)
        self.assertEqual(lin[10].rhs, 1)
        self.assertEqual(lin[10].linear.to_dict(), {8: 1.0, 11: 1.0})
        self.assertEqual(lin[11].sense, Constraint.Sense.LE)
        self.assertEqual(lin[11].rhs, 2)
        self.assertEqual(lin[11].linear.to_dict(), {4: 1, 5: 1, 7: 1, 8: 1, 10: 1, 11: 1})

    def test_interpret_nv3(self):
        """Test interpret for num_vehicles=3"""
        vehicle_routing = VehicleRouting(self.graph, num_vehicles=3)
        self.assertEqual(
            vehicle_routing.interpret(self.result_nv3),
            [[[0, 1], [1, 0]], [[0, 2], [2, 0]], [[0, 3], [3, 0]]],
        )

    def test_edgelist_nv3(self):
        """Test _edgelist for num_vehicles=3"""
        vehicle_routing = VehicleRouting(self.graph, num_vehicles=3)
        self.assertEqual(
            vehicle_routing._edgelist(vehicle_routing.interpret(self.result_nv3)),
            [[0, 1], [1, 0], [0, 2], [2, 0], [0, 3], [3, 0]],
        )

    def test_edge_color_nv3(self):
        """Test _edge_color for num_vehicles=3"""
        vehicle_routing = VehicleRouting(self.graph, num_vehicles=3)
        self.assertEqual(
            vehicle_routing._edge_color(vehicle_routing.interpret(self.result_nv3)),
            [0.0, 0.0, 1 / 3, 1 / 3, 2 / 3, 2 / 3],
        )

    def test_create_random_instance(self):
        """Test create_random_instance"""
        vehicle_routing = VehicleRouting.create_random_instance(n=4, seed=600)
        graph = vehicle_routing.graph
        for node in graph.nodes:
            self.assertEqual(graph.nodes[node]["pos"], self.graph.nodes[node]["pos"])
        for edge in graph.edges:
            self.assertEqual(graph.edges[edge]["weight"], self.graph.edges[edge]["weight"])

    def test_num_vehicles(self):
        """Test num_vehicles"""
        vehicle_routing = VehicleRouting(self.graph, num_vehicles=2)
        vehicle_routing.num_vehicles = 5
        self.assertEqual(vehicle_routing.num_vehicles, 5)

    def test_depot(self):
        """Test depot"""
        vehicle_routing = VehicleRouting(self.graph, depot=0)
        vehicle_routing.depot = 2
        self.assertEqual(vehicle_routing.depot, 2)
