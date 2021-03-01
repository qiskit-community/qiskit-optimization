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
import numpy as np
import networkx as nx

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import (OptimizationResult,
                                            OptimizationResultStatus)
from qiskit_optimization.applications.ising.vehicle_routing import VehicleRouting
from qiskit_optimization.problems import (Constraint, QuadraticObjective, VarType)
from test.optimization_test_case import QiskitOptimizationTestCase


class TestVehicleRouting(QiskitOptimizationTestCase):
    """ Test VehicleRouting class"""

    def setUp(self):
        super().setUp()
        random.seed(600)
        low = 0
        high = 100
        pos = {i: (random.randint(low, high), random.randint(low, high)) for i in range(4)}
        self.graph = nx.random_geometric_graph(4, np.hypot(high-low, high-low)+1, pos=pos)
        for u, v in self.graph.edges:
            delta = [self.graph.nodes[u]['pos'][i] - self.graph.nodes[v]['pos'][i]
                     for i in range(2)]
            self.graph.edges[u, v]['weight'] = np.rint(np.hypot(delta[0], delta[1]))
        qp = QuadraticProgram()
        for i in range(12):
            qp.binary_var()
        self.result = OptimizationResult(
            x=[1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0], fval=184, variables=qp.variables,
            status=OptimizationResultStatus.SUCCESS)

    def test_to_quadratic_program(self):
        """Test to_quadratic_program"""
        vehicle_routing = VehicleRouting(self.graph)
        qp = vehicle_routing.to_quadratic_program(num_vehicle=2)
        # Test name
        self.assertEqual(qp.name, 'Vehicle Routing')
        # Test variables
        self.assertEqual(qp.get_num_vars(), 12)
        for var in qp.variables:
            self.assertEqual(var.vartype, VarType.BINARY)
        # Test objective
        obj = qp.objective
        self.assertEqual(obj.sense, QuadraticObjective.Sense.MINIMIZE)
        self.assertEqual(obj.constant, 0)
        self.assertDictEqual(obj.linear.to_dict(), {0: 49.0, 1: 36.0, 2: 21.0, 3: 49.0, 4: 65.0,
                                                    5: 67.0, 6: 36.0, 7: 65.0, 8: 29.0, 9: 21.0,
                                                    10: 67.0, 11: 29.0})
        self.assertEqual(obj.quadratic.to_dict(), {})
        # Test constraint
        lin = qp.linear_constraints
        self.assertEqual(len(lin), 12)
        for i in range(3):
            self.assertEqual(lin[i].sense, Constraint.Sense.EQ)
            self.assertEqual(lin[i].rhs, 1)
            self.assertEqual(lin[i].linear.to_dict(), {3*(i+1): 1, 3*(i+1)+1: 1, 3*(i+1)+2: 1})
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
        self.assertEqual(vehicle_routing.interpret(self.result, num_vehicle=2),
                         [[[0, 1], [1, 0]], [[0, 2], [2, 3], [3, 0]]])

    def test_edgelist(self):
        vehicle_routing = VehicleRouting(self.graph)
        self.assertEqual(vehicle_routing._edgelist(vehicle_routing.interpret(self.result,
                                                                             num_vehicle=2)),
                         [[0, 1], [1, 0], [0, 2], [2, 3], [3, 0]])

    def test_edge_color(self):
        vehicle_routing = VehicleRouting(self.graph)
        self.assertEqual(vehicle_routing._edge_color(vehicle_routing.interpret(self.result,
                                                                               num_vehicle=2)),
                         [0.0, 0.0, 0.5, 0.5, 0.5])

    def test_random_graph(self):
        vehicle_routing = VehicleRouting.random_graph(n=4, seed=600)
        graph = vehicle_routing.graph()
        for node in graph.nodes:
            self.assertEqual(graph.nodes[node]['pos'], self.graph.nodes[node]['pos'])
        for edge in graph.edges:
            self.assertEqual(graph.edges[edge]['weight'], self.graph.edges[edge]['weight'])
