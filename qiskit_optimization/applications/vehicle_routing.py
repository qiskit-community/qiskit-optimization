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

"""An application class for the vehicle routing problem."""

import itertools
import random
from typing import List, Dict, Union, Optional

import networkx as nx
import numpy as np
from docplex.mp.model import Model

from qiskit_optimization.algorithms import OptimizationResult
from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from .graph_optimization_application import GraphOptimizationApplication


class VehicleRouting(GraphOptimizationApplication):
    """Optimization application for the "vehicle routing problem" [1] based on a ``NetworkX`` graph.

    References:
        [1]: "Vehicle routing problem", https://en.wikipedia.org/wiki/Vehicle_routing_problem
    """

    def __init__(
        self,
        graph: Union[nx.Graph, np.ndarray, List],
        num_vehicles: int = 2,
        depot: int = 0,
    ) -> None:
        """
        Args:
            graph: A graph representing a vehicle routing problem. It can be specified directly as a
            NetworkX Graph, or as an array or list if format suitable to build out a ``NetworkX``
            graph.
            num_vehicles: The number of vehicles
            depot: The index of the depot node where all the vehicle depart
        """
        super().__init__(graph)
        self._num_vehicles = num_vehicles
        self._depot = depot

    def to_quadratic_program(self) -> QuadraticProgram:
        """Convert a vehicle routing problem instance into a
        :class:`~qiskit_optimization.problems.QuadraticProgram`

        Returns:
            The :class:`~qiskit_optimization.problems.QuadraticProgram` created
            from the vehicle routing problem instance.
        """
        mdl = Model(name="Vehicle routing")
        n = self._graph.number_of_nodes()
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[(i, j)] = mdl.binary_var(name="x_{0}_{1}".format(i, j))
        mdl.minimize(
            mdl.sum(
                self._graph.edges[i, j]["weight"] * x[(i, j)]
                for i in range(n)
                for j in range(n)
                if i != j
            )
        )
        # Only 1 edge goes out from each node
        for i in range(n):
            if i != self.depot:
                mdl.add_constraint(mdl.sum(x[i, j] for j in range(n) if i != j) == 1)
        # Only 1 edge comes into each node
        for j in range(n):
            if j != self.depot:
                mdl.add_constraint(mdl.sum(x[i, j] for i in range(n) if i != j) == 1)
        # For the depot node
        mdl.add_constraint(
            mdl.sum(x[i, self.depot] for i in range(n) if i != self.depot) == self.num_vehicles
        )
        mdl.add_constraint(
            mdl.sum(x[self.depot, j] for j in range(n) if j != self.depot) == self.num_vehicles
        )

        # To eliminate sub-routes
        node_list = [i for i in range(n) if i != self.depot]
        clique_set = []
        for i in range(2, len(node_list) + 1):
            for comb in itertools.combinations(node_list, i):
                clique_set.append(list(comb))
        for clique in clique_set:
            mdl.add_constraint(
                mdl.sum(x[(i, j)] for i in clique for j in clique if i != j) <= len(clique) - 1
            )
        op = from_docplex_mp(mdl)
        return op

    def interpret(self, result: Union[OptimizationResult, np.ndarray]) -> List[List[List[int]]]:
        """Interpret a result as a list of the routes for each vehicle

        Args:
            result : The calculated result of the problem

        Returns:
            A list of the routes for each vehicle
        """
        x = self._result_to_x(result)
        n = self._graph.number_of_nodes()
        idx = 0
        edge_list = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    if x[idx]:
                        edge_list.append([i, j])
                    idx += 1
        route_list = []  # type: List[List[List[int]]]
        for k in range(self.num_vehicles):
            i = 0
            start = self.depot
            route_list.append([])
            while i < len(edge_list):
                if edge_list[i][0] == start:
                    if edge_list[i][1] == self.depot:
                        # If a loop is completed
                        route_list[k].append(edge_list.pop(i))
                        break
                    # Move onto the next edge
                    start = edge_list[i][1]
                    route_list[k].append(edge_list.pop(i))
                    i = 0
                    continue
                i += 1
        if edge_list:
            route_list.append(edge_list)

        return route_list

    def _draw_result(
        self,
        result: Union[OptimizationResult, np.ndarray],
        pos: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        """Draw the result with colors

        Args:
            result: The calculated result for the problem
            pos: The positions of nodes
        """
        import matplotlib.pyplot as plt

        route_list = self.interpret(result)
        nx.draw(self._graph, with_labels=True, pos=pos)
        nx.draw_networkx_edges(
            self._graph,
            pos,
            edgelist=self._edgelist(route_list),
            width=8,
            alpha=0.5,
            edge_color=self._edge_color(route_list),
            edge_cmap=plt.cm.plasma,
        )

    def _edgelist(self, route_list: List[List[List[int]]]):
        # Arrange route_list and return the list of the edges for the edge list of
        # nx.draw_networkx_edges
        return [edge for k in range(len(route_list)) for edge in route_list[k]]

    def _edge_color(self, route_list: List[List[List[int]]]):
        # Arrange route_list and return the list of the colors of each route
        # for edge_color of nx.draw_networkx_edges
        return [k / len(route_list) for k in range(len(route_list)) for edge in route_list[k]]

    @property
    def num_vehicles(self) -> int:
        """Getter of num_vehicles

        Returns:
            The number of the vehicles
        """
        return self._num_vehicles

    @num_vehicles.setter
    def num_vehicles(self, num_vehicles: int) -> None:
        """Setter of num_vehicles

        Args:
            num_vehicles: The number of vehicle
        """
        self._num_vehicles = num_vehicles

    @property
    def depot(self) -> int:
        """Getter of depot

        Returns:
            The node index of the depot where all the vehicles depart
        """
        return self._depot

    @depot.setter
    def depot(self, depot: int) -> None:
        """Setter of depot

        Args:
            depot: The node index of the depot where all the vehicles depart
        """
        self._depot = depot

    @staticmethod
    # pylint: disable=undefined-variable
    def create_random_instance(
        n: int,
        low: int = 0,
        high: int = 100,
        seed: Optional[int] = None,
        num_vehicle: int = 2,
        depot: int = 0,
    ) -> "VehicleRouting":
        """Create a random instance of the vehicle routing problem.

        Args:
            n: the number of nodes.
            low: The minimum value for the coordinate of a node.
            high: The maximum value for the coordinate of a node.
            seed: the seed for the random coordinates.
            num_vehicle: The number of the vehicles
            depot: The index of the depot node where all the vehicle depart

        Returns:
            A VehicleRouting instance created from the input information
        """
        random.seed(seed)
        pos = {i: (random.randint(low, high), random.randint(low, high)) for i in range(n)}
        graph = nx.random_geometric_graph(n, np.hypot(high - low, high - low) + 1, pos=pos)
        for w, v in graph.edges:
            delta = [graph.nodes[w]["pos"][i] - graph.nodes[v]["pos"][i] for i in range(2)]
            graph.edges[w, v]["weight"] = np.rint(np.hypot(delta[0], delta[1]))
        return VehicleRouting(graph, num_vehicle, depot)
