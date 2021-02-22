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

"""
Convert vertex cover instances into Pauli list
Deal with Gset format. See https://web.stanford.edu/~yyye/yyye/Gset/
"""
import itertools
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from docplex.mp.model import Model

from .graph_application import GraphApplication
from qiskit_optimization.problems.quadratic_program import QuadraticProgram


class VehicleRouting(GraphApplication):

    def __init__(self, graph):
        super().__init__(graph)

    def to_quadratic_program(self, num_vehicle, depot=0):
        mdl = Model(name='vehicle_routing')
        n = self._graph.number_of_nodes()
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[(i, j)] = mdl.binary_var(name='x_{0}_{1}'.format(i, j))
        mdl.minimize(mdl.sum(self._graph.edges[i, j]['weight']*x[(i, j)]
                             for i in range(n) for j in range(n) if i != j))
        # Only 1 edge goes out from each node
        for i in range(n):
            if i != depot:
                mdl.add_constraint(mdl.sum(x[i, j] for j in range(n) if i != j) == 1)
        # Only 1 edge comes into each node
        for j in range(n):
            if j != depot:
                mdl.add_constraint(mdl.sum(x[i, j] for i in range(n) if i != j) == 1)
        # For the depot node
        mdl.add_constraint(mdl.sum(x[i, depot] for i in range(n) if i != depot) == num_vehicle)
        mdl.add_constraint(mdl.sum(x[depot, j] for j in range(n) if j != depot) == num_vehicle)

        # To eliminate sub-routes
        node_list = [i for i in range(n) if i != depot]
        clique_set = []
        for i in range(2, len(node_list)+1):
            for comb in itertools.combinations(node_list, i):
                clique_set.append(list(comb))
        for clique in clique_set:
            mdl.add_constraint(mdl.sum(x[(i, j)]
                                       for i in clique
                                       for j in clique if i != j) <= len(clique) - 1)
        qp = QuadraticProgram()
        qp.from_docplex(mdl)
        return qp

    def interpret(self, result, num_vehicle, depot=0):
        n = self._graph.number_of_nodes()
        idx = 0
        edge_list = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    if result.x[idx]:
                        edge_list.append([i, j])
                    idx += 1
        route_list = []
        for k in range(num_vehicle):
            i = 0
            start = depot
            route_list.append([])
            while(i < len(edge_list)):
                if edge_list[i][0] == start:
                    if edge_list[i][1] == 0:
                        # If a loop is completed
                        route_list[k].append(edge_list.pop(i))
                        break
                    # Move onto the next edge
                    start = edge_list[i][1]
                    route_list[k].append(edge_list.pop(i))
                    i = 0
                    continue
                i += 1
        if len(edge_list):
            route_list.append(edge_list)

        return route_list

    def draw_graph(self, result, num_vehicle, depot=0, pos=None):
        route_list = self.interpret(result, num_vehicle, depot)
        len_list = len(route_list)
        nx.draw(self._graph, with_labels=True, pos=pos)
        color_list = [k/len_list for k in range(len_list) for edge in route_list[k]]
        route_list = [edge for k in range(len_list) for edge in route_list[k]]
        nx.draw_networkx_edges(
            self._graph,
            pos,
            edgelist=route_list,
            width=8, alpha=0.5, edge_color=color_list, edge_cmap=plt.cm.plasma
            )

    @staticmethod
    def random_graph(n, low=0, high=100, seed=None):
        random.seed(seed)
        pos = {i: (random.randint(low, high), random.randint(low, high)) for i in range(n)}
        g = nx.random_geometric_graph(n, np.hypot(high-low, high-low)+1, pos=pos)
        for u, v in g.edges:
            delta = [g.nodes[u]['pos'][i] - g.nodes[v]['pos'][i] for i in range(2)]
            g.edges[u, v]['weight'] = np.rint(np.hypot(delta[0], delta[1]))
        return VehicleRouting(g)
