import random

import networkx as nx
import numpy as np
from docplex.mp.model import Model

from .graph_application import GraphApplication
from qiskit_optimization.problems.quadratic_program import QuadraticProgram


class TSP(GraphApplication):

    def __init__(self, graph):
        super().__init__(graph)

    def to_quadratic_program(self):
        mdl = Model(name='tsp')
        n = self._graph.number_of_nodes()
        x = {(i, p): mdl.binary_var(name='x_{0}_{1}'.format(i, p))
             for i in range(n) for p in range(n)}
        tsp_func = mdl.sum(self._graph.edges[i, j]['weight'] * x[(i, p)] * x[(j, (p+1) % n)]
                           for i in range(n) for j in range(n) for p in range(n) if i != j)
        mdl.minimize(tsp_func)
        for i in range(n):
            mdl.add_constraint(mdl.sum(x[(i, p)] for p in range(n)) == 1)
        for p in range(n):
            mdl.add_constraint(mdl.sum(x[(i, p)] for i in range(n)) == 1)
        qp = QuadraticProgram()
        qp.from_docplex(mdl)
        return qp

    def draw_graph(self, result, pos=None):
        route = self.interpret(result)
        nx.draw(self._graph, with_labels=True, pos=pos)
        nx.draw_networkx_edges(
            self._graph,
            pos,
            edgelist=[(route[i], route[(i+1) % len(route)]) for i in range(len(route))],
            width=8, alpha=0.5, edge_color="tab:red",
            )

    def interpret(self, result):
        n = int(np.sqrt(len(result.x)))
        route = []
        for p__ in range(n):
            p_step = []
            for i in range(n):
                if result.x[i * n + p__]:
                    p_step.append(i)
            if len(p_step) == 1:
                route.extend(p_step)
            else:
                route.append(p_step)
        return route

    @staticmethod
    def random_graph(n, low=0, high=100, seed=None):
        random.seed(seed)
        pos = {i: (random.randint(low, high), random.randint(low, high)) for i in range(n)}
        g = nx.random_geometric_graph(n, np.hypot(high-low, high-low)+1, pos=pos)
        for u, v in g.edges:
            delta = [g.nodes[u]['pos'][i] - g.nodes[v]['pos'][i] for i in range(2)]
            g.edges[u, v]['weight'] = np.rint(np.hypot(delta[0], delta[1]))
        return TSP(g)

    @staticmethod
    def parse_tsplib_format(filename):
        """Read graph in TSPLIB format from file.

        Args:
            filename (str): name of the file.

        Returns:
            TspData: instance data.

        """
        name = ''
        coord = []
        with open(filename) as infile:
            coord_section = False
            for line in infile:
                if line.startswith('NAME'):
                    name = line.split(':')[1]
                    name.strip()
                elif line.startswith('TYPE'):
                    typ = line.split(':')[1]
                    typ.strip()
                    if typ != 'TSP':
                        raise QiskitOptimizationError(
                            'This supports only "TSP" type. Actual: %s', typ)
                elif line.startswith('DIMENSION'):
                    dim = int(line.split(':')[1])
                    coord = np.zeros((dim, 2))
                elif line.startswith('EDGE_WEIGHT_TYPE'):
                    typ = line.split(':')[1]
                    typ.strip()
                    if typ != 'EUC_2D':
                        raise QiskitOptimizationError(
                            'This supports only "EUC_2D" edge weight. Actual: %s', typ)
                elif line.startswith('NODE_COORD_SECTION'):
                    coord_section = True
                elif coord_section:
                    v = line.split()
                    index = int(v[0]) - 1
                    coord[index][0] = float(v[1])
                    coord[index][1] = float(v[2])

        x_max = max(coord_[0] for coord_ in coord)
        x_min = min(coord_[0] for coord_ in coord)
        y_max = max(coord_[1] for coord_ in coord)
        y_min = min(coord_[1] for coord_ in coord)

        g = nx.random_geometric_graph(n, np.hypot(x_max-x_min, y_max-y_min)+1, pos=coord)
        for u, v in g.edges:
            delta = [g.nodes[u]['pos'][i] - g.nodes[v]['pos'][i] for i in range(2)]
            g.edges[u, v]['weight'] = np.rint(np.hypot(delta[0], delta[1]))
        return TSP(g)
