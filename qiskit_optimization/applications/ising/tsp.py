import random

import networkx as nx
import numpy as np
from docplex.mp.model import Model

from .graph_application import GraphApplication
from qiskit_optimization.problems.quadratic_program import QuadraticProgram


class TSP(GraphApplication):

    def __init__(self, g):
        super().__init__(g)

    def _build_quadratic_program(self):
        mdl = Model(name='tsp')
        n = self._g.number_of_nodes()
        x = {(i, p): mdl.binary_var(name='x_{0}_{1}'.format(i, p))
             for i in range(n) for p in range(n)}
        tsp_func = mdl.sum(self._g.edges[i, j]['weight'] * x[(i, p)] * x[(j, (p+1) % n)]
                           for i in range(n) for j in range(n) for p in range(n) if i != j)
        mdl.minimize(tsp_func)
        for i in range(n):
            mdl.add_constraint(mdl.sum(x[(i, p)] for p in range(n)) == 1)
        for p in range(n):
            mdl.add_constraint(mdl.sum(x[(i, p)] for i in range(n)) == 1)
        qp = QuadraticProgram()
        qp.from_docplex(mdl)
        return qp

    def plot_graph(self, x, pos=None):
        route = self.interpret(x)
        nx.draw(self._g, with_labels=True, pos=pos)
        nx.draw_networkx_edges(
            self._g,
            pos,
            edgelist=[(route[i], route[(i+1) % len(route)]) for i in range(len(route))],
            width=8, alpha=0.5, edge_color="tab:red",
            )

    def interpret(self, x):
        n = int(np.sqrt(len(x)))
        route = []
        for p__ in range(n):
            for i in range(n):
                if x[i * n + p__]:
                    route.append(i)
        return route

    @staticmethod
    def random_graph(n=5, low=0, high=100, seed=None):
        random.seed(seed)
        pos = {i: (random.randint(low, high), random.randint(low, high)) for i in range(n)}
        g = nx.random_geometric_graph(n, 1+(high-low)**2, pos=pos)
        for u, v in g.edges:
            delta = [g.nodes[u]['pos'][i] - g.nodes[v]['pos'][i] for i in range(2)]
            g.edges[u, v]['weight'] = np.rint(np.hypot(delta[0], delta[1]))
        return g
