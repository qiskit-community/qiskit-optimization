import networkx as nx
from docplex.mp.model import Model

from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from .graph_application import GraphApplication


class Maxcut(GraphApplication):

    def __init__(self, g):
        super().__init__(g)

    def to_quadratic_program(self):
        mdl = Model(name='maxcut')
        x = {i: mdl.binary_var(name='x_{0}'.format(i)) for i in range(self._g.number_of_nodes())}
        for u, v in self._g.edges:
            self._g.edges[u, v].setdefault('weight', 1)
        objective = mdl.sum(self._g.edges[i, j]['weight'] * x[i]
                            * (1 - x[j]) for i, j in self._g.edges)
        mdl.maximize(objective)
        qp = QuadraticProgram()
        qp.from_docplex(mdl)
        return qp

    def plot_graph(self, x, pos=None):
        colors = ['r' if value == 0 else 'b' for value in x]
        nx.draw(self._g, node_color=colors, pos=pos, with_labels=True)

    def interpret(self, x):
        cut = [[], []]
        for i, value in enumerate(x):
            if value == 0:
                cut[0].append(i)
            else:
                cut[1].append(i)
        return cut
