import networkx as nx
from docplex.mp.model import Model

from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from .graph_application import GraphApplication


class Maxcut(GraphApplication):

    def __init__(self, graph):
        super().__init__(graph)

    def to_quadratic_program(self):
        mdl = Model(name='Max-cut')
        x = {i: mdl.binary_var(name='x_{0}'.format(i)) for i in range(self._graph.number_of_nodes())}
        for u, v in self._graph.edges:
            self._graph.edges[u, v].setdefault('weight', 1)
        objective = mdl.sum(self._graph.edges[i, j]['weight'] * x[i]
                            * (1 - x[j]) for i, j in self._graph.edges)
        mdl.maximize(objective)
        qp = QuadraticProgram()
        qp.from_docplex(mdl)
        return qp

    def draw_graph(self, result=None, pos=None):
        if result is None:
            nx.draw(self._graph, pos=pos, with_labels=True)
        else:
            nx.draw(self._graph, node_color=self._node_color, pos=pos, with_labels=True)

    def interpret(self, result):
        cut = [[], []]
        for i, value in enumerate(result.x):
            if value == 0:
                cut[0].append(i)
            else:
                cut[1].append(i)
        return cut

    def _node_color(self, result):
        return ['r' if value == 0 else 'b' for value in result.x]
