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
Convert graph partitioning instances into Pauli list
Deal with Gset format. See https://web.stanford.edu/~yyye/yyye/Gset/
"""

import networkx as nx
from docplex.mp.model import Model

from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from .graph_application import GraphApplication


class GraphPartitioning(GraphApplication):
    def __init__(self, g):
        super().__init__(g)

    def to_quadratic_program(self):
        mdl = Model(name='graph partinioning')
        n = self._graph.number_of_nodes()
        x = {i: mdl.binary_var(name='x_{0}'.format(i)) for i in range(n)}
        for u, v in self._graph.edges:
            self._graph.edges[u, v].setdefault('weight', 1)
        objective = mdl.sum(self._graph.edges[i, j]['weight'] *
                            (x[i] + x[j] - 2*x[i]*x[j]) for i, j in self._graph.edges)
        mdl.minimize(objective)
        mdl.add_constraint(mdl.sum([x[i] for i in x]) == n//2)
        qp = QuadraticProgram()
        qp.from_docplex(mdl)
        return qp

    def interpret(self, result):
        partition = [[], []]
        for i, value in enumerate(result.x):
            if value == 0:
                partition[0].append(i)
            else:
                partition[1].append(i)
        return partition

    def draw_graph(self, result=None, pos=None):
        if result is None:
            nx.draw(self._graph, pos=pos, with_labels=True)
        else:
            colors = ['r' if value == 0 else 'b' for value in result.x]
            nx.draw(self._graph, node_color=colors, pos=pos, with_labels=True)
