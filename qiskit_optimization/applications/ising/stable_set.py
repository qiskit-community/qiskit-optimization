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
Convert stable set instances into Pauli list.  We read instances in
the Gset format, see https://web.stanford.edu/~yyye/yyye/Gset/ , for
compatibility with the maxcut format, but the weights on the edges
as they are not really used and are always assumed to be 1.  The
graph is represented by an adjacency matrix.
"""

import networkx as nx
import numpy as np
from docplex.mp.model import Model

from .graph_application import GraphApplication
from qiskit_optimization.problems.quadratic_program import QuadraticProgram


class StableSet(GraphApplication):

    def __init__(self, g):
        super().__init__(g)

    def to_quadratic_program(self):
        mdl = Model(name='stable set')
        n = self._graph.number_of_nodes()
        x = {i: mdl.binary_var(name='x_{0}'.format(i)) for i in range(n)}
        for u, v in self._graph.edges:
            self._graph.edges[u, v].setdefault('weight', 1)
        objective = mdl.sum(x[i] for i in x)
        for u, v in self._graph.edges:
            mdl.add_constraint(x[u] + x[v] <= 1)
        mdl.maximize(objective)
        qp = QuadraticProgram()
        qp.from_docplex(mdl)
        return qp

    def interpret(self, result):
        stable_set = []
        for i, value in enumerate(result.x):
            if value:
                stable_set.append(i)
        return stable_set

    def draw_graph(self, result, pos=None):
        if result is None:
            nx.draw(self._graph, pos=pos, with_labels=True)
        else:
            colors = ['r' if value == 1 else 'darkgrey' for value in result.x]
            nx.draw(self._graph, node_color=colors, pos=pos, with_labels=True)
