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

import networkx as nx
import numpy as np
from docplex.mp.model import Model

from .graph_application import GraphApplication
from qiskit_optimization.problems.quadratic_program import QuadraticProgram


class Clique(GraphApplication):

    def __init__(self, graph):
        super().__init__(graph)

    def to_quadratic_program(self, K=None):
        complement_g = nx.complement(self._graph)

        mdl = Model(name='clique')
        n = self._graph.number_of_nodes()
        x = {i: mdl.binary_var(name='x_{0}'.format(i)) for i in range(n)}
        for u, v in complement_g.edges:
            mdl.add_constraint(x[u] + x[v] <= 1)
        if K is None:
            mdl.maximize(mdl.sum(x[i] for i in x))
        else:
            mdl.add_constraint(mdl.sum(x[i] for i in x) == K)
        qp = QuadraticProgram()
        qp.from_docplex(mdl)
        return qp

    def interpret(self, result):
        clique = []
        for i, value in enumerate(result.x):
            if value:
                clique.append(i)
        return clique

    def draw_graph(self, result, pos=None):
        if result is None:
            nx.draw(self._graph, pos=pos, with_labels=True)
        else:
            colors = ['r' if value == 1 else 'darkgrey' for value in result.x]
            nx.draw(self._graph, node_color=colors, pos=pos, with_labels=True)

    def is_feasible(self, result, K=None):
        return self.to_quadratic_program(K=K).is_feasible(result.x)

    def evaluate(self, result, K=None):
        return self.to_quadratic_program(K=K).objective.evaluate(result.x)
