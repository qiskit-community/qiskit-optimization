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


"""An application class for the Max-cut."""

from typing import List, Dict, Optional
import networkx as nx
import numpy as np
from docplex.mp.model import Model

from qiskit_optimization.algorithms import OptimizationResult
from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from .graph_optimization_application import GraphOptimizationApplication


class Maxcut(GraphOptimizationApplication):
    """Convert a Max-cut problem [1] instance based on a graph of Networkx
    into a :class:`~qiskit_optimization.problems.QuadraticProgram`

    References:
        [1]: "Maximum cut",
        https://en.wikipedia.org/wiki/Maximum_cut
    """

    def to_quadratic_program(self) -> QuadraticProgram:
        """Convert a Max-cut problem instance into a
        :class:`~qiskit_optimization.problems.QuadraticProgram`

        Returns:
            The :class:`~qiskit_optimization.problems.QuadraticProgram` created
            from the Max-cut problem instance.
        """
        mdl = Model(name='Max-cut')
        x = {i: mdl.binary_var(name='x_{0}'.format(i))
             for i in range(self._graph.number_of_nodes())}
        for u, v in self._graph.edges:
            self._graph.edges[u, v].setdefault('weight', 1)
        objective = mdl.sum(self._graph.edges[i, j]['weight'] * x[i]
                            * (1 - x[j]) for i, j in self._graph.edges)
        mdl.maximize(objective)
        op = QuadraticProgram()
        op.from_docplex(mdl)
        return op

    def draw_graph(self, result: Optional[OptimizationResult] = None,
                   pos: Optional[Dict[int, np.ndarray]] = None) -> None:
        """Draw a graph with the result. When the result is None, draw an original graph without
        colors.

        Args:
            result : The calculated result for the problem
            pos: The positions of nodes
        """
        if result is None:
            nx.draw(self._graph, pos=pos, with_labels=True)
        else:
            nx.draw(self._graph, node_color=self._node_color, pos=pos, with_labels=True)

    def interpret(self, result: OptimizationResult) -> List[List[int]]:
        """Interpret a result as two lists of node indices

        Args:
            result : The calculated result of the problem

        Returns:
            Two lists of node indices correspond to two node sets for the Max-cut
        """
        cut = [[], []]
        for i, value in enumerate(result.x):
            if value == 0:
                cut[0].append(i)
            else:
                cut[1].append(i)
        return cut

    def _node_color(self, result: OptimizationResult) -> List[str]:
        # Return a list of strings for draw_graph.
        # Color a node with red when the corresponding variable is 1.
        # Otherwise color it with blue.
        return ['r' if value == 0 else 'b' for value in result.x]
