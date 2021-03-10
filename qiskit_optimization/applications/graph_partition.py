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

"""An application class for the graph partitioning."""

from typing import Dict, List, Optional

import networkx as nx
import numpy as np
from docplex.mp.model import Model

from qiskit_optimization.algorithms import OptimizationResult
from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from .graph_optimization_application import GraphOptimizationApplication


class GraphPartition(GraphOptimizationApplication):
    """Optimization application for the "graph partition" [1] problem based on a NetworkX graph.

    References:
        [1]: "Graph partition", https://en.wikipedia.org/wiki/Graph_partition
    """

    def to_quadratic_program(self) -> QuadraticProgram:
        """Convert a graph partition instance into a
        :class:`~qiskit_optimization.problems.QuadraticProgram`

        Returns:
            The :class:`~qiskit_optimization.problems.QuadraticProgram` created
            from the graph partition instance.
        """
        mdl = Model(name='Graph partinion')
        n = self._graph.number_of_nodes()
        x = {i: mdl.binary_var(name='x_{0}'.format(i)) for i in range(n)}
        for w, v in self._graph.edges:
            self._graph.edges[w, v].setdefault('weight', 1)
        objective = mdl.sum(self._graph.edges[i, j]['weight'] *
                            (x[i] + x[j] - 2*x[i]*x[j]) for i, j in self._graph.edges)
        mdl.minimize(objective)
        mdl.add_constraint(mdl.sum([x[i] for i in x]) == n//2)
        op = QuadraticProgram()
        op.from_docplex(mdl)
        return op

    def interpret(self, result: OptimizationResult) -> List[List[int]]:
        """Interpret a result as a list of node indices

        Args:
            result : The calculated result of the problem

        Returns:
            A list of node indices divided into two groups.
        """
        partition = [[], []]  # type: List[List[int]]
        for i, value in enumerate(result.x):
            if value == 0:
                partition[0].append(i)
            else:
                partition[1].append(i)
        return partition

    def draw(self, result: Optional[OptimizationResult] = None,
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
            nx.draw(self._graph, node_color=self._node_colors, pos=pos, with_labels=True)

    def _node_colors(self, result: OptimizationResult) -> List[str]:
        # Return a list of strings for draw.
        # Color a node with red when the corresponding variable is 1.
        # Otherwise color it with blue.
        return ['r' if value else 'b' for value in result.x]
