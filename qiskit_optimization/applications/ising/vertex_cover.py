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

"""An application class for the vertex cover."""

from typing import Dict, List, Optional

import networkx as nx
import numpy as np
from docplex.mp.model import Model

from qiskit_optimization.algorithms import OptimizationResult
from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from .graph_optimization_application import GraphOptimizationApplication


class VertexCover(GraphOptimizationApplication):
    """Convert a vertex cover [1] instance based on a graph of NetworkX into a
    :class:`~qiskit_optimization.problems.QuadraticProgram`

    References:
        [1]: "Vertex cover", https://en.wikipedia.org/wiki/Vertex_cover
    """

    def to_quadratic_program(self) -> QuadraticProgram:
        """Convert a vertex cover instance into a
        :class:`~qiskit_optimization.problems.QuadraticProgram`

        Returns:
            The :class:`~qiskit_optimization.problems.QuadraticProgram` created
            from the vertex cover instance.
        """
        mdl = Model(name='Vertex cover')
        n = self._graph.number_of_nodes()
        x = {i: mdl.binary_var(name='x_{0}'.format(i)) for i in range(n)}
        objective = mdl.sum(x[i] for i in x)
        for w, v in self._graph.edges:
            mdl.add_constraint(x[w] + x[v] >= 1)
        mdl.minimize(objective)
        op = QuadraticProgram()
        op.from_docplex(mdl)
        return op

    def interpret(self, result: OptimizationResult) -> List[int]:
        """Interpret a result as a list of node indices

        Args:
            result : The calculated result of the problem

        Returns:
            A list of node indices whose corresponding variable is 1
        """
        vertex_cover = []
        for i, value in enumerate(result.x):
            if value:
                vertex_cover.append(i)
        return vertex_cover

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
            nx.draw(self._graph, node_color=self._node_colors(result), pos=pos, with_labels=True)

    def _node_colors(self, result: OptimizationResult) -> List[str]:
        # Return a list of strings for draw_graph.
        # Color a node with red when the corresponding variable is 1.
        # Otherwise color it with darkgrey.
        return ['r' if value else 'darkgrey' for value in result.x]
