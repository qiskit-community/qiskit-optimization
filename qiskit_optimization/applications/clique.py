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

"""An application class for the clique."""
from typing import Optional, Union, List, Dict

import networkx as nx
import numpy as np
from docplex.mp.model import Model

from qiskit_optimization.algorithms import OptimizationResult
from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from .graph_optimization_application import GraphOptimizationApplication


class Clique(GraphOptimizationApplication):
    """Optimization application for the "clique" [1] problem based on a NetworkX graph.

    References:
        [1]: "Clique (graph theory)",
        https://en.wikipedia.org/wiki/Clique_(graph_theory)
    """

    def __init__(
        self, graph: Union[nx.Graph, np.ndarray, List], size: Optional[int] = None
    ) -> None:
        """
        Args:
            graph: A graph representing a clique problem. It can be specified directly as a
            NetworkX Graph, or as an array or list if format suitable to build out a NetworkX graph.
            size: The size of the clique. When it's None, this class makes an optimization model for
            a maximal clique instead of the specified size of a clique.
        """
        super().__init__(graph)
        self._size = size

    def to_quadratic_program(self) -> QuadraticProgram:
        """Convert a clique problem instance into a
        :class:`~qiskit_optimization.problems.QuadraticProgram`.
        When "size" is None, this makes an optimization model for a maximal clique
        instead of the specified size of a clique.

        Returns:
            The :class:`~qiskit_optimization.problems.QuadraticProgram` created
            from the clique problem instance.
        """
        complement_g = nx.complement(self._graph)

        mdl = Model(name="Clique")
        n = self._graph.number_of_nodes()
        x = {i: mdl.binary_var(name="x_{0}".format(i)) for i in range(n)}
        for w, v in complement_g.edges:
            mdl.add_constraint(x[w] + x[v] <= 1)
        if self.size is None:
            mdl.maximize(mdl.sum(x[i] for i in x))
        else:
            mdl.add_constraint(mdl.sum(x[i] for i in x) == self.size)
        op = from_docplex_mp(mdl)
        return op

    def interpret(self, result: Union[OptimizationResult, np.ndarray]) -> List[int]:
        """Interpret a result as a list of node indices

        Args:
            result : The calculated result of the problem

        Returns:
            The list of node indices whose corresponding variable is 1
        """
        x = self._result_to_x(result)
        clique = []
        for i, value in enumerate(x):
            if value:
                clique.append(i)
        return clique

    def _draw_result(
        self,
        result: Union[OptimizationResult, np.ndarray],
        pos: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        """Draw the result with colors

        Args:
            result : The calculated result for the problem
            pos: The positions of nodes
        """
        x = self._result_to_x(result)
        nx.draw(self._graph, node_color=self._node_colors(x), pos=pos, with_labels=True)

    def _node_colors(self, x: np.ndarray) -> List[str]:
        # Return a list of strings for draw.
        # Color a node with red when the corresponding variable is 1.
        # Otherwise color it with dark gray.
        return ["r" if x[node] else "darkgrey" for node in self._graph.nodes]

    @property
    def size(self) -> int:
        """Getter of size

        Returns:
            The size of the clique
        """
        return self._size

    @size.setter
    def size(self, size: int) -> None:
        """Setter of size

        Args:
            size: The size of the clique
        """
        self._size = size
