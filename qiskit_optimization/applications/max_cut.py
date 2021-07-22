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

from typing import List, Dict, Optional, Union
import networkx as nx
import numpy as np
from docplex.mp.model import Model

from qiskit_optimization.algorithms import OptimizationResult
from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from .graph_optimization_application import GraphOptimizationApplication


class Maxcut(GraphOptimizationApplication):
    """Optimization application for the "max-cut" [1] problem based on a NetworkX graph.

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
        mdl = Model(name="Max-cut")
        x = {
            i: mdl.binary_var(name="x_{0}".format(i)) for i in range(self._graph.number_of_nodes())
        }
        for w, v in self._graph.edges:
            self._graph.edges[w, v].setdefault("weight", 1)
        objective = mdl.sum(
            self._graph.edges[i, j]["weight"] * x[i] * (1 - x[j])
            + self._graph.edges[i, j]["weight"] * x[j] * (1 - x[i])
            for i, j in self._graph.edges
        )
        mdl.maximize(objective)
        op = from_docplex_mp(mdl)
        return op

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
        nx.draw(self._graph, node_color=self._node_color(x), pos=pos, with_labels=True)

    def interpret(self, result: Union[OptimizationResult, np.ndarray]) -> List[List[int]]:
        """Interpret a result as two lists of node indices

        Args:
            result : The calculated result of the problem

        Returns:
            Two lists of node indices correspond to two node sets for the Max-cut
        """
        x = self._result_to_x(result)
        cut = [[], []]  # type: List[List[int]]
        for i, value in enumerate(x):
            if value == 0:
                cut[0].append(i)
            else:
                cut[1].append(i)
        return cut

    def _node_color(self, x: np.ndarray) -> List[str]:
        # Return a list of strings for draw.
        # Color a node with red when the corresponding variable is 1.
        # Otherwise color it with blue.
        return ["r" if value == 0 else "b" for value in x]

    @staticmethod
    def parse_gset_format(filename: str) -> np.ndarray:
        """Read graph in Gset format from file.

        Args:
            filename: the name of the file.

        Returns:
            An adjacency matrix as a 2D numpy array.
        """
        n = -1
        with open(filename) as infile:
            header = True
            m = -1
            count = 0
            for line in infile:
                # pylint: disable=unnecessary-lambda
                v = map(lambda e: int(e), line.split())
                if header:
                    n, m = v
                    w = np.zeros((n, n))
                    header = False
                else:
                    s__, t__, _ = v
                    s__ -= 1  # adjust 1-index
                    t__ -= 1  # ditto
                    w[s__, t__] = t__
                    count += 1
            assert m == count
        w += w.T
        return w

    @staticmethod
    def get_gset_result(x: np.ndarray) -> Dict[int, int]:
        """Get graph solution in Gset format from binary string.

        Args:
            x: binary string as numpy array.

        Returns:
            A graph solution in Gset format.
        """
        return {i + 1: 1 - x[i] for i in range(len(x))}
