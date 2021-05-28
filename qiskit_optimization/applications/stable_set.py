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

"""An application class for the stable set."""

from typing import Dict, List, Optional, Union

from retworkx.visualization import mpl_draw
import numpy as np
from docplex.mp.model import Model

from qiskit_optimization.algorithms import OptimizationResult
from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from .graph_optimization_application import GraphOptimizationApplication


class StableSet(GraphOptimizationApplication):
    """Optimization application for the "stable set" [1] problem based on a NetworkX graph.

    References:
        [1]: "Independent set (graph theory)",
        https://en.wikipedia.org/wiki/Independent_set_(graph_theory)
    """

    def to_quadratic_program(self) -> QuadraticProgram:
        """Convert a stable set instance into a
        :class:`~qiskit_optimization.problems.QuadraticProgram`

        Returns:
            The :class:`~qiskit_optimization.problems.QuadraticProgram` created
            from the stable set instance.
        """
        mdl = Model(name="Stable set")
        n = self._graph.num_nodes()
        x = {i: mdl.binary_var(name="x_{0}".format(i)) for i in range(n)}
        objective = mdl.sum(x[i] for i in x)
        for w, v in self._graph.edge_list():
            mdl.add_constraint(x[w] + x[v] <= 1)
        mdl.maximize(objective)
        op = QuadraticProgram()
        op.from_docplex(mdl)
        return op

    def interpret(self, result: Union[OptimizationResult, np.ndarray]) -> List[int]:
        """Interpret a result as a list of node indices

        Args:
            result : The calculated result of the problem

        Returns:
            A list of node indices whose corresponding variable is 1
        """
        x = self._result_to_x(result)
        stable_set = []
        for i, value in enumerate(x):
            if value:
                stable_set.append(i)
        return stable_set

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
        mpl_draw(self._graph, node_color=self._node_colors(x), pos=pos, with_labels=True)

    def _node_colors(self, x: np.ndarray):
        # Return a list of strings for draw.
        # Color a node with red when the corresponding variable is 1.
        # Otherwise color it with dark gray.
        return ["r" if x[node] == 1 else "darkgrey" for node in self._graph.nodes()]
