# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""An application class for the Max-k-cut."""

from typing import List, Dict, Tuple, Optional, Union
import networkx as nx
import numpy as np
from docplex.mp.model import Model

from qiskit.utils import algorithm_globals
from qiskit_optimization.algorithms import OptimizationResult
from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from .graph_optimization_application import GraphOptimizationApplication

try:
    from matplotlib.pyplot import cm
    from matplotlib.colors import to_rgba

    _HAS_MATPLOTLIB = True
except ImportError:

    _HAS_MATPLOTLIB = False


class Maxkcut(GraphOptimizationApplication):
    """Optimization application for the "max-k-cut" [1] problem based on a NetworkX graph.

    References:
        [1]: Z. Tabi et al.,
             "Quantum Optimization for the Graph Coloring Problem with Space-Efficient Embedding"
             2020 IEEE International Conference on Quantum Computing and Engineering (QCE),
             2020, pp. 56-62, doi: 10.1109/QCE49297.2020.00018.,
             https://ieeexplore.ieee.org/document/9259934
    """

    def __init__(
        self,
        graph: Union[nx.Graph, np.ndarray, List],
        k: int,
    ) -> None:
        """
        Args:
            graph: A graph representing a problem. It can be specified directly as a
                `NetworkX <https://networkx.org/>`_ graph,
                or as an array or list format suitable to build out a NetworkX graph.
            k: The number of colors
        """
        super().__init__(graph=graph)
        self._subsets_num = k
        self._colors: Union[List[Tuple[float, float, float, float]], List[str]] = None
        self._seed: int = None

    def to_quadratic_program(self) -> QuadraticProgram:
        """Convert a Max-k-cut problem instance into a
        :class:`~qiskit_optimization.problems.QuadraticProgram`

        Returns:
            The :class:`~qiskit_optimization.problems.QuadraticProgram` created
            from the Max-k-cut problem instance.
        """
        for w, v in self._graph.edges:
            self._graph.edges[w, v].setdefault("weight", 1)

        mdl = Model(name="Max-k-cut")
        n = self._graph.number_of_nodes()
        k = self._subsets_num
        x = {(v, i): mdl.binary_var(name=f"x_{v}_{i}") for v in range(n) for i in range(k)}
        first_penalty = mdl.sum_squares((1 - mdl.sum(x[v, i] for i in range(k)) for v in range(n)))
        second_penalty = mdl.sum(
            mdl.sum(self._graph.edges[v, w]["weight"] * x[v, i] * x[w, i] for i in range(k))
            for v, w in self._graph.edges
        )
        objective = first_penalty + second_penalty
        mdl.minimize(objective)

        op = from_docplex_mp(mdl)
        return op

    def interpret(self, result: Union[OptimizationResult, np.ndarray]) -> List[List[int]]:
        """Interpret a result as k lists of node indices

        Args:
            result : The calculated result of the problem

        Returns:
            k lists of node indices correspond to k node sets for the Max-k-cut
        """
        x = self._result_to_x(result)
        n = self._graph.number_of_nodes()
        cut = [[] for i in range(self._subsets_num)]  # type: List[List[int]]

        n_selected = x.reshape((n, self._subsets_num))
        for i in range(n):
            node_in_subset = np.where(n_selected[i] == 1)[0]  # one-hot encoding
            if len(node_in_subset) != 0:
                cut[node_in_subset[0]].append(i)

        return cut

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

    def _node_color(
        self, x: np.ndarray
    ) -> Union[List[Tuple[float, float, float, float]], List[str]]:
        # Return a list of colors for draw.

        n = self._graph.number_of_nodes()

        # k colors chosen (randomly or from cm.rainbow), or from given color list
        if self._colors is None:
            if _HAS_MATPLOTLIB:
                colors = cm.rainbow(np.linspace(0, 1, self._subsets_num))
            else:
                if self._seed:
                    algorithm_globals.random_seed = self._seed
                colors = [
                    "#"
                    + "".join(
                        [algorithm_globals.random.choice("0123456789ABCDEF") for i in range(6)]
                    )
                    for j in range(self._subsets_num)
                ]
        else:
            colors = self._colors

        gray = to_rgba("lightgray") if _HAS_MATPLOTLIB else "lightgray"
        node_colors = [gray for _ in range(n)]

        n_selected = x.reshape((n, self._subsets_num))
        for i in range(n):
            node_in_subset = np.where(n_selected[i] == 1)  # one-hot encoding
            if len(node_in_subset[0]) != 0:
                node_colors[i] = (
                    to_rgba(colors[node_in_subset[0][0]])
                    if _HAS_MATPLOTLIB
                    else colors[node_in_subset[0][0]]
                )

        return node_colors

    @property
    def k(self) -> int:
        """Getter of k

        Returns:
            The number of colors
        """
        return self._subsets_num

    @k.setter
    def k(self, k: int) -> None:
        """Setter of k

        Args:
            k: The number of colors

        Raises:
            ValueError: if the size of the colors is different than the k parameter.
        """
        self._subsets_num = k
        if self._colors and len(self._colors) != self._subsets_num:
            self._colors = None
            raise ValueError(
                f"Number of colors in the list is different than the parameter"
                f" k = {self._subsets_num} specified for this problem,"
                f" the colors have not been assigned"
            )

    @property
    def colors(self) -> Union[List[Tuple[float, float, float, float]], List[str]]:
        """Getter of colors list

        Returns:
            The k size color list
        """
        return self._colors

    @colors.setter
    def colors(self, colors: Union[List[Tuple[float, float, float, float]], List[str]]) -> None:
        """Setter of colors list
        Colors list must be the same length as the k parameter. Color can be a string or rgb or
        rgba tuple of floats from 0-1. If numeric values are specified, they will be mapped to
        colors using the cmap and vmin, vmax parameters. See matplotlib colors docs for more
        details (https://matplotlib.org/stable/gallery/color/named_colors.html).

        Examples:
            [[0.0, 0.5, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], ...]
            ["g", "r", "b", ...]
            ["cyan", "purple", ...]

        Args:
            colors: The k size color list

        Raises:
            ValueError: if the size of the colors is different than the k parameter.
        """
        if colors and len(colors) == self._subsets_num:
            self._colors = colors
        else:
            self._colors = None
            raise ValueError(
                f"Number of colors in the list is different than the parameter"
                f" k = {self._subsets_num} specified for this problem,"
                f" the colors have not been assigned"
            )

    @property
    def seed(self) -> int:
        """Getter of seed

        Returns:
            The seed value for random generation of colors
        """
        return self._seed

    @seed.setter
    def seed(self, seed: int) -> None:
        """Setter of seed

        Args:
            seed: The seed value for random generation of colors
        """
        self._seed = seed
