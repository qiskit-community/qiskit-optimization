# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An abstract class for graph optimization application classes."""

from abc import abstractmethod
from typing import Union, Optional, Dict, List

import networkx as nx
import numpy as np

import qiskit_optimization.optionals as _optionals
from .optimization_application import OptimizationApplication
from ..algorithms import OptimizationResult
from ..deprecation import DeprecatedType, deprecate_method


class GraphOptimizationApplication(OptimizationApplication):
    """
    An abstract class for graph optimization applications.
    """

    def __init__(self, graph: Union[nx.Graph, np.ndarray, List]) -> None:
        """
        Args:
            graph: A graph representing a problem. It can be specified directly as a
                `NetworkX <https://networkx.org/>`_ graph,
                or as an array or list format suitable to build out a NetworkX graph.
        """
        # The view of the graph is stored which means the graph can not be changed.
        self._graph = nx.Graph(graph).copy(as_view=True)

    @_optionals.HAS_MATPLOTLIB.require_in_call
    def draw(
        self,
        result: Optional[Union[OptimizationResult, np.ndarray]] = None,
        pos: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        """Draw a graph with the result. When the result is None, draw an original graph without
        colors.

        Args:
            result: The calculated result for the problem
            pos: The positions of nodes
        """
        if result is None:
            nx.draw(self._graph, pos=pos, with_labels=True)
        else:
            self._draw_result(result, pos)

    @abstractmethod
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
        pass

    @property
    def graph(self) -> nx.Graph:
        """Getter of the graph

        Returns:
            A graph for a problem
        """
        return self._graph

    @staticmethod
    @deprecate_method(
        "0.3.0", DeprecatedType.FUNCTION, "networkx.gnm_random_graph", "in NetworkX, directly"
    )
    def random_graph(num_nodes: int, num_edges: int, seed: Optional[int] = None) -> nx.Graph:
        """
        Args:
            num_nodes: The number of nodes in a graph
            num_edges: The number of edges in a graph
            seed: seed for a random graph

        Returns:
            A random graph of NetworkX
        """
        return nx.gnm_random_graph(num_nodes, num_edges, seed)
