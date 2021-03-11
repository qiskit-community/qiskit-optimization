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

"""An abstract class for graph optimization application classes."""

from typing import Union, Optional, Dict, List
from abc import abstractmethod

import networkx as nx
import numpy as np

from qiskit_optimization.algorithms import OptimizationResult
from .optimization_application import OptimizationApplication


class GraphOptimizationApplication(OptimizationApplication):
    """
    An abstract class for graph optimization applications.
    """

    def __init__(self, graph: Union[nx.Graph, np.ndarray, List]) -> None:
        """
        Args:
            graph: A graph representing a problem. It can be specified directly as a
            NetworkX Graph, or as an array or list if format suitable to build out a NetworkX graph.
        """
        # The view of the graph is stored which means the graph can not be changed.
        self._graph = nx.Graph(graph).copy(as_view=True)

    @abstractmethod
    def draw(self, result: Optional[Union[OptimizationResult, np.ndarray]] = None,
             pos: Optional[Dict[int, np.ndarray]] = None) -> None:
        """An abstract method to draw the graph based on the result.

        Args:
            result: The calculated result for the problem
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
    def random_graph(num_nodes: int, num_edges: int, seed: Optional[int] = None) -> nx.Graph:
        """

        Args:
            num_nodes: The number of nodes in a graph
            num_edges: The number of edges in a graph
            seed: seed for a random graph

        Returns:
            A random graph of NetworkX
        """
        graph = nx.gnm_random_graph(num_nodes, num_edges, seed)
        return graph

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
                v = map(lambda e: int(e), line.split())  # pylint: disable=unnecessary-lambda
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
