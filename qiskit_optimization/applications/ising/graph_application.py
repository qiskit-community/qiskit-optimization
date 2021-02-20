from typing import Optional
from abc import abstractmethod

import networkx as nx

from .base_application import BaseApplication


class GraphApplication(BaseApplication):
    """
    An abstract class for graph problems in optimization problems
    """

    def __init__(self, graph):
        self._graph = graph.copy(as_view=True)

    @abstractmethod
    def draw_graph(self, result=None):
        raise NotImplementedError

    def graph(self):
        return self._graph

    @staticmethod
    def random_graph(n, m, seed=None):
        g = nx.gnm_random_graph(n, m, seed)
        return g
