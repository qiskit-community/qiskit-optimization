from abc import abstractmethod

import networkx as nx

from .base_application import BaseApplication


class GraphApplication(BaseApplication):
    """
    An abstract class for graph problems in optimization problems
    """

    def __init__(self, g):
        self._g = g.copy(as_view=True)

    @staticmethod
    def random_graph(n, m, seed=None):
        g = nx.gnm_random_graph(n, m, seed)
        return g

    @abstractmethod
    def plot_graph(self):
        raise NotImplementedError

    def g(self):
        return self._g
