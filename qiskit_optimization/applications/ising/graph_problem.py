from .optimization_problem import OptimizationProblem
from abc import abstractmethod


class GraphProblem(OptimizationProblem):
    """
    An abstract class for graph problems in optimization problems
    """

    def random_graph(self, n, m, seed=None):
        g = nx.gnm_random_graph(n, m, seed)
        self.g = g
        return self.g

    @abstractmethod
    def plot_graph(self):
        raise NotImplementedError
