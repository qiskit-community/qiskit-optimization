import copy
from abc import ABC, abstractmethod


class BaseApplication(ABC):
    """
    An abstract class for optimization problems
    """

    @abstractmethod
    def to_quadratic_program(self):
        raise NotImplementedError

    @abstractmethod
    def interpret():
        raise NotImplementedError

    def is_feasible(self, x):
        return self.to_quadratic_program().is_feasible(x)

    def evaluate(self, x):
        return self.to_quadratic_program().objective.evaluate(x)

