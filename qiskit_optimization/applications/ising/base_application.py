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

    def is_feasible(self, result):
        return self.to_quadratic_program().is_feasible(result.x)

    def evaluate(self, result):
        return self.to_quadratic_program().objective.evaluate(result.x)

