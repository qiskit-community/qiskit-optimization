from abc import ABC, abstractmethod


class OptimizationProblem(ABC):
    """
    An abstract class for optimization problems
    """

    @abstractmethod
    def to_quadratic_problem(self):
        raise NotImplementedError

    def is_feasible(self, x):
        qp = self.to_quadratic_problem()
        return qp.is_feasible(x)

    def objective_value(self, x):
        qp = self.to_quadratic_problem()
        var_values = {}
        for i, var in enumerate(qp.variables):
            var_values[var.name] = x[i]
        return qp.substitute_variables(var_values).objective.constant
