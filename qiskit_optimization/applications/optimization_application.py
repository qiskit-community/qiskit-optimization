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

"""An abstract class for optimization application classes."""

from abc import ABC, abstractmethod

from qiskit_optimization.algorithms import OptimizationResult
from qiskit_optimization.problems.quadratic_program import QuadraticProgram


class OptimizationApplication(ABC):
    """
    An abstract class for optimization applications.
    """

    @abstractmethod
    def to_quadratic_program(self) -> QuadraticProgram:
        """Convert a problem instance into a
        :class:`~qiskit_optimization.problems.QuadraticProgram`
        """
        pass

    @abstractmethod
    def interpret(self, result: OptimizationResult):
        """Convert the calculation result of the problem
        (:class:`~qiskit_optimization.algorithms.OptimizationResult`) to the answer of the problem
        in an easy-to-understand format.

        Args:
            result: The calculated result of the problem
        """
        pass

    def is_feasible(self, result: OptimizationResult) -> bool:
        """Check whether the result is feasible or not

        Args:
            result: The calculated result of the problem

        Returns:
            The result is feasible or not
        """
        return self.to_quadratic_program().is_feasible(result.x)

    def evaluate(self, result: OptimizationResult) -> float:
        """Calculate the value of the objective function based on the result.

        Args:
            result: The calculated result of the problem

        Returns:
            The value of the objective function based on the result
        """
        return self.to_quadratic_program().objective.evaluate(result.x)
