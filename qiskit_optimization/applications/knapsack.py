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

"""An application class for the Knapsack problem"""
from typing import List, Union

import numpy as np
from docplex.mp.model import Model

from qiskit_optimization.algorithms import OptimizationResult
from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from .optimization_application import OptimizationApplication


class Knapsack(OptimizationApplication):
    """Optimization application for the "knapsack problem" [1].

    References:
        [1]: "Knapsack problem",
        https://en.wikipedia.org/wiki/Knapsack_problem
    """

    def __init__(self, values: List[int], weights: List[int], max_weight: int) -> None:
        """
        Args:
            values: A list of the values of items
            weights: A list of the weights of items
            max_weight: The maximum weight capacity
        """
        self._values = values
        self._weights = weights
        self._max_weight = max_weight

    def to_quadratic_program(self) -> QuadraticProgram:
        """Convert a knapsack problem instance into a
        :class:`~qiskit_optimization.problems.QuadraticProgram`

        Returns:
            The :class:`~qiskit_optimization.problems.QuadraticProgram` created
            from the knapsack problem instance.
        """
        mdl = Model(name="Knapsack")
        x = {i: mdl.binary_var(name="x_{0}".format(i)) for i in range(len(self._values))}
        mdl.maximize(mdl.sum(self._values[i] * x[i] for i in x))
        mdl.add_constraint(mdl.sum(self._weights[i] * x[i] for i in x) <= self._max_weight)
        op = from_docplex_mp(mdl)
        return op

    def interpret(self, result: Union[OptimizationResult, np.ndarray]) -> List[int]:
        """Interpret a result as item indices

        Args:
            result : The calculated result of the problem

        Returns:
            A list of items whose corresponding variable is 1
        """
        x = self._result_to_x(result)
        return [i for i, value in enumerate(x) if value]

    @property
    def max_weight(self) -> int:
        """Getter of max_weight

        Returns:
            The maximal weight for the knapsack problem
        """
        return self._max_weight

    @max_weight.setter
    def max_weight(self, max_weight: int) -> None:
        """Setter of max_weight

        Args:
            max_weight: The maximal weight for the knapsack problem
        """
        self._max_weight = max_weight
