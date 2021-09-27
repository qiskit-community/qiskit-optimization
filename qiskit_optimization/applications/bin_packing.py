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

"""An application class for the set packing."""

from typing import List, Union

import numpy as np
from docplex.mp.model import Model

from qiskit_optimization.algorithms import OptimizationResult
from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from .optimization_application import OptimizationApplication


class BinPacking(OptimizationApplication):
    """Optimization application for the "bin packing" [1] problem.

    References:
        [1]: "Bin packing",
        https://en.wikipedia.org/wiki/Bin_packing_problem
    """

    def __init__(self, weights: List[int], max_weight: int) -> None:
        """
        Args:
            weights: A list of the weights of items
            max_weight: The maximum bin weight capacity
        """
        self._weights = weights
        self._max_weight = max_weight

    def to_quadratic_program(self) -> QuadraticProgram:
        """Convert a bin packing problem instance into a
        :class:`~qiskit_optimization.problems.QuadraticProgram`

        Returns:
            The :class:`~qiskit_optimization.problems.QuadraticProgram` created
            from the bin packing problem instance.
        """
        mdl = Model(name="BinPacking")
        num_items = len(self._weights)
        x = mdl.binary_var_list([f"x{i}" for i in range(num_items)])
        mdl.minimize(mdl.sum([x[i] for i in range(num_items)]))
        y = mdl.binary_var_list([f"y{i//num_items},{i%num_items}" for i in range(num_items ** 2)])
        for j in range(num_items):
            # First set of constraints: the items must be in any bin
            constraint0 = mdl.sum([y[i * num_items + j] for i in range(num_items)])
            mdl.add_constraint(constraint0 == 1, f"cons0,{j}")
        for i in range(num_items):
            # Second set of constraints: weight constraints
            constraint1 = mdl.sum(
                [self._weights[j] * y[i * num_items + j] for j in range(num_items)]
            )
            mdl.add_constraint(constraint1 <= self._max_weight * x[i], f"cons1,{i}")
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
            The maximal weight for the bin packing problem
        """
        return self._max_weight

    @max_weight.setter
    def max_weight(self, max_weight: int) -> None:
        """Setter of max_weight

        Args:
            max_weight: The maximal weight for the bin packing problem
        """
        self._max_weight = max_weight
