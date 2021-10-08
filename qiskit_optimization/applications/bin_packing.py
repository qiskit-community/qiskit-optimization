# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An application class for the bin packing."""

from typing import List, Union, Optional

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

    def __init__(
        self, weights: List[int], max_weight: int, max_number_of_bins: Optional[int] = None
    ) -> None:
        """
        Args:
            weights: A list of the weights of items
            max_weight: The maximum bin weight capacity
            max_number_of_bins: The maximum number of bins by default equal to the number of items
        """
        self._weights = weights
        self._max_weight = max_weight
        if max_number_of_bins is None:
            self._max_number_of_bins = len(weights)
        else:
            self._max_number_of_bins = max_number_of_bins

    def to_quadratic_program(self) -> QuadraticProgram:
        """Convert a bin packing problem instance into a
        :class:`~qiskit_optimization.problems.QuadraticProgram`

        Returns:
            The :class:`~qiskit_optimization.problems.QuadraticProgram` created
            from the bin packing problem instance.
        """
        mdl = Model(name="BinPacking")
        num_bins = self._max_number_of_bins
        num_items = len(self._weights)
        x = mdl.binary_var_list([f"x{i}" for i in range(num_items)])
        mdl.minimize(mdl.sum([x[i] for i in range(num_items)]))
        y = mdl.binary_var_matrix(num_bins, num_items, name="y")
        for i in range(num_items):
            # First set of constraints: the items must be in any bin
            mdl.add_constraint(mdl.sum([y[(j, i)] for j in range(num_bins)]) == 1)
        for i in range(num_bins):
            # Second set of constraints: weight constraints
            mdl.add_constraint(
                mdl.sum([self._weights[j] * y[(i, j)] for j in range(num_items)])
                <= self._max_weight * x[i]
            )
        op = from_docplex_mp(mdl)
        return op

    def interpret(self, result: Union[OptimizationResult, np.ndarray]) -> List[int]:
        """Interpret a result as item indices

        Args:
            result : The calculated result of the problem

        Returns:
            items_in_bins: A list of lists with the items in each bin
        """
        x = self._result_to_x(result)
        num_items = len(self._weights)
        num_bins = self._max_number_of_bins
        bins = x[:num_bins]
        items = np.array(x[num_bins:]).reshape((num_bins, num_items))
        items_in_bins = [
            [j for j in range(num_items) if bins[i] and items[i, j]] for i in range(num_bins)
        ]
        return items_in_bins
