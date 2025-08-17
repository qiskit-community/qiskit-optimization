# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An application class for the bin packing."""
from __future__ import annotations


import numpy as np
from docplex.mp.model import Model

import qiskit_optimization.optionals as _optionals
from qiskit_optimization.algorithms import OptimizationResult
from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from .optimization_application import OptimizationApplication

if _optionals.HAS_MATPLOTLIB:
    from matplotlib.pyplot import Figure
else:

    class Figure:  # type: ignore
        """Empty Figure class
        Replacement Figure for when matplotlib is not present.
        """

        pass


class BinPacking(OptimizationApplication):
    """Optimization application for the "bin packing" [1] problem.

    References:
        [1]: "Bin packing",
        `https://en.wikipedia.org/wiki/Bin_packing_problem
        <https://en.wikipedia.org/wiki/Bin_packing_problem>`_
    """

    def __init__(
        self, weights: list[int], max_weight: int, max_number_of_bins: int | None = None
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
        y = mdl.binary_var_list(num_bins, name="y")
        mdl.minimize(mdl.sum(y))
        x = mdl.binary_var_matrix(num_items, num_bins, name="x")
        for i in range(num_items):
            # First set of constraints: the items must be in any bin
            mdl.add_constraint(mdl.sum(x[i, j] for j in range(num_bins)) == 1)
        for j in range(num_bins):
            # Second set of constraints: weight constraints
            mdl.add_constraint(
                mdl.sum(self._weights[i] * x[i, j] for i in range(num_items))
                <= self._max_weight * y[j]
            )
        op = from_docplex_mp(mdl)
        return op

    def interpret(self, result: OptimizationResult | np.ndarray) -> list[list[int]]:
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
        items = np.array(x[num_bins:]).reshape((num_items, num_bins))
        items_in_bins = [
            [i for i in range(num_items) if bins[j] and items[i, j]] for j in range(num_bins)
        ]
        return items_in_bins

    @_optionals.HAS_MATPLOTLIB.require_in_call
    def get_figure(self, result: OptimizationResult | np.ndarray) -> Figure:
        """Get plot of the solution of the Bin Packing Problem.

        Args:
            result : The calculated result of the problem

        Returns:
            fig: A plot of the solution, where x and y represent the bins and
            sum of the weights respectively.
        """
        import matplotlib.pyplot as plt

        colors = plt.colormaps["jet"].resampled(len(self._weights))
        items_in_bins = self.interpret(result)
        num_bins = len(items_in_bins)
        fig, axes = plt.subplots()
        for _, bin_i in enumerate(items_in_bins):
            sum_items = 0
            for item in bin_i:
                axes.bar(
                    _,
                    self._weights[item],
                    bottom=sum_items,
                    label=f"Item {item}",
                    color=colors(item),
                )
                sum_items += self._weights[item]
        axes.hlines(
            self._max_weight,
            -0.5,
            num_bins - 0.5,
            linestyle="--",
            color="tab:red",
            label="Max Weight",
        )
        axes.set_xticks(np.arange(num_bins))
        axes.set_xlabel("Bin")
        axes.set_ylabel("Weight")
        axes.legend()
        return fig
