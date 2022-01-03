# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An application class for the exact cover."""
from typing import List, Union, cast

import numpy as np
from docplex.mp.model import Model

from qiskit_optimization.algorithms import OptimizationResult
from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from .optimization_application import OptimizationApplication


class ExactCover(OptimizationApplication):
    """Optimization application for the "exact cover" [1] problem.

    References:
        [1]: "Exact cover",
        https://en.wikipedia.org/wiki/Exact_cover
    """

    def __init__(self, subsets: List[List[int]]) -> None:
        """
        Args:
            subsets: A list of subsets
        """
        self._subsets = subsets
        self._set = []
        for sub in self._subsets:
            self._set.extend(sub)
        self._set = cast(List, np.unique(self._set))

    def to_quadratic_program(self) -> QuadraticProgram:
        """Convert an exact cover instance into a
        :class:`~qiskit_optimization.problems.QuadraticProgram`

        Returns:
            The :class:`~qiskit_optimization.problems.QuadraticProgram` created
            from the exact cover instance.
        """
        mdl = Model(name="Exact cover")
        x = {i: mdl.binary_var(name=f"x_{i}") for i in range(len(self._subsets))}
        mdl.minimize(mdl.sum(x[i] for i in x))
        for element in self._set:
            mdl.add_constraint(
                mdl.sum(x[i] for i, sub in enumerate(self._subsets) if element in sub) == 1
            )
        op = from_docplex_mp(mdl)
        return op

    def interpret(self, result: Union[OptimizationResult, np.ndarray]) -> List[List[int]]:
        """Interpret a result as a list of subsets

        Args:
            result: The calculated result of the problem

        Returns:
            A list of subsets whose corresponding variable is 1
        """
        x = self._result_to_x(result)
        sub = []
        for i, value in enumerate(x):
            if value:
                sub.append(self._subsets[i])
        return sub
