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


class SetPacking(OptimizationApplication):
    """Optimization application for the "set packing" [1] problem.

    References:
        [1]: "Set packing",
        https://en.wikipedia.org/wiki/Set_packing
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
        self._set = np.unique(self._set)

    def to_quadratic_program(self) -> QuadraticProgram:
        """Convert a set packing instance into a
        :class:`~qiskit_optimization.problems.QuadraticProgram`

        Returns:
            The :class:`~qiskit_optimization.problems.QuadraticProgram` created
            from the set packing instance.
        """
        mdl = Model(name="Set packing")
        x = {i: mdl.binary_var(name="x_{0}".format(i)) for i in range(len(self._subsets))}
        mdl.maximize(mdl.sum(x[i] for i in x))
        for element in self._set:
            mdl.add_constraint(
                mdl.sum(x[i] for i, sub in enumerate(self._subsets) if element in sub) <= 1
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
