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

"""Converter to convert a maximization problem to minimization problem."""

import copy
from typing import Optional, Union, List

import numpy as np

from .quadratic_program_converter import QuadraticProgramConverter
from ..exceptions import QiskitOptimizationError
from ..problems.quadratic_objective import QuadraticObjective
from ..problems.quadratic_program import QuadraticProgram


class MaximizeToMinimize(QuadraticProgramConverter):
    """Convert a maximization problem to minimization problem."""

    def __init__(self) -> None:
        self._src = None  # type: Optional[QuadraticProgram]
        self._dst = None  # type: Optional[QuadraticProgram]

    def convert(self, problem: QuadraticProgram) -> QuadraticProgram:
        """Convert a problem into a new minimization problem.

        Args:
            problem: The problem to be solved, that is a maximization or minimization problem.

        Returns:
            The converted problem, that is a minimization problem.
        """

        # Copy original QP as reference.
        self._src = copy.deepcopy(problem)
        self._dst = copy.deepcopy(problem)

        if self._src.objective.sense == QuadraticObjective.Sense.MAXIMIZE:
            # Turn the problem to `ObjSense.MINIMIZE` by flipping the sign of the objective function

            self._dst.objective.sense = QuadraticObjective.Sense.MINIMIZE
            self._dst.objective.constant = (-1) * self._src.objective.constant
            self._dst.objective.linear = (-1) * self._src.objective.linear.coefficients
            self._dst.objective.quadratic = (-1) * self._src.objective.quadratic.coefficients

        return self._dst

    def interpret(self, x: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Convert the result of the converted problem back to that of the original problem.

        Args:
            x: The result of the converted problem or the given result in case of FAILURE.

        Returns:
            The result of the original problem.

        Raises:
            QiskitOptimizationError: if the number of variables in the result differs from
                                     that of the original problem.
        """
        if len(x) != self._src.get_num_vars():
            raise QiskitOptimizationError(
                "The number of variables in the passed result differs from "
                "that of the original problem."
            )
        return np.asarray(x)
