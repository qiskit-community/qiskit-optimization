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

"""Converters to flip problem sense, e.g. maximization to minimization and vice versa."""

import copy
from typing import Optional, List, Union

import numpy as np

from .quadratic_program_converter import QuadraticProgramConverter
from ..exceptions import QiskitOptimizationError
from ..problems.quadratic_objective import ObjSense
from ..problems.quadratic_program import QuadraticProgram


class _FlipProblemSense(QuadraticProgramConverter):
    """Flip the sense of a problem, e.g. converts from maximization to minimization and
    vice versa, regardless of the current sense."""

    def __init__(self) -> None:
        self._src_num_vars: Optional[int] = None

    def convert(self, problem: QuadraticProgram) -> QuadraticProgram:
        """Flip the sense of a problem.

        Args:
            problem: The problem to be flipped.

        Returns:
            A converted problem, that has the flipped sense.
        """

        # copy original number of variables as reference.
        self._src_num_vars = problem.get_num_vars()
        desired_sense = self._get_desired_sense(problem)

        # flip the problem sense
        if problem.objective.sense != desired_sense:
            desired_problem = copy.deepcopy(problem)
            desired_problem.objective.sense = desired_sense
            desired_problem.objective.constant = (-1) * problem.objective.constant
            desired_problem.objective.linear = (-1) * problem.objective.linear.coefficients
            desired_problem.objective.quadratic = (-1) * problem.objective.quadratic.coefficients
        else:
            desired_problem = problem

        return desired_problem

    def _get_desired_sense(self, problem: QuadraticProgram) -> ObjSense:
        """
        Computes a desired sense of the problem. By default, flip the sense.

        Args:
            problem: a problem to check

        Returns:
            A desired sense, if the problem was a minimization problem, then the sense is
            maximization and vice versa.
        """
        if problem.objective.sense == ObjSense.MAXIMIZE:
            return ObjSense.MINIMIZE
        else:
            return ObjSense.MAXIMIZE

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
        if len(x) != self._src_num_vars:
            raise QiskitOptimizationError(
                f"The number of variables in the passed result differs from "
                f"that of the original problem, should be {self._src_num_vars}, but got {len(x)}."
            )
        return np.asarray(x)


class MaximizeToMinimize(_FlipProblemSense):
    """Convert a maximization problem to a minimization problem only if it is a maximization
    problem, otherwise problem's sense is unchanged."""

    def _get_desired_sense(self, problem: QuadraticProgram) -> ObjSense:
        return ObjSense.MINIMIZE


class MinimizeToMaximize(_FlipProblemSense):
    """Convert a minimization problem to a maximization problem only if it is a minimization
    problem, otherwise problem's sense is unchanged."""

    def _get_desired_sense(self, problem: QuadraticProgram) -> ObjSense:
        return ObjSense.MAXIMIZE
