# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A converter from quadratic program to a QUBO."""

from typing import List, Optional, Union, cast

import numpy as np

from ..converters.flip_problem_sense import MaximizeToMinimize
from ..converters.inequality_to_equality import InequalityToEquality
from ..converters.integer_to_binary import IntegerToBinary
from ..converters.linear_equality_to_penalty import LinearEqualityToPenalty
from ..converters.linear_inequality_to_penalty import LinearInequalityToPenalty
from ..exceptions import QiskitOptimizationError
from ..problems.quadratic_program import QuadraticProgram
from .quadratic_program_converter import QuadraticProgramConverter


class QuadraticProgramToQubo(QuadraticProgramConverter):
    """Convert a given optimization problem to a new problem that is a QUBO.

    Examples:
        >>> from qiskit_optimization.problems import QuadraticProgram
        >>> from qiskit_optimization.converters import QuadraticProgramToQubo
        >>> problem = QuadraticProgram()
        >>> # define a problem
        >>> conv = QuadraticProgramToQubo()
        >>> problem2 = conv.convert(problem)
    """

    def __init__(self, penalty: Optional[float] = None) -> None:
        """
        Args:
            penalty: Penalty factor to scale equality constraints that are added to objective.
                If None is passed, a penalty factor will be automatically calculated on every
                conversion.
        """
        self._penalize_lin_eq_constraints = LinearEqualityToPenalty(penalty=penalty)
        self._penalize_lin_ineq_constraints = LinearInequalityToPenalty(penalty=penalty)
        self._converters = [
            self._penalize_lin_ineq_constraints,
            InequalityToEquality(mode="integer"),
            IntegerToBinary(),
            self._penalize_lin_eq_constraints,
            MaximizeToMinimize(),
        ]

    def convert(self, problem: QuadraticProgram) -> QuadraticProgram:
        """Convert a problem with linear constraints into new one with a QUBO form.

        Args:
            problem: The problem with linear constraints to be solved.

        Returns:
            The problem converted in QUBO format as minimization problem.

        Raises:
            QiskitOptimizationError: In case of an incompatible problem.
        """

        # analyze compatibility of problem
        msg = self.get_compatibility_msg(problem)
        if len(msg) > 0:
            raise QiskitOptimizationError("Incompatible problem: {}".format(msg))

        for conv in self._converters:
            problem = conv.convert(problem)
        return problem

    def interpret(self, x: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Convert a result of a converted problem into that of the original problem.

        Args:
            x: The result of the converted problem.

        Returns:
            The result of the original problem.
        """
        for conv in self._converters[::-1]:
            x = conv.interpret(x)
        return cast(np.ndarray, x)

    @staticmethod
    def get_compatibility_msg(problem: QuadraticProgram) -> str:
        """Checks whether a given problem can be solved with this optimizer.

        Checks whether the given problem is compatible, i.e., whether the problem can be converted
        to a QUBO, and otherwise, returns a message explaining the incompatibility.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            A message describing the incompatibility.
        """

        # initialize message
        msg = ""
        # check whether there are incompatible variable types
        if problem.get_num_continuous_vars() > 0:
            msg += "Continuous variables are not supported! "

        # check whether there are incompatible constraint types
        if len(problem.quadratic_constraints) > 0:
            msg += "Quadratic constraints are not supported. "
        # check whether there are float coefficients in constraints
        compatible_with_integer_slack = True
        for l_constraint in problem.linear_constraints:
            linear = l_constraint.linear.to_dict()
            if any(isinstance(coef, float) and not coef.is_integer() for coef in linear.values()):
                compatible_with_integer_slack = False
        for q_constraint in problem.quadratic_constraints:
            linear = q_constraint.linear.to_dict()
            quadratic = q_constraint.quadratic.to_dict()
            if any(
                isinstance(coef, float) and not coef.is_integer() for coef in quadratic.values()
            ) or any(isinstance(coef, float) and not coef.is_integer() for coef in linear.values()):
                compatible_with_integer_slack = False
        if not compatible_with_integer_slack:
            msg += "Can not convert inequality constraints to equality constraint because \
                    float coefficients are in constraints. "

        # if an error occurred, return error message, otherwise, return None
        return msg

    def is_compatible(self, problem: QuadraticProgram) -> bool:
        """Checks whether a given problem can be solved with the optimizer implementing this method.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            Returns True if the problem is compatible, False otherwise.
        """
        return len(self.get_compatibility_msg(problem)) == 0

    @property
    def penalty(self) -> Optional[float]:
        """Returns the penalty factor used in conversion.

        Returns:
            The penalty factor used in conversion.
        """
        return self._penalize_lin_eq_constraints.penalty

    @penalty.setter
    def penalty(self, penalty: Optional[float]) -> None:
        """Set a new penalty factor.

        Args:
            penalty: The new penalty factor.
                     If None is passed, a penalty factor will be automatically calculated on every
                     conversion.
        """
        self._penalize_lin_ineq_constraints.penalty = penalty
        self._penalize_lin_eq_constraints.penalty = penalty
