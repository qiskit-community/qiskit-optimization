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

"""Converter to convert a problem with equality constraints to unconstrained with penalty terms."""

import logging
from typing import Optional, cast, Union, Tuple, List

import numpy as np

from .quadratic_program_converter import QuadraticProgramConverter
from ..exceptions import QiskitOptimizationError
from ..problems.constraint import Constraint
from ..problems.quadratic_objective import QuadraticObjective
from ..problems.quadratic_program import QuadraticProgram
from ..problems.variable import Variable

logger = logging.getLogger(__name__)


class LinearEqualityToPenalty(QuadraticProgramConverter):
    """Convert a problem with only equality constraints to unconstrained with penalty terms."""

    def __init__(self, penalty: Optional[float] = None) -> None:
        """
        Args:
            penalty: Penalty factor to scale equality constraints that are added to objective.
                     If None is passed, a penalty factor will be automatically calculated on
                     every conversion.
        """
        self._src_num_vars: Optional[int] = None
        self._penalty: Optional[float] = penalty
        self._should_define_penalty: bool = penalty is None

    def convert(self, problem: QuadraticProgram) -> QuadraticProgram:
        """Convert a problem with equality constraints into an unconstrained problem.

        Args:
            problem: The problem to be solved, that does not contain inequality constraints.

        Returns:
            The converted problem, that is an unconstrained problem.

        Raises:
            QiskitOptimizationError: If an inequality constraint exists.
        """

        # create empty QuadraticProgram model
        self._src_num_vars = problem.get_num_vars()
        dst = QuadraticProgram(name=problem.name)

        # If no penalty was given, set the penalty coefficient by _auto_define_penalty()
        if self._should_define_penalty:
            penalty = self._auto_define_penalty(problem)
        else:
            penalty = self._penalty

        # Set variables
        for x in problem.variables:
            if x.vartype == Variable.Type.CONTINUOUS:
                dst.continuous_var(x.lowerbound, x.upperbound, x.name)
            elif x.vartype == Variable.Type.BINARY:
                dst.binary_var(x.name)
            elif x.vartype == Variable.Type.INTEGER:
                dst.integer_var(x.lowerbound, x.upperbound, x.name)
            else:
                raise QiskitOptimizationError("Unsupported vartype: {}".format(x.vartype))

        # get original objective terms
        offset = problem.objective.constant
        linear = problem.objective.linear.to_dict()
        quadratic = problem.objective.quadratic.to_dict()
        sense = problem.objective.sense.value

        # convert linear constraints into penalty terms
        for constraint in problem.linear_constraints:

            if constraint.sense != Constraint.Sense.EQ:
                raise QiskitOptimizationError(
                    "An inequality constraint exists. "
                    "The method supports only equality constraints."
                )

            constant = constraint.rhs
            row = constraint.linear.to_dict()

            # constant parts of penalty*(Constant-func)**2: penalty*(Constant**2)
            offset += sense * penalty * constant ** 2

            # linear parts of penalty*(Constant-func)**2: penalty*(-2*Constant*func)
            for j, coef in row.items():
                # if j already exists in the linear terms dic, add a penalty term
                # into existing value else create new key and value in the linear_term dict
                linear[j] = linear.get(j, 0.0) + sense * penalty * -2 * coef * constant

            # quadratic parts of penalty*(Constant-func)**2: penalty*(func**2)
            for j, coef_1 in row.items():
                for k, coef_2 in row.items():
                    # if j and k already exist in the quadratic terms dict,
                    # add a penalty term into existing value
                    # else create new key and value in the quadratic term dict

                    # according to implementation of quadratic terms in OptimizationModel,
                    # don't need to multiply by 2, since loops run over (x, y) and (y, x).
                    tup = cast(Union[Tuple[int, int], Tuple[str, str]], (j, k))
                    quadratic[tup] = quadratic.get(tup, 0.0) + sense * penalty * coef_1 * coef_2

        if problem.objective.sense == QuadraticObjective.Sense.MINIMIZE:
            dst.minimize(offset, linear, quadratic)
        else:
            dst.maximize(offset, linear, quadratic)

        # Update the penalty to the one just used
        self._penalty = penalty

        return dst

    @staticmethod
    def _auto_define_penalty(problem: QuadraticProgram) -> float:
        """Automatically define the penalty coefficient.

        Returns:
            Return the minimum valid penalty factor calculated
            from the upper bound and the lower bound of the objective function.
            If a constraint has a float coefficient,
            return the default value for the penalty factor.
        """
        default_penalty = 1e5

        # Check coefficients of constraints.
        # If a constraint has a float coefficient, return the default value for the penalty factor.
        terms = []
        for constraint in problem.linear_constraints:
            terms.append(constraint.rhs)
            terms.extend(constraint.linear.to_array().tolist())
        if any(isinstance(term, float) and not term.is_integer() for term in terms):
            logger.warning(
                "Warning: Using %f for the penalty coefficient because "
                "a float coefficient exists in constraints. \n"
                "The value could be too small. "
                "If so, set the penalty coefficient manually.",
                default_penalty,
            )
            return default_penalty

        lin_b = problem.objective.linear.bounds
        quad_b = problem.objective.quadratic.bounds
        return 1.0 + (lin_b.upperbound - lin_b.lowerbound) + (quad_b.upperbound - quad_b.lowerbound)

    def interpret(self, x: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Convert the result of the converted problem back to that of the original problem

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
                "The number of variables in the passed result differs from "
                "that of the original problem."
            )
        return np.asarray(x)

    @property
    def penalty(self) -> Optional[float]:
        """Returns the penalty factor used in conversion.

        Returns:
            The penalty factor used in conversion.
        """
        return self._penalty

    @penalty.setter
    def penalty(self, penalty: Optional[float]) -> None:
        """Set a new penalty factor.

        Args:
            penalty: The new penalty factor.
                     If None is passed, a penalty factor will be automatically calculated
                     on every conversion.
        """
        self._penalty = penalty
        self._should_define_penalty = penalty is None
