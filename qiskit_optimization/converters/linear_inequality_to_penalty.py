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

"""Converter to convert a problem with inequality constraints to unconstrained with penalty terms."""

import logging
from typing import Optional, Union, Tuple, List, Dict

import numpy as np

from .quadratic_program_converter import QuadraticProgramConverter
from ..exceptions import QiskitOptimizationError
from ..problems.constraint import Constraint, ConstraintSense
from ..problems.quadratic_objective import QuadraticObjective
from ..problems.quadratic_program import QuadraticProgram
from ..problems.variable import Variable

logger = logging.getLogger(__name__)


class LinearInequalityToPenalty(QuadraticProgramConverter):
    r"""Convert linear inequality constraints to penalty terms of the objective function.

    There are some linear constraints which do not require slack variables to
    construct penalty terms [1]. This class supports the following inequality constraints.

    .. math::

        \begin{array}{}
        \text { Inequality constraint } & & \text { Penalty term } \\
        x \leq y & \rightarrow  & P(x-x y) \\
        x \geq y & \rightarrow  & P(y-x y) \\
        \sum_{i=1}^n x_i \leq 1, n \geq 2 & \rightarrow & P \sum_{i, j : i < j} x_i x_j\\
        \sum_{i=1}^n x_i \geq n-1, n \geq 2 & \rightarrow & P \sum_{i, j : i < j} (1 - x_i) (1 - x_j)
        \end{array}

    Note that x, y, z and :math:`x_i` are binary variables, and P is a penalty factor,
    where the value of P is automatically determined or supplied by users.

    If constraints match with any of the patterns, they are converted into penalty terms and added
    to the objective function. Otherwise, constraints are kept as is.

    References:
        [1]: Fred Glover, et al. (2019),
             A Tutorial on Formulating and Using QUBO Models,
             `arXiv:1811.11538 <https://arxiv.org/abs/1811.11538>`_.

    """

    def __init__(self, penalty: Optional[float] = None) -> None:
        """
        Args:
            penalty: Penalty factor to scale equality constraints that are added to objective.
                     If None is passed, a penalty factor will be automatically calculated on
                     every conversion.
        """

        self._src_num_vars: Optional[int] = None
        self._dst: Optional[QuadraticProgram] = None
        self._penalty: Optional[float] = penalty
        self._should_define_penalty: bool = penalty is None

    def convert(self, problem: QuadraticProgram) -> QuadraticProgram:
        r"""Convert inequality constraints into penalty terms of the objective function.

        This methods converts the following patterns where x, y, and :math:`x_i` are binary variables
        and P is a penalty factor.

        .. math::

            \begin{array}{}
            \text { Inequality constraint } & & \text { Penalty term } \\
            x \leq y & \rightarrow  & P(x-x y) \\
            x \geq y & \rightarrow  & P(y-x y) \\
            \sum_{i=1}^n x_i \leq 1, n \geq 2 & \rightarrow & P \sum_{i, j : i < j} x_i x_j\\
            \sum_{i=1}^n x_i \geq n-1, n \geq 2 & \rightarrow & P \sum_{i, j : i < j} (1 - x_i) (1 - x_j)
            \end{array}

        Args:
            problem: The problem to be solved.

        Returns:
            The converted problem

        Raises:
            QiskitOptimizationError: If an unsupported-type variable exists.
        """

        # create empty QuadraticProgram model
        self._src_num_vars = problem.get_num_vars()
        self._dst = QuadraticProgram(name=problem.name)

        # If no penalty was given, set the penalty coefficient by _auto_define_penalty()
        if self._should_define_penalty:
            penalty = self._auto_define_penalty(problem)
        else:
            penalty = self._penalty

        # Set variables
        for x in problem.variables:
            if x.vartype == Variable.Type.CONTINUOUS:
                self._dst.continuous_var(x.lowerbound, x.upperbound, x.name)
            elif x.vartype == Variable.Type.BINARY:
                self._dst.binary_var(x.name)
            elif x.vartype == Variable.Type.INTEGER:
                self._dst.integer_var(x.lowerbound, x.upperbound, x.name)
            else:
                raise QiskitOptimizationError(f"Unsupported vartype: {x.vartype}")

        # get original objective terms
        offset = problem.objective.constant
        linear = problem.objective.linear.to_dict()
        quadratic = problem.objective.quadratic.to_dict()
        sense = problem.objective.sense.value

        # convert linear constraints into penalty terms
        for constraint in problem.linear_constraints:

            # special constraint check function here
            if not self._is_matched_constraint(problem, constraint):
                self._dst.linear_constraint(
                    constraint.linear.coefficients,
                    constraint.sense,
                    constraint.rhs,
                    constraint.name,
                )
                continue

            conv_offset, conv_linear, conv_quadratic, varmap = self._conversion_table(constraint)

            # constant part
            offset += sense * penalty * conv_offset

            # linear parts of penalty
            for j, j_2 in varmap.items():
                # if j already exists in the linear terms dic, add a penalty term
                # into existing value else create new key and value in the linear_term dict

                if conv_linear[j] != 0:
                    linear[j_2] = linear.get(j_2, 0.0) + sense * penalty * conv_linear[j]

            # quadratic parts of penalty
            for j, j_2 in varmap.items():
                for k in range(j, len(varmap)):
                    # if j and k already exist in the quadratic terms dict,
                    # add a penalty term into existing value
                    # else create new key and value in the quadratic term dict

                    if conv_quadratic[j][k] != 0:
                        tup = (j_2, varmap[k])
                        quadratic[tup] = (
                            quadratic.get(tup, 0.0) + sense * penalty * conv_quadratic[j][k]
                        )

        # Copy quadratic_constraints
        for quadratic_constraint in problem.quadratic_constraints:
            self._dst.quadratic_constraint(
                quadratic_constraint.linear.coefficients,
                quadratic_constraint.quadratic.coefficients,
                quadratic_constraint.sense,
                quadratic_constraint.rhs,
                quadratic_constraint.name,
            )

        if problem.objective.sense == QuadraticObjective.Sense.MINIMIZE:
            self._dst.minimize(offset, linear, quadratic)
        else:
            self._dst.maximize(offset, linear, quadratic)

        # Update the penalty to the one just used
        self._penalty = penalty

        return self._dst

    @staticmethod
    def _conversion_table(
        constraint,
    ) -> Tuple[int, np.ndarray, np.ndarray, Dict[int, int]]:
        """Construct conversion matrix for special constraint.

        Returns:
            Return conversion table which is used to construct
            penalty term in main function.

        Raises:
            QiskitOptimizationError: if the constraint is invalid.
        """

        vars_dict = constraint.linear.to_dict()
        coeffs = list(vars_dict.values())
        varmap = dict(enumerate(vars_dict.keys()))
        rhs = constraint.rhs
        sense = constraint.sense

        num_vars = len(vars_dict)

        # initialize return values, these are used for converted offset, linear
        # and quadratic terms
        offset = 0
        linear = np.zeros(num_vars, dtype=int)
        quadratic = np.zeros((num_vars, num_vars), dtype=int)

        # rhs = num_vars - 1 correspond to multiple variable with >= n - 1 case.
        if sense == ConstraintSense.GE and rhs == num_vars - 1:
            # x_1 + ... + x_n >= n - 1
            # The number of offset is combination ( nC2 )
            offset = num_vars * (num_vars - 1) // 2
            linear = np.full(num_vars, 1 - num_vars, dtype=int)
            quadratic = np.triu(np.ones((num_vars, num_vars), dtype=int), k=1)
        elif sense == ConstraintSense.LE and rhs == 1:
            # x_1 + ... + x_n <= 1
            quadratic = np.triu(np.ones((num_vars, num_vars), dtype=int), k=1)
        elif rhs == 0:
            if num_vars != 2:
                raise QiskitOptimizationError(
                    f"Internal error: invalid number of variables {num_vars} {constraint.name}"
                )
            quadratic = np.array([[0, -1], [0, 0]])
            if sense == ConstraintSense.GE:
                # x >= y case
                if coeffs[0] < 0.0:
                    linear[0] = 1
                else:
                    linear[1] = 1
            elif sense == ConstraintSense.LE:
                # x <= y case
                if coeffs[0] > 0.0:
                    linear[0] = 1
                else:
                    linear[1] = 1
        else:
            raise QiskitOptimizationError(f"Internal error: invalid constraint {constraint.name}")

        return offset, linear, quadratic, varmap

    @staticmethod
    def _is_matched_constraint(problem, constraint) -> bool:
        """Determine if constraint is special or not.

        Returns:
            True: when constraint is special
            False: when constraint is not special
        """

        params = constraint.linear.to_dict()
        num_vars = len(params)
        rhs = constraint.rhs
        sense = constraint.sense
        coeff_array = np.array(list(params.values()))

        # Binary parameter?
        if any(problem.variables[i].vartype != Variable.Type.BINARY for i in params.keys()):
            return False

        if num_vars == 2 and rhs == 0:
            if sense in (Constraint.Sense.LE, Constraint.Sense.GE):
                # x-y<=0
                # x-y>=0
                return coeff_array.min() == -1.0 and coeff_array.max() == 1.0
        elif num_vars >= 2:
            if sense == Constraint.Sense.LE and rhs == 1:
                if all(i == 1 for i in params.values()):
                    # x1+x2+...<=1
                    return True
            elif sense == Constraint.Sense.GE and rhs == num_vars - 1:
                if all(i == 1 for i in params.values()):
                    # x1+x2+...>=n-1
                    return True

        return False

    @staticmethod
    def _auto_define_penalty(problem) -> float:
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
                f"The number of variables in the passed result ({len(x)}) differs from "
                f"that of the original problem ({self._src_num_vars})."
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
