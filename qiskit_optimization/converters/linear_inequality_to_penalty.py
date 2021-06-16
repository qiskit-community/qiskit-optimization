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

import itertools
import logging
from math import fsum
from typing import Optional, cast, Union, Tuple, List

import numpy as np

from .quadratic_program_converter import QuadraticProgramConverter
from ..exceptions import QiskitOptimizationError
from ..problems.constraint import Constraint, ConstraintSense
from ..problems.quadratic_objective import QuadraticObjective
from ..problems.quadratic_program import QuadraticProgram
from ..problems.variable import Variable

logger = logging.getLogger(__name__)


class LinearInequalityToPenalty(QuadraticProgramConverter):
    r"""Convert a problem of inequality constraints to unconstrained with penalty terms.

    There are some constraints which do not require to add slack variables to
    construct penalty terms [1].Supported inequality constraint in this class is shown below.

    .. math::

        \begin{array}{}
        \text { Inequality Constraint } & & \text { Equivalent Penalty } \\
        x+y \leq 1 & \rightarrow & P(x y) \\
        x+y \geq 1 & \rightarrow & P(1-x-y+x y) \\
        x \leq y & \rightarrow  & P(x-x y) \\
        x \geq y & \rightarrow  & P(y-x y) \\
        x+y + z \leq 1 & \rightarrow & P\left(x y + y z + z x \right) \\
        \sum_i^n x_i \leq 1 & \rightarrow & P\left(\sum_{i, j : i < j} x_i x_j\right) \\
        \sum_i^n x_i \geq n-1 & \rightarrow & P\left(\sum_{i, j : i < j} (1 - x_i) (1 - x_j) \right) \\
        \end{array}

    Where x, y, z or :math:`x_i` are binary variables, and P is penalty constant.
    In this class,value of P is automatically determined, but can be supplied as
    argument at the timing of instantiation.

    If the constraint does not match the pattern of classical constraint, the constraint
    is kept as is. Otherwise, constraint is converted into equivalent penalty and added
    to objective function.

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

        self._src_num_vars = None  # type: Optional[int]
        self._dst = None  # type: Optional[QuadraticProgram]
        self.penalty = penalty  # type: Optional[float]

    def convert(self, problem: QuadraticProgram) -> QuadraticProgram:
        """Convert a problem of inequality constraints into an unconstrained problem.

        Args:
            problem: The problem to be solved, that does not contain inequality constraints.

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
                raise QiskitOptimizationError("Unsupported vartype: {}".format(x.vartype))

        # get original objective terms
        offset = problem.objective.constant
        linear = problem.objective.linear.to_dict()
        quadratic = problem.objective.quadratic.to_dict()
        sense = problem.objective.sense.value

        # convert linear constraints into penalty terms
        for constraint in problem.linear_constraints:

            # special constraint check function here
            if not self._is_matched_constraint(problem, constraint):
                self._dst.linear_constraints.append(constraint)
                continue

            conv_matrix = self._conversion_matrix(constraint)
            rowlist = list(constraint.linear.to_dict().items())

            # constant part
            if conv_matrix[0][0] != 0:
                offset += sense * penalty * conv_matrix[0][0]

            # linear parts of penalty
            for (j, z) in enumerate(rowlist):
                # if j already exists in the linear terms dic, add a penalty term
                # into existing value else create new key and value in the linear_term dict

                if conv_matrix[0][j + 1] != 0:
                    linear[z[0]] = linear.get(z[0], 0.0) + sense * penalty * conv_matrix[0][j + 1]

            # quadratic parts of penalty
            for (j, z) in enumerate(rowlist):
                for k in range(j, len(rowlist)):
                    # if j and k already exist in the quadratic terms dict,
                    # add a penalty term into existing value
                    # else create new key and value in the quadratic term dict

                    if conv_matrix[j + 1][k + 1] != 0:
                        tup = cast(Union[Tuple[int, int], Tuple[str, str]], (z[0], rowlist[k][0]))
                        quadratic[tup] = (
                            quadratic.get(tup, 0.0) + sense * penalty * conv_matrix[j + 1][k + 1]
                        )

        # Copy quadratic_constraints
        for quadratic_constraint in problem.quadratic_constraints:
            self._dst.quadratic_constraints.append(quadratic_constraint)

        if problem.objective.sense == QuadraticObjective.Sense.MINIMIZE:
            self._dst.minimize(offset, linear, quadratic)
        else:
            self._dst.maximize(offset, linear, quadratic)

        # Update the penalty to the one just used
        self._penalty = penalty  # type: float

        return self._dst

    def _conversion_matrix(self, constraint) -> np.ndarray:
        """Construct conversion matrix for special constraint.

        Returns:
            Return conversion matrix which is used to construct
            penalty term in main function.

        """

        vars_dict = constraint.linear.to_dict(use_name=True)
        vrs = list(vars_dict.items())
        rhs = constraint.rhs
        sense = constraint.sense

        num_vars = len(vrs)
        combinations = itertools.combinations(np.arange(num_vars), 2)

        # conversion matrix
        conv_matrix = np.zeros((num_vars + 1, num_vars + 1), dtype=int)

        for combination in combinations:
            index1 = combination[0] + 1
            index2 = combination[1] + 1

            if rhs in (1, num_vars - 1):
                if sense == ConstraintSense.GE:
                    conv_matrix[0][0] = conv_matrix[0][0] + 1
                    conv_matrix[0][index1] = conv_matrix[0][index1] - 1
                    conv_matrix[0][index2] = conv_matrix[0][index2] - 1
                conv_matrix[index1][index2] = 1
            elif rhs == 0:
                if sense == ConstraintSense.GE:
                    # x >= y case
                    if vrs[index1 - 1][1] < 0.0:
                        conv_matrix[0][index1] = 1
                    elif vrs[index2 - 1][1] < 0.0:
                        conv_matrix[0][index2] = 1
                elif sense == ConstraintSense.LE:
                    # x <= y case
                    if vrs[index1 - 1][1] > 0.0:
                        conv_matrix[0][index1] = 1
                    elif vrs[index2 - 1][1] > 0.0:
                        conv_matrix[0][index2] = 1

                conv_matrix[index1][index2] = -1

        return conv_matrix

    def _is_matched_constraint(self, problem, constraint) -> bool:
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
        if not all(problem.variables[i].vartype == Variable.Type.BINARY for i in params.keys()):
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

    def _auto_define_penalty(self, problem) -> float:
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
            terms.extend(coef for coef in constraint.linear.to_dict().values())
        if any(isinstance(term, float) and not term.is_integer() for term in terms):
            logger.warning(
                "Warning: Using %f for the penalty coefficient because "
                "a float coefficient exists in constraints. \n"
                "The value could be too small. "
                "If so, set the penalty coefficient manually.",
                default_penalty,
            )
            return default_penalty

        # (upper bound - lower bound) can be calculate as the sum of absolute value of coefficients
        # Firstly, add 1 to guarantee that infeasible answers will be greater than upper bound.
        penalties = [1.0]
        # add linear terms of the object function.
        penalties.extend(abs(coef) for coef in problem.objective.linear.to_dict().values())
        # add quadratic terms of the object function.
        penalties.extend(abs(coef) for coef in problem.objective.quadratic.to_dict().values())

        return fsum(penalties)

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
        self._should_define_penalty = penalty is None  # type: bool
