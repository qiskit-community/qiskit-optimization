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
"""The inequality to equality converter."""

import copy
import logging
from typing import List, Optional, Union

import numpy as np

from .quadratic_program_converter import QuadraticProgramConverter
from ..exceptions import QiskitOptimizationError
from ..problems.constraint import Constraint
from ..problems.quadratic_objective import QuadraticObjective
from ..problems.quadratic_program import QuadraticProgram
from ..problems.variable import Variable

logger = logging.getLogger(__name__)


class IndicatorToInequality(QuadraticProgramConverter):
    """Convert inequality constraints into equality constraints by introducing slack variables.

    Examples:
        >>> from qiskit_optimization.problems import QuadraticProgram
        >>> from qiskit_optimization.converters import IndicatorToInequality
        >>> problem = QuadraticProgram()
        >>> # define a problem
        >>> conv = IndicatorToInequality()
        >>> problem2 = conv.convert(problem)
    """

    _delimiter = "@"  # users are supposed not to use this character in variable names

    def __init__(self) -> None:
        self._src_num_var = 0
        self._dst = None  # type: Optional[QuadraticProgram]

    def convert(self, problem: QuadraticProgram) -> QuadraticProgram:
        """Convert a problem with indicator constraints into one with only inequality constraints.

        Args:
            problem: The problem to be solved, that may contain indicator constraints.

        Returns:
            The converted problem, that contain only inequality constraints.

        Raises:
            QiskitOptimizationError: If a variable type is not supported.
            QiskitOptimizationError: If an unsupported mode is selected.
            QiskitOptimizationError: If an unsupported sense is specified.
        """
        self._src_num_var = problem.get_num_vars()
        self._dst = QuadraticProgram(name=problem.name)

        # Copy variables
        for x in problem.variables:
            if x.vartype == Variable.Type.BINARY:
                self._dst.binary_var(name=x.name)
            elif x.vartype == Variable.Type.INTEGER:
                self._dst.integer_var(name=x.name, lowerbound=x.lowerbound, upperbound=x.upperbound)
            elif x.vartype == Variable.Type.CONTINUOUS:
                self._dst.continuous_var(
                    name=x.name, lowerbound=x.lowerbound, upperbound=x.upperbound
                )
            else:
                raise QiskitOptimizationError("Unsupported variable type {}".format(x.vartype))

        # Copy the objective function
        constant = problem.objective.constant
        linear = problem.objective.linear.to_dict(use_name=True)
        quadratic = problem.objective.quadratic.to_dict(use_name=True)
        if problem.objective.sense == QuadraticObjective.Sense.MINIMIZE:
            self._dst.minimize(constant, linear, quadratic)
        else:
            self._dst.maximize(constant, linear, quadratic)

        # For linear constraints
        for l_constraint in problem.linear_constraints:
            linear = l_constraint.linear.to_dict(use_name=True)
            self._dst.linear_constraint(
                linear, l_constraint.sense, l_constraint.rhs, l_constraint.name
            )
        # For quadratic constraints
        for q_constraint in problem.quadratic_constraints:
            linear = q_constraint.linear.to_dict(use_name=True)
            quadratic = q_constraint.quadratic.to_dict(use_name=True)
            self._dst.quadratic_constraint(
                linear,
                quadratic,
                q_constraint.sense,
                q_constraint.rhs,
                q_constraint.name,
            )
        # For indicator constraints
        for i_constraint in problem.indicator_constraints:
            self._convert_indicator_constraint(problem, i_constraint)

        return self._dst

    def _convert_indicator_constraint(self, problem, indicator_const):
        # convert indicator constraints to inequality constraints
        new_linear = indicator_const.linear.to_dict(use_name=True)
        new_rhs = indicator_const.rhs
        sense = indicator_const.sense
        new_name = indicator_const.name + self._delimiter + "indicator"
        if sense == Constraint.Sense.LE:
            _, lhs_ub = self._calc_linear_bounds(problem, new_linear)
            big_m = lhs_ub - new_rhs
            if indicator_const.active_value:
                new_linear[indicator_const.binary_var.name] = big_m
                new_rhs = new_rhs + big_m
            else:
                new_linear[indicator_const.binary_var.name] = -big_m
            self._dst.linear_constraint(new_linear, "<=", new_rhs, new_name)
        elif sense == Constraint.Sense.GE:
            lhs_lb, _ = self._calc_linear_bounds(problem, new_linear)
            big_m = new_rhs - lhs_lb
            if indicator_const.active_value:
                new_linear[indicator_const.binary_var.name] = - big_m
                new_rhs = new_rhs - big_m
            else:
                new_linear[indicator_const.binary_var.name] = big_m
            self._dst.linear_constraint(new_linear, ">=", new_rhs, new_name)
        elif sense == Constraint.Sense.EQ:
            # for equality constraints, add both GE and LE constraints.
            # new_linear2, new_rhs2, and big_m2 are for a >= constraint
            new_linear2 = indicator_const.linear.to_dict(use_name=True)
            new_rhs2 = indicator_const.rhs
            lhs_lb, lhs_ub = self._calc_linear_bounds(problem, new_linear)
            big_m = lhs_ub - new_rhs
            big_m2 = new_rhs - lhs_lb
            if indicator_const.active_value:
                new_linear[indicator_const.binary_var.name] = big_m
                new_rhs = new_rhs + big_m
                new_linear2[indicator_const.binary_var.name] = - big_m2
                new_rhs2 = new_rhs2 - big_m2
            else:
                new_linear[indicator_const.binary_var.name] = -big_m
                new_linear2[indicator_const.binary_var.name] = big_m2
            self._dst.linear_constraint(new_linear, "<=", new_rhs, new_name+"_LE")
            self._dst.linear_constraint(new_linear2, ">=", new_rhs2, new_name+"_GE")

    def _calc_linear_bounds(self, problem, linear):
        lhs_lb, lhs_ub = 0, 0
        for var_name, v in linear.items():
            x = problem.get_variable(var_name)
            lhs_lb += min(x.lowerbound * v, x.upperbound * v)
            lhs_ub += max(x.lowerbound * v, x.upperbound * v)
        return lhs_lb, lhs_ub

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
        if len(x) != self._src_num_var:
            raise QiskitOptimizationError(
                "The number of variables in the passed result differs from "
                "that of the original problem."
            )
        return np.asarray(x)
