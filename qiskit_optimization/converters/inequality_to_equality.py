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
import math
from typing import List, Optional, Union, cast

import numpy as np

from ..exceptions import QiskitOptimizationError
from ..problems.constraint import Constraint
from ..problems.linear_constraint import LinearConstraint
from ..problems.quadratic_constraint import QuadraticConstraint
from ..problems.quadratic_objective import QuadraticObjective
from ..problems.quadratic_program import QuadraticProgram
from ..problems.variable import Variable
from .quadratic_program_converter import QuadraticProgramConverter


class InequalityToEquality(QuadraticProgramConverter):
    """Convert inequality constraints into equality constraints by introducing slack variables.

    Examples:
        >>> from qiskit_optimization.problems import QuadraticProgram
        >>> from qiskit_optimization.converters import InequalityToEquality
        >>> problem = QuadraticProgram()
        >>> # define a problem
        >>> conv = InequalityToEquality()
        >>> problem2 = conv.convert(problem)
    """

    _delimiter = "@"  # users are supposed not to use this character in variable names

    def __init__(self, mode: str = "auto") -> None:
        """
        Args:
            mode: To chose the type of slack variables. There are 3 options for mode.

                - 'integer': All slack variables will be integer variables.
                - 'continuous': All slack variables will be continuous variables
                - 'auto': Try to use integer variables but if it's not possible,
                   use continuous variables
        """
        self._src: Optional[QuadraticProgram] = None
        self._dst: Optional[QuadraticProgram] = None
        self._mode = mode

    def convert(self, problem: QuadraticProgram) -> QuadraticProgram:
        """Convert a problem with inequality constraints into one with only equality constraints.

        Args:
            problem: The problem to be solved, that may contain inequality constraints.

        Returns:
            The converted problem, that contain only equality constraints.

        Raises:
            QiskitOptimizationError: If a variable type is not supported.
            QiskitOptimizationError: If an unsupported mode is selected.
            QiskitOptimizationError: If an unsupported sense is specified.
        """
        self._src = copy.deepcopy(problem)
        self._dst = QuadraticProgram(name=problem.name)

        # set a converting mode
        mode = self._mode
        if mode not in ["integer", "continuous", "auto"]:
            raise QiskitOptimizationError(f"Unsupported mode is selected: {mode}")

        # Copy variables
        for x in self._src.variables:
            if x.vartype == Variable.Type.BINARY:
                self._dst.binary_var(name=x.name)
            elif x.vartype == Variable.Type.INTEGER:
                self._dst.integer_var(name=x.name, lowerbound=x.lowerbound, upperbound=x.upperbound)
            elif x.vartype == Variable.Type.CONTINUOUS:
                self._dst.continuous_var(
                    name=x.name, lowerbound=x.lowerbound, upperbound=x.upperbound
                )
            else:
                raise QiskitOptimizationError(f"Unsupported variable type {x.vartype}")

        # Copy the objective function
        constant = self._src.objective.constant
        linear = self._src.objective.linear.to_dict(use_name=True)
        quadratic = self._src.objective.quadratic.to_dict(use_name=True)
        if self._src.objective.sense == QuadraticObjective.Sense.MINIMIZE:
            self._dst.minimize(constant, linear, quadratic)
        else:
            self._dst.maximize(constant, linear, quadratic)

        # For linear constraints
        for lin_const in self._src.linear_constraints:
            if lin_const.sense == Constraint.Sense.EQ:
                self._dst.linear_constraint(
                    lin_const.linear.coefficients, lin_const.sense, lin_const.rhs, lin_const.name
                )
            elif lin_const.sense in [Constraint.Sense.LE, Constraint.Sense.GE]:
                self._add_slack_var_linear_constraint(lin_const)
            else:
                raise QiskitOptimizationError(
                    f"Internal error: type of sense in {lin_const.name} is not supported: "
                    f"{lin_const.sense}"
                )

        # For quadratic constraints
        for quad_const in self._src.quadratic_constraints:
            if quad_const.sense == Constraint.Sense.EQ:
                self._dst.quadratic_constraint(
                    quad_const.linear.coefficients,
                    quad_const.quadratic.coefficients,
                    quad_const.sense,
                    quad_const.rhs,
                    quad_const.name,
                )
            elif quad_const.sense in [Constraint.Sense.LE, Constraint.Sense.GE]:
                self._add_slack_var_quadratic_constraint(quad_const)
            else:
                raise QiskitOptimizationError(
                    f"Internal error: type of sense in {quad_const.name} is not supported: "
                    f"{quad_const.sense}"
                )

        return self._dst

    def _add_slack_var_linear_constraint(self, constraint: LinearConstraint):
        linear = constraint.linear
        sense = constraint.sense
        name = constraint.name

        any_float = self._any_float(linear.to_array())
        mode = self._mode
        if mode == "integer":
            if any_float:
                raise QiskitOptimizationError(
                    f'"{name}" contains float coefficients. '
                    'We can not use an integer slack variable for "{name}"'
                )
        elif mode == "auto":
            mode = "continuous" if any_float else "integer"

        new_rhs = constraint.rhs
        if mode == "integer":
            # If rhs is float number, round up/down to the nearest integer.
            if sense == Constraint.Sense.LE:
                new_rhs = math.floor(new_rhs)
            if sense == Constraint.Sense.GE:
                new_rhs = math.ceil(new_rhs)

        lin_bounds = linear.bounds
        lhs_lb = lin_bounds.lowerbound
        lhs_ub = lin_bounds.upperbound

        var_ub = 0.0
        sign = 0
        if sense == Constraint.Sense.LE:
            var_ub = new_rhs - lhs_lb
            if var_ub > 0:
                sign = 1
        elif sense == Constraint.Sense.GE:
            var_ub = lhs_ub - new_rhs
            if var_ub > 0:
                sign = -1

        new_linear = linear.to_dict(use_name=True)
        if var_ub > 0:
            # Add a slack variable.
            mode_name = {"integer": "int", "continuous": "continuous"}
            slack_name = f"{name}{self._delimiter}{mode_name[mode]}_slack"
            if mode == "integer":
                self._dst.integer_var(name=slack_name, lowerbound=0, upperbound=var_ub)
            elif mode == "continuous":
                self._dst.continuous_var(name=slack_name, lowerbound=0, upperbound=var_ub)
            new_linear[slack_name] = sign
        self._dst.linear_constraint(new_linear, "==", new_rhs, name)

    def _add_slack_var_quadratic_constraint(self, constraint: QuadraticConstraint):
        quadratic = constraint.quadratic
        linear = constraint.linear
        sense = constraint.sense
        name = constraint.name

        any_float = self._any_float(linear.to_array()) or self._any_float(quadratic.to_array())
        mode = self._mode
        if mode == "integer":
            if any_float:
                raise QiskitOptimizationError(
                    f'"{name}" contains float coefficients. '
                    'We can not use an integer slack variable for "{name}"'
                )
        elif mode == "auto":
            mode = "continuous" if any_float else "integer"

        new_rhs = constraint.rhs
        if mode == "integer":
            # If rhs is float number, round up/down to the nearest integer.
            if sense == Constraint.Sense.LE:
                new_rhs = math.floor(new_rhs)
            if sense == Constraint.Sense.GE:
                new_rhs = math.ceil(new_rhs)

        lin_bounds = linear.bounds
        quad_bounds = quadratic.bounds
        lhs_lb = lin_bounds.lowerbound + quad_bounds.lowerbound
        lhs_ub = lin_bounds.upperbound + quad_bounds.upperbound

        var_ub = 0.0
        sign = 0
        if sense == Constraint.Sense.LE:
            var_ub = new_rhs - lhs_lb
            if var_ub > 0:
                sign = 1
        elif sense == Constraint.Sense.GE:
            var_ub = lhs_ub - new_rhs
            if var_ub > 0:
                sign = -1

        new_linear = linear.to_dict(use_name=True)
        if var_ub > 0:
            # Add a slack variable.
            mode_name = {"integer": "int", "continuous": "continuous"}
            slack_name = f"{name}{self._delimiter}{mode_name[mode]}_slack"
            if mode == "integer":
                self._dst.integer_var(name=slack_name, lowerbound=0, upperbound=var_ub)
            elif mode == "continuous":
                self._dst.continuous_var(name=slack_name, lowerbound=0, upperbound=var_ub)
            new_linear[slack_name] = sign
        self._dst.quadratic_constraint(new_linear, quadratic.coefficients, "==", new_rhs, name)

    def interpret(self, x: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Convert a result of a converted problem into that of the original problem.

        Args:
            x: The result of the converted problem or the given result in case of FAILURE.

        Returns:
            The result of the original problem.
        """
        # convert back the optimization result into that of the original problem
        names = [var.name for var in self._dst.variables]

        # interpret slack variables
        sol = {name: x[i] for i, name in enumerate(names)}
        new_x = np.zeros(self._src.get_num_vars())
        for i, var in enumerate(self._src.variables):
            new_x[i] = sol[var.name]
        return new_x

    @staticmethod
    def _any_float(values: np.ndarray) -> bool:
        """Check whether the list contains float or not.
        This method is used to check whether a constraint contain float coefficients or not.

        Args:
            values: Coefficients of the constraint

        Returns:
            bool: If the constraint contains float coefficients, this returns True, else False.
        """
        return any(isinstance(v, float) and not cast(float, v).is_integer() for v in values)

    @property
    def mode(self) -> str:
        """Returns the mode of the converter

        Returns:
            The mode of the converter used for additional slack variables
        """
        return self._mode

    @mode.setter
    def mode(self, mode: str) -> None:
        """Set a new mode for the converter

        Args:
            mode: The new mode for the converter
        """
        self._mode = mode
