# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Substitute variables of QuadraticProgram."""

import logging
from collections import defaultdict
from dataclasses import dataclass
from math import isclose
from typing import Dict, Optional, Tuple, Union, cast

from ..exceptions import QiskitOptimizationError
from ..infinity import INFINITY
from .constraint import ConstraintSense
from .linear_expression import LinearExpression
from .quadratic_expression import QuadraticExpression
from .quadratic_program import QuadraticProgram

logger = logging.getLogger(__name__)


@dataclass
class SubstitutionExpression:
    """Represents a substitution of a variable with a linear expression.

    If `variable` is `None`, it substitutes a variable with the constant value.
    Otherwise, it substitutes a variable with (constant + coefficient * new_variable).
    """

    const: float = 0.0
    """Constant value"""
    coeff: float = 0.0
    """Coefficient of the new variable"""
    variable: Optional[str] = None
    """Variable name or `None`"""


def substitute_variables(
    quadratic_program: QuadraticProgram,
    constants: Optional[Dict[Union[str, int], float]] = None,
    variables: Optional[Dict[Union[str, int], Tuple[Union[str, int], float]]] = None,
) -> QuadraticProgram:
    """Substitutes variables with constants or other variables.

    Args:
        quadratic_program: a quadratic program whose variables are substituted.

        constants: replace variable by constant
            e.g., {'x': 2} means 'x' is substituted with 2

        variables: replace variables by weighted other variable
            need to copy everything using name reference to make sure that indices are matched
            correctly. The lower and upper bounds are updated accordingly.
            e.g., {'x': ('y', 2)} means 'x' is substituted with 'y' * 2

    Returns:
        An optimization problem by substituting variables with constants or other variables.
        If the substitution is valid, `QuadraticProgram.status` is still
        `QuadraticProgram.Status.VALID`.
        Otherwise, it gets `QuadraticProgram.Status.INFEASIBLE`.

    Raises:
        QiskitOptimizationError: if the substitution is invalid as follows.
            - Same variable is substituted multiple times.
            - Coefficient of variable substitution is zero.
    """
    # guarantee that there is no overlap between variables to be replaced and combine input
    subs = {}
    if constants:
        for i, v in constants.items():
            # substitute i <- v
            i_2 = quadratic_program.get_variable(i).name
            if i_2 in subs:
                raise QiskitOptimizationError(
                    f"Cannot substitute the same variable twice: {i} <- {v}"
                )
            subs[i_2] = SubstitutionExpression(const=v)

    if variables:
        for i, (j, v) in variables.items():
            if v == 0:
                raise QiskitOptimizationError(f"coefficient must be non-zero: {i} {j} {v}")
            # substitute i <- j * v
            i_2 = quadratic_program.get_variable(i).name
            j_2 = quadratic_program.get_variable(j).name
            if i_2 == j_2:
                raise QiskitOptimizationError(
                    f"Cannot substitute the same variable: {i} <- {j} {v}"
                )
            if i_2 in subs:
                raise QiskitOptimizationError(
                    f"Cannot substitute the same variable twice: {i} <- {j} {v}"
                )
            if j_2 in subs:
                raise QiskitOptimizationError(
                    "Cannot substitute by variable that gets substituted itself: " f"{i} <- {j} {v}"
                )
            subs[i_2] = SubstitutionExpression(variable=j_2, coeff=v)

    return _SubstituteVariables().substitute_variables(quadratic_program, subs)


class _SubstituteVariables:
    """A class to substitute variables of an optimization problem with constants for other
    variables"""

    def __init__(self):
        self._src: Optional[QuadraticProgram] = None
        self._dst: Optional[QuadraticProgram] = None
        self._subs: Dict[str, SubstitutionExpression] = {}

    def substitute_variables(
        self, quadratic_program: QuadraticProgram, subs: Dict[str, SubstitutionExpression]
    ) -> QuadraticProgram:
        """Substitutes variables with constants or other variables.

        Args:
            quadratic_program: a quadratic program whose variables are substituted.

            subs: substitution expressions as a dictionary.
                e.g., {'x': SubstitutionExpression(const=1, coeff=2, variable='y'} means
                `x` is substituted with `1 + 2 * y`.

        Returns:
            An optimization problem by substituting variables with constants or other variables.
            If the substitution is valid, `QuadraticProgram.status` is still
            `QuadraticProgram.Status.VALID`.
            Otherwise, it gets `QuadraticProgram.Status.INFEASIBLE`.
        """
        self._src = quadratic_program
        self._dst = QuadraticProgram(quadratic_program.name)
        self._subs = subs
        results = [
            self._variables(),
            self._objective(),
            self._linear_constraints(),
            self._quadratic_constraints(),
        ]
        if not all(results):
            self._dst._status = QuadraticProgram.Status.INFEASIBLE
        return self._dst

    @staticmethod
    def _feasible(sense: ConstraintSense, rhs: float) -> bool:
        """Checks feasibility of the following condition
        0 `sense` rhs
        """
        # I use the following pylint option because `rhs` should come to right
        # pylint: disable=misplaced-comparison-constant
        if sense == ConstraintSense.EQ:
            if 0 == rhs:
                return True
        elif sense == ConstraintSense.LE:
            if 0 <= rhs:
                return True
        elif sense == ConstraintSense.GE:
            if 0 >= rhs:
                return True
        return False

    def _variables(self) -> bool:
        # copy variables that are not replaced
        feasible = True
        for var in self._src.variables:
            name = var.name
            vartype = var.vartype
            lowerbound = var.lowerbound
            upperbound = var.upperbound
            if name not in self._subs:
                self._dst._add_variable(lowerbound, upperbound, vartype, name)

        for i, expr in self._subs.items():
            lb_i = self._src.get_variable(i).lowerbound
            ub_i = self._src.get_variable(i).upperbound
            # substitute x_i <- x_j * coeff + const
            # lb_i <= x_i <= ub_i  -->
            #   (lb_i - const) / coeff <=  x_j  <= (ub_i - const) / coeff    if coeff > 0
            #   (ub_i - const) / coeff <=  x_j  <= (lb_i - const) / coeff    if coeff < 0
            #                     lb_i <= const <= ub_i                      if coeff == 0
            if isclose(expr.coeff, 0.0, abs_tol=1e-10):
                if not lb_i <= expr.const <= ub_i:
                    logger.warning("Infeasible substitution for variable: %s", i)
                    feasible = False
            else:
                if abs(lb_i) < INFINITY:
                    new_lb_i = (lb_i - expr.const) / expr.coeff
                else:
                    new_lb_i = lb_i if expr.coeff > 0 else -lb_i
                if abs(ub_i) < INFINITY:
                    new_ub_i = (ub_i - expr.const) / expr.coeff
                else:
                    new_ub_i = ub_i if expr.coeff > 0 else -ub_i
                var_j = self._dst.get_variable(expr.variable)
                lb_j = var_j.lowerbound
                ub_j = var_j.upperbound
                if expr.coeff > 0:
                    var_j.lowerbound = max(lb_j, new_lb_i)
                    var_j.upperbound = min(ub_j, new_ub_i)
                else:
                    var_j.lowerbound = max(lb_j, new_ub_i)
                    var_j.upperbound = min(ub_j, new_lb_i)

        for var in self._dst.variables:
            if var.lowerbound > var.upperbound:
                logger.warning(
                    "Infeasible lower and upper bounds: %s %f %f",
                    var,
                    var.lowerbound,
                    var.upperbound,
                )
                feasible = False

        return feasible

    def _linear_expression(self, lin_expr: LinearExpression) -> Tuple[float, LinearExpression]:
        const = 0.0
        lin_dict: Dict[str, float] = defaultdict(float)
        for i, w_i in lin_expr.to_dict(use_name=True).items():
            i = cast(str, i)
            expr_i = self._subs.get(i, SubstitutionExpression(coeff=1, variable=i))
            const += w_i * expr_i.const
            if expr_i.variable:
                lin_dict[expr_i.variable] += w_i * expr_i.coeff
        new_lin = LinearExpression(
            quadratic_program=self._dst, coefficients=lin_dict if lin_dict else {}
        )
        return const, new_lin

    def _quadratic_expression(
        self, quad_expr: QuadraticExpression
    ) -> Tuple[float, Optional[LinearExpression], Optional[QuadraticExpression]]:
        const = 0.0
        lin_dict: Dict[str, float] = defaultdict(float)
        quad_dict: Dict[Tuple[str, str], float] = defaultdict(float)
        for (i, j), w_ij in quad_expr.to_dict(use_name=True).items():
            i = cast(str, i)
            j = cast(str, j)
            expr_i = self._subs.get(i, SubstitutionExpression(coeff=1, variable=i))
            expr_j = self._subs.get(j, SubstitutionExpression(coeff=1, variable=j))
            const += w_ij * expr_i.const * expr_j.const
            if expr_i.variable:
                lin_dict[expr_i.variable] += w_ij * expr_i.coeff * expr_j.const
            if expr_j.variable:
                lin_dict[expr_j.variable] += w_ij * expr_j.coeff * expr_i.const
            if expr_i.variable and expr_j.variable:
                quad_dict[expr_i.variable, expr_j.variable] += w_ij * expr_i.coeff * expr_j.coeff
        new_lin = LinearExpression(
            quadratic_program=self._dst, coefficients=lin_dict if lin_dict else {}
        )
        new_quad = QuadraticExpression(
            quadratic_program=self._dst, coefficients=quad_dict if quad_dict else {}
        )
        return const, new_lin, new_quad

    def _objective(self) -> bool:
        obj = self._src.objective
        const1, lin1 = self._linear_expression(obj.linear)
        const2, lin2, quadratic = self._quadratic_expression(obj.quadratic)

        constant = obj.constant + const1 + const2
        linear = lin1.coefficients + lin2.coefficients
        if obj.sense == obj.sense.MINIMIZE:
            self._dst.minimize(constant=constant, linear=linear, quadratic=quadratic.coefficients)
        else:
            self._dst.maximize(constant=constant, linear=linear, quadratic=quadratic.coefficients)
        return True

    def _linear_constraints(self) -> bool:
        feasible = True
        for lin_cst in self._src.linear_constraints:
            constant, linear = self._linear_expression(lin_cst.linear)
            rhs = lin_cst.rhs - constant
            if linear.coefficients.nnz > 0:
                self._dst.linear_constraint(
                    name=lin_cst.name,
                    linear=linear.coefficients,
                    sense=lin_cst.sense,
                    rhs=rhs,
                )
            else:
                if not self._feasible(lin_cst.sense, rhs):
                    logger.warning("constraint %s is infeasible due to substitution", lin_cst.name)
                    feasible = False
        return feasible

    def _quadratic_constraints(self) -> bool:
        feasible = True
        for quad_cst in self._src.quadratic_constraints:
            const1, lin1 = self._linear_expression(quad_cst.linear)
            const2, lin2, quadratic = self._quadratic_expression(quad_cst.quadratic)
            rhs = quad_cst.rhs - const1 - const2
            linear = lin1.coefficients + lin2.coefficients

            if quadratic.coefficients.nnz > 0:
                self._dst.quadratic_constraint(
                    name=quad_cst.name,
                    linear=linear,
                    quadratic=quadratic.coefficients,
                    sense=quad_cst.sense,
                    rhs=rhs,
                )
            elif linear.nnz > 0:
                name = quad_cst.name
                lin_names = set(lin.name for lin in self._dst.linear_constraints)
                while name in lin_names:
                    name = "_" + name
                self._dst.linear_constraint(name=name, linear=linear, sense=quad_cst.sense, rhs=rhs)
            else:
                if not self._feasible(quad_cst.sense, rhs):
                    logger.warning("constraint %s is infeasible due to substitution", quad_cst.name)
                    feasible = False
        return feasible
