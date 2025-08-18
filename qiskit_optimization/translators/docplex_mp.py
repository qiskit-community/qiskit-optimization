# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Translator between a docplex.mp model and a quadratic program"""
from __future__ import annotations

from math import isclose
from typing import Any, cast
from warnings import warn

from docplex.mp.basic import Expr
from docplex.mp.constants import ComparisonType
from docplex.mp.constr import (
    IndicatorConstraint,
    LinearConstraint,
    NotEqualConstraint,
    QuadraticConstraint,
)
from docplex.mp.dvar import Var
from docplex.mp.linear import AbstractLinearExpr
from docplex.mp.model import Model
from docplex.mp.quad import QuadExpr
from docplex.mp.vartype import BinaryVarType, ContinuousVarType, IntegerVarType

from qiskit_optimization.exceptions import QiskitOptimizationError
from qiskit_optimization.problems.constraint import Constraint
from qiskit_optimization.problems.quadratic_objective import QuadraticObjective
from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from qiskit_optimization.problems.variable import Variable


def to_docplex_mp(quadratic_program: QuadraticProgram) -> Model:
    """Returns a docplex.mp model corresponding to a quadratic program.

    Args:
        quadratic_program: The quadratic program to be translated.

    Returns:
        The docplex.mp model corresponding to a quadratic program.

    Raises:
        QiskitOptimizationError: if the model contains non-supported elements (should never happen).
    """
    # initialize model
    mdl = Model(quadratic_program.name)

    # add variables
    var = {}
    for idx, x in enumerate(quadratic_program.variables):
        if x.vartype == Variable.Type.CONTINUOUS:
            var[idx] = mdl.continuous_var(lb=x.lowerbound, ub=x.upperbound, name=x.name)
        elif x.vartype == Variable.Type.BINARY:
            var[idx] = mdl.binary_var(name=x.name)
        elif x.vartype == Variable.Type.INTEGER:
            var[idx] = mdl.integer_var(lb=x.lowerbound, ub=x.upperbound, name=x.name)
        else:
            # should never happen
            raise QiskitOptimizationError(f"Internal error: unsupported variable type: {x.vartype}")

    # add objective
    objective = (
        quadratic_program.objective.constant
        + mdl.sum(
            v * var[cast(int, i)] for i, v in quadratic_program.objective.linear.to_dict().items()
        )
        + mdl.sum(
            v * var[cast(int, i)] * var[cast(int, j)]
            for (i, j), v in quadratic_program.objective.quadratic.to_dict().items()
        )
    )
    if quadratic_program.objective.sense == QuadraticObjective.Sense.MINIMIZE:
        mdl.minimize(objective)
    else:
        mdl.maximize(objective)

    # add linear constraints
    for l_constraint in quadratic_program.linear_constraints:
        name = l_constraint.name
        rhs = l_constraint.rhs
        if rhs == 0 and l_constraint.linear.coefficients.nnz == 0:
            continue
        linear_expr = mdl.sum(
            v * var[cast(int, j)] for j, v in l_constraint.linear.to_dict().items()
        )
        sense = l_constraint.sense
        if sense == Constraint.Sense.EQ:
            mdl.add_constraint(linear_expr == rhs, ctname=name)
        elif sense == Constraint.Sense.GE:
            mdl.add_constraint(linear_expr >= rhs, ctname=name)
        elif sense == Constraint.Sense.LE:
            mdl.add_constraint(linear_expr <= rhs, ctname=name)
        else:
            # should never happen
            raise QiskitOptimizationError(f"Internal error: unsupported constraint sense: {sense}")

    # add quadratic constraints
    for q_constraint in quadratic_program.quadratic_constraints:
        name = q_constraint.name
        rhs = q_constraint.rhs
        if (
            rhs == 0
            and q_constraint.linear.coefficients.nnz == 0
            and q_constraint.quadratic.coefficients.nnz == 0
        ):
            continue
        quadratic_expr = mdl.sum(
            v * var[cast(int, j)] for j, v in q_constraint.linear.to_dict().items()
        ) + mdl.sum(
            v * var[cast(int, j)] * var[cast(int, k)]
            for (j, k), v in q_constraint.quadratic.to_dict().items()
        )
        sense = q_constraint.sense
        if sense == Constraint.Sense.EQ:
            mdl.add_constraint(quadratic_expr == rhs, ctname=name)
        elif sense == Constraint.Sense.GE:
            mdl.add_constraint(quadratic_expr >= rhs, ctname=name)
        elif sense == Constraint.Sense.LE:
            mdl.add_constraint(quadratic_expr <= rhs, ctname=name)
        else:
            # should never happen
            raise QiskitOptimizationError(f"Internal error: unsupported constraint sense: {sense}")

    return mdl


# from_docplex_mp


class _FromDocplexMp:
    _sense_dict = {ComparisonType.EQ: "==", ComparisonType.LE: "<=", ComparisonType.GE: ">="}

    def __init__(self, model: Model):
        """
        Args:
            model: Docplex model
        """
        self._model: Model = model
        self._quadratic_program: QuadraticProgram = QuadraticProgram()
        self._var_names: dict[Var, str] = {}
        self._var_bounds: dict[str, tuple[float, float]] = {}

    def _variables(self):
        # keep track of names separately, since docplex allows to have None names.
        for x in self._model.iter_variables():
            if isinstance(x.vartype, ContinuousVarType):
                x_new = self._quadratic_program.continuous_var(x.lb, x.ub, x.name)
            elif isinstance(x.vartype, BinaryVarType):
                x_new = self._quadratic_program.binary_var(x.name)
            elif isinstance(x.vartype, IntegerVarType):
                x_new = self._quadratic_program.integer_var(x.lb, x.ub, x.name)
            else:
                raise QiskitOptimizationError(f"Unsupported variable type: {x.name} {x.vartype}")
            self._var_names[x] = x_new.name
            self._var_bounds[x.name] = (x_new.lowerbound, x_new.upperbound)

    def _linear_expr(self, expr: AbstractLinearExpr) -> dict[str, float]:
        # AbstractLinearExpr is a parent of LinearExpr, ConstantExpr, and ZeroExpr
        linear = {}
        for x, coeff in expr.iter_terms():
            linear[self._var_names[x]] = coeff
        return linear

    def _quadratic_expr(
        self, expr: QuadExpr
    ) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
        linear = self._linear_expr(expr.get_linear_part())
        quad = {}
        for x, y, coeff in expr.iter_quad_triplets():
            i = self._var_names[x]
            j = self._var_names[y]
            quad[i, j] = coeff
        return linear, quad

    def quadratic_program(self, indicator_big_m: float | None) -> QuadraticProgram:
        """Generate a quadratic program corresponding to the input Docplex model.

        Args:
            indicator_big_m: The big-M value used for the big-M formulation to convert
            indicator constraints into linear constraints.
            If ``None``, it is automatically derived from the model.

        Returns:
            a quadratic program corresponding to the input Docplex model.

        """
        self._quadratic_program = QuadraticProgram(self._model.name)

        # prepare variables
        self._variables()

        # objective sense
        minimize = self._model.objective_sense.is_minimize()

        # make sure objective expression is linear or quadratic and not a variable
        if isinstance(self._model.objective_expr, Var):
            self._model.objective_expr = self._model.objective_expr + 0  # Var + 0 -> LinearExpr

        constant = self._model.objective_expr.constant
        if isinstance(self._model.objective_expr, QuadExpr):
            linear, quadratic = self._quadratic_expr(self._model.objective_expr)
        else:
            linear = self._linear_expr(self._model.objective_expr.get_linear_part())
            quadratic = {}

        # set objective
        if minimize:
            self._quadratic_program.minimize(constant, linear, quadratic)
        else:
            self._quadratic_program.maximize(constant, linear, quadratic)

        # set linear constraints
        for constraint in self._model.iter_linear_constraints():
            linear, sense, rhs = self._linear_constraint(constraint)
            if not linear:  # lhs == 0
                warn(f"Trivial constraint: {constraint}", stacklevel=3)
            self._quadratic_program.linear_constraint(linear, sense, rhs, constraint.name)

        # set quadratic constraints
        for constraint in self._model.iter_quadratic_constraints():
            linear, quadratic, sense, rhs = self._quadratic_constraint(constraint)
            if not linear and not quadratic:  # lhs == 0
                warn(f"Trivial constraint: {constraint}", stacklevel=3)
            self._quadratic_program.quadratic_constraint(
                linear, quadratic, sense, rhs, constraint.name
            )

        # set indicator constraints
        for index, constraint in enumerate(self._model.iter_indicator_constraints()):
            linear, _, _ = self._linear_constraint(constraint.linear_constraint)
            if not linear:  # lhs == 0
                warn(f"Trivial constraint: {constraint}", stacklevel=3)
            prefix = constraint.name or f"ind{index}"
            linear_constraints = self._indicator_constraints(constraint, prefix, indicator_big_m)
            for linear, sense, rhs, name in linear_constraints:
                self._quadratic_program.linear_constraint(linear, sense, rhs, name)

        return self._quadratic_program

    @staticmethod
    def _subtract(dict1: dict[Any, float], dict2: dict[Any, float]) -> dict[Any, float]:
        """Calculate dict1 - dict2"""
        ret = dict1.copy()
        for key, val2 in dict2.items():
            if key in dict1:
                val1 = ret[key]
                if isclose(val1, val2):
                    del ret[key]
                else:
                    ret[key] -= val2
            else:
                ret[key] = -val2
        return ret

    def _linear_constraint(
        self, constraint: LinearConstraint
    ) -> tuple[dict[str, float], str, float]:
        left_expr = constraint.get_left_expr()
        right_expr = constraint.get_right_expr()
        # for linear constraints we may get an instance of Var instead of expression,
        # e.g. x + y = z
        if not isinstance(left_expr, (Expr, Var)):
            raise QiskitOptimizationError(f"Unsupported expression: {left_expr} {type(left_expr)}")
        if not isinstance(right_expr, (Expr, Var)):
            raise QiskitOptimizationError(
                f"Unsupported expression: {right_expr} {type(right_expr)}"
            )
        if constraint.sense not in self._sense_dict:
            raise QiskitOptimizationError(f"Unsupported constraint sense: {constraint}")

        if isinstance(left_expr, Var):
            left_expr = left_expr + 0  # Var + 0 -> LinearExpr
        left_linear = self._linear_expr(left_expr)

        if isinstance(right_expr, Var):
            right_expr = right_expr + 0
        right_linear = self._linear_expr(right_expr)

        linear = self._subtract(left_linear, right_linear)
        rhs = right_expr.constant - left_expr.constant
        return linear, self._sense_dict[constraint.sense], rhs

    def _quadratic_constraint(
        self, constraint: QuadraticConstraint
    ) -> tuple[dict[str, float], dict[tuple[str, str], float], str, float]:
        left_expr = constraint.get_left_expr()
        right_expr = constraint.get_right_expr()
        if not isinstance(left_expr, (Expr, Var)):
            raise QiskitOptimizationError(f"Unsupported expression: {left_expr} {type(left_expr)}")
        if not isinstance(right_expr, (Expr, Var)):
            raise QiskitOptimizationError(
                f"Unsupported expression: {right_expr} {type(right_expr)}"
            )
        if constraint.sense not in self._sense_dict:
            raise QiskitOptimizationError(f"Unsupported constraint sense: {constraint}")

        if isinstance(left_expr, Var):
            left_expr = left_expr + 0  # Var + 0 -> LinearExpr
        if left_expr.is_quad_expr():
            left_lin, left_quad = self._quadratic_expr(left_expr)
        else:
            left_lin = self._linear_expr(left_expr)
            left_quad = {}

        if isinstance(right_expr, Var):
            right_expr = right_expr + 0
        if right_expr.is_quad_expr():
            right_lin, right_quad = self._quadratic_expr(right_expr)
        else:
            right_lin = self._linear_expr(right_expr)
            right_quad = {}

        linear = self._subtract(left_lin, right_lin)
        quadratic = self._subtract(left_quad, right_quad)
        rhs = right_expr.constant - left_expr.constant
        return linear, quadratic, self._sense_dict[constraint.sense], rhs

    def _linear_bounds(self, linear: dict[str, float]):
        linear_lb = 0.0
        linear_ub = 0.0
        for var_name, val in linear.items():
            x_lb, x_ub = self._var_bounds[var_name]
            x_lb *= val
            x_ub *= val
            linear_lb += min(x_lb, x_ub)
            linear_ub += max(x_lb, x_ub)
        return linear_lb, linear_ub

    def _indicator_constraints(
        self,
        constraint: IndicatorConstraint,
        name: str,
        indicator_big_m: float | None = None,
    ):
        binary_var = constraint.binary_var
        active_value = constraint.active_value
        linear_constraint = constraint.linear_constraint
        linear, sense, rhs = self._linear_constraint(linear_constraint)
        linear_lb, linear_ub = self._linear_bounds(linear)
        ret = []
        if sense in ["<=", "=="]:
            big_m = max(0.0, linear_ub - rhs) if indicator_big_m is None else indicator_big_m
            if active_value:
                # rhs += big_m * (1 - binary_var)
                linear2 = self._subtract(linear, {binary_var.name: -big_m})
                rhs2 = rhs + big_m
            else:
                # rhs += big_m * binary_var
                linear2 = self._subtract(linear, {binary_var.name: big_m})
                rhs2 = rhs
            name2 = name + "_LE" if sense == "==" else name
            ret.append((linear2, "<=", rhs2, name2))
        if sense in [">=", "=="]:
            big_m = max(0.0, rhs - linear_lb) if indicator_big_m is None else indicator_big_m
            if active_value:
                # rhs += -big_m * (1 - binary_var)
                linear2 = self._subtract(linear, {binary_var.name: big_m})
                rhs2 = rhs - big_m
            else:
                # rhs += -big_m * binary_var
                linear2 = self._subtract(linear, {binary_var.name: -big_m})
                rhs2 = rhs
            name2 = name + "_GE" if sense == "==" else name
            ret.append((linear2, ">=", rhs2, name2))
        if sense not in ["<=", ">=", "=="]:
            raise QiskitOptimizationError(
                f"Internal error: invalid sense of indicator constraint: {sense}"
            )
        return ret


def from_docplex_mp(model: Model, indicator_big_m: float | None = None) -> QuadraticProgram:
    """Translate a docplex.mp model into a quadratic program.

    Note that this supports the following features of docplex:

    - linear / quadratic objective function
    - linear / quadratic / indicator constraints
    - binary / integer / continuous variables
    - logical expressions (``logical_not``, ``logical_and``, and ``logical_or``)

    Args:
        model: The docplex.mp model to be loaded.
        indicator_big_m: The big-M value used for the big-M formulation to convert
            indicator constraints into linear constraints.
            If ``None``, it is automatically derived from the model.

    Returns:
        The quadratic program corresponding to the model.

    Raises:
        QiskitOptimizationError: if the model contains unsupported elements.
    """
    if not isinstance(model, Model):
        raise QiskitOptimizationError(f"The model is not compatible: {model}")

    if model.number_of_user_cut_constraints > 0:
        raise QiskitOptimizationError("User cut constraints are not supported")

    if model.number_of_lazy_constraints > 0:
        raise QiskitOptimizationError("Lazy constraints are not supported")

    if model.number_of_sos > 0:
        raise QiskitOptimizationError("SOS sets are not supported")

    # check constraint type
    for constraint in model.iter_constraints():
        # If any constraint is not linear/quadratic/indicator constraints, it raises an error.
        if isinstance(constraint, LinearConstraint):
            if isinstance(constraint, NotEqualConstraint):
                # Notice that NotEqualConstraint is a subclass of Docplex's LinearConstraint,
                # but it cannot be handled by optimization.
                raise QiskitOptimizationError(f"Unsupported constraint: {constraint}")
        elif not isinstance(constraint, (QuadraticConstraint, IndicatorConstraint)):
            raise QiskitOptimizationError(f"Unsupported constraint: {constraint}")

    return _FromDocplexMp(model).quadratic_program(indicator_big_m)
