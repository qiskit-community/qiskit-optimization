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

"""Translator between a docplex.mp model and a quadratic program"""

from typing import Dict, Optional, Tuple, cast

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
    objective = quadratic_program.objective.constant
    for i, v in quadratic_program.objective.linear.to_dict().items():
        objective += v * var[cast(int, i)]
    for (i, j), v in quadratic_program.objective.quadratic.to_dict().items():
        objective += v * var[cast(int, i)] * var[cast(int, j)]
    if quadratic_program.objective.sense == QuadraticObjective.Sense.MINIMIZE:
        mdl.minimize(objective)
    else:
        mdl.maximize(objective)

    # add linear constraints
    for i, l_constraint in enumerate(quadratic_program.linear_constraints):
        name = l_constraint.name
        rhs = l_constraint.rhs
        if rhs == 0 and l_constraint.linear.coefficients.nnz == 0:
            continue
        linear_expr = 0
        for j, v in l_constraint.linear.to_dict().items():
            linear_expr += v * var[cast(int, j)]
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
    for i, q_constraint in enumerate(quadratic_program.quadratic_constraints):
        name = q_constraint.name
        rhs = q_constraint.rhs
        if (
            rhs == 0
            and q_constraint.linear.coefficients.nnz == 0
            and q_constraint.quadratic.coefficients.nnz == 0
        ):
            continue
        quadratic_expr = 0
        for j, v in q_constraint.linear.to_dict().items():
            quadratic_expr += v * var[cast(int, j)]
        for (j, k), v in q_constraint.quadratic.to_dict().items():
            quadratic_expr += v * var[cast(int, j)] * var[cast(int, k)]
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

    @classmethod
    def _linear_constraint(
        cls, var_names: Dict[Var, str], constraint: LinearConstraint
    ) -> Tuple[Dict[str, float], str, float]:
        left_expr = constraint.get_left_expr()
        right_expr = constraint.get_right_expr()
        # for linear constraints we may get an instance of Var instead of expression,
        # e.g. x + y = z
        if not isinstance(left_expr, (AbstractLinearExpr, Var)):
            raise QiskitOptimizationError(f"Unsupported expression: {left_expr} {type(left_expr)}")
        if not isinstance(right_expr, (AbstractLinearExpr, Var)):
            raise QiskitOptimizationError(
                f"Unsupported expression: {right_expr} {type(right_expr)}"
            )
        if isinstance(left_expr, Var):
            left_expr = left_expr + 0
        if isinstance(right_expr, Var):
            right_expr = right_expr + 0

        linear = {}
        for x in left_expr.iter_variables():
            linear[var_names[x]] = left_expr.get_coef(x)
        for x in right_expr.iter_variables():
            linear[var_names[x]] = linear.get(var_names[x], 0.0) - right_expr.get_coef(x)

        rhs = right_expr.constant - left_expr.constant

        if constraint.sense not in cls._sense_dict:
            raise QiskitOptimizationError(f"Unsupported constraint sense: {constraint}")

        return linear, cls._sense_dict[constraint.sense], rhs

    @classmethod
    def _quadratic_constraint(
        cls, var_names: Dict[Var, str], constraint: QuadraticConstraint
    ) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float], str, float]:
        left_expr = constraint.get_left_expr()
        right_expr = constraint.get_right_expr()
        if not isinstance(left_expr, (QuadExpr, AbstractLinearExpr, Var)):
            raise QiskitOptimizationError(f"Unsupported expression: {left_expr} {type(left_expr)}")
        if not isinstance(right_expr, (QuadExpr, AbstractLinearExpr, Var)):
            raise QiskitOptimizationError(
                f"Unsupported expression: {right_expr} {type(right_expr)}"
            )

        lin = {}
        quad = {}

        if left_expr.is_quad_expr():
            for x in left_expr.linear_part.iter_variables():
                lin[var_names[x]] = left_expr.linear_part.get_coef(x)
            for quad_triplet in left_expr.iter_quad_triplets():
                i = var_names[quad_triplet[0]]
                j = var_names[quad_triplet[1]]
                v = quad_triplet[2]
                quad[i, j] = v
        else:
            for x in left_expr.iter_variables():
                lin[var_names[x]] = left_expr.get_coef(x)

        if right_expr.is_quad_expr():
            for x in right_expr.linear_part.iter_variables():
                lin[var_names[x]] = lin.get(var_names[x], 0.0) - right_expr.linear_part.get_coef(x)
            for quad_triplet in right_expr.iter_quad_triplets():
                i = var_names[quad_triplet[0]]
                j = var_names[quad_triplet[1]]
                v = quad_triplet[2]
                quad[i, j] = quad.get((i, j), 0.0) - v
        else:
            for x in right_expr.iter_variables():
                lin[var_names[x]] = lin.get(var_names[x], 0.0) - right_expr.get_coef(x)

        rhs = right_expr.constant - left_expr.constant

        if constraint.sense not in cls._sense_dict:
            raise QiskitOptimizationError(f"Unsupported constraint sense: {constraint}")

        return lin, quad, cls._sense_dict[constraint.sense], rhs

    @staticmethod
    def _linear_bounds(var_bounds: Dict[str, Tuple[float, float]], linear: Dict[str, float]):
        linear_lb = 0.0
        linear_ub = 0.0
        for var_name, val in linear.items():
            x_lb, x_ub = var_bounds[var_name]
            x_lb *= val
            x_ub *= val
            linear_lb += min(x_lb, x_ub)
            linear_ub += max(x_lb, x_ub)
        return linear_lb, linear_ub

    @classmethod
    def _indicator_constraints(
        cls,
        var_names: Dict[Var, str],
        var_bounds: Dict[str, Tuple[float, float]],
        constraint: IndicatorConstraint,
        indicator_big_m: Optional[float] = None,
    ):
        name = constraint.name
        binary_var = constraint.binary_var
        active_value = constraint.active_value
        linear_constraint = constraint.linear_constraint
        linear, sense, rhs = cls._linear_constraint(var_names, linear_constraint)
        linear_lb, linear_ub = cls._linear_bounds(var_bounds, linear)
        if sense == "<=":
            big_m = max(0.0, linear_ub - rhs) if indicator_big_m is None else indicator_big_m
            if active_value:
                linear[binary_var.name] = big_m
                rhs += big_m
            else:
                linear[binary_var.name] = -big_m
            return [(linear, sense, rhs, name)]
        elif sense == ">=":
            big_m = max(0.0, rhs - linear_lb) if indicator_big_m is None else indicator_big_m
            if active_value:
                linear[binary_var.name] = -big_m
                rhs -= big_m
            else:
                linear[binary_var.name] = big_m
            return [(linear, sense, rhs, name)]
        elif sense == "==":
            # for equality constraints, add both GE and LE constraints.
            # linear2, rhs2, and big_m2 are for the GE constraint.
            linear2 = linear.copy()
            rhs2 = rhs
            big_m = max(0.0, linear_ub - rhs) if indicator_big_m is None else indicator_big_m
            big_m2 = max(0.0, rhs - linear_lb) if indicator_big_m is None else indicator_big_m
            if active_value:
                linear[binary_var.name] = big_m
                rhs += big_m
                linear2[binary_var.name] = -big_m2
                rhs2 -= big_m2
            else:
                linear[binary_var.name] = -big_m
                linear2[binary_var.name] = big_m2
            return [(linear, "<=", rhs, name + "_LE"), (linear2, ">=", rhs2, name + "_GE")]
        else:
            raise QiskitOptimizationError(
                f"Internal error: invalid sense of indicator constraint: {sense}"
            )


def from_docplex_mp(model: Model, indicator_big_m: Optional[float] = None) -> QuadraticProgram:
    """Translate a docplex.mp model into a quadratic program.

    Note that this supports only basic functions of docplex as follows:
    - quadratic objective function
    - linear / quadratic / indicator constraints
    - binary / integer / continuous variables

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

    # get name
    quadratic_program = QuadraticProgram(model.name)

    # get variables
    # keep track of names separately, since docplex allows to have None names.
    var_names = {}
    var_bounds = {}
    for x in model.iter_variables():
        if isinstance(x.vartype, ContinuousVarType):
            x_new = quadratic_program.continuous_var(x.lb, x.ub, x.name)
        elif isinstance(x.vartype, BinaryVarType):
            x_new = quadratic_program.binary_var(x.name)
        elif isinstance(x.vartype, IntegerVarType):
            x_new = quadratic_program.integer_var(x.lb, x.ub, x.name)
        else:
            raise QiskitOptimizationError(f"Unsupported variable type: {x.name} {x.vartype}")
        var_names[x] = x_new.name
        var_bounds[x.name] = (x_new.lowerbound, x_new.upperbound)

    # objective sense
    minimize = model.objective_sense.is_minimize()

    # make sure objective expression is linear or quadratic and not a variable
    if isinstance(model.objective_expr, Var):
        model.objective_expr = model.objective_expr + 0

    # get objective offset
    constant = model.objective_expr.constant

    # get linear part of objective
    linear = {}
    linear_part = model.objective_expr.get_linear_part()
    for x in linear_part.iter_variables():
        linear[var_names[x]] = linear_part.get_coef(x)

    # get quadratic part of objective
    quadratic = {}
    if isinstance(model.objective_expr, QuadExpr):
        for quad_triplet in model.objective_expr.iter_quad_triplets():
            i = var_names[quad_triplet[0]]
            j = var_names[quad_triplet[1]]
            v = quad_triplet[2]
            quadratic[i, j] = v

    # set objective
    if minimize:
        quadratic_program.minimize(constant, linear, quadratic)
    else:
        quadratic_program.maximize(constant, linear, quadratic)

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

    # get linear constraints
    for constraint in model.iter_linear_constraints():
        lhs, sense, rhs = _FromDocplexMp._linear_constraint(var_names, constraint)
        quadratic_program.linear_constraint(lhs, sense, rhs, constraint.name)

    # get quadratic constraints
    for constraint in model.iter_quadratic_constraints():
        linear, quadratic, sense, rhs = _FromDocplexMp._quadratic_constraint(var_names, constraint)
        quadratic_program.quadratic_constraint(linear, quadratic, sense, rhs, constraint.name)

    # get indicator constraints
    for constraint in model.iter_indicator_constraints():
        linear_constraints = _FromDocplexMp._indicator_constraints(
            var_names, var_bounds, constraint, indicator_big_m
        )
        for linear, sense, rhs, name in linear_constraints:
            quadratic_program.linear_constraint(linear, sense, rhs, name)

    return quadratic_program
