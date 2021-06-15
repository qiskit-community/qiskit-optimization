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

from typing import TYPE_CHECKING, cast

from docplex.mp.constr import LinearConstraint as DocplexLinearConstraint
from docplex.mp.constr import NotEqualConstraint
from docplex.mp.constr import QuadraticConstraint as DocplexQuadraticConstraint
from docplex.mp.dvar import Var
from docplex.mp.model import Model
from docplex.mp.quad import QuadExpr
from docplex.mp.vartype import BinaryVarType, ContinuousVarType, IntegerVarType

from qiskit_optimization.exceptions import QiskitOptimizationError
from qiskit_optimization.problems.constraint import Constraint
from qiskit_optimization.problems.quadratic_objective import QuadraticObjective
from qiskit_optimization.problems.variable import Variable

if TYPE_CHECKING:
    # pylint: disable=cyclic-import
    from qiskit_optimization.problems.quadratic_program import QuadraticProgram


def to_docplex_mp(quadratic_program: "QuadraticProgram") -> Model:
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
            raise QiskitOptimizationError(
                "Internal error: unsupported variable type: {}".format(x.vartype)
            )

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
            raise QiskitOptimizationError(
                "Internal error: unsupported constraint sense: {}".format(sense)
            )

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
            raise QiskitOptimizationError(
                "Internal error: unsupported constraint sense: {}".format(sense)
            )

    return mdl


def from_docplex_mp(model: Model) -> "QuadraticProgram":
    """Translate a docplex.mp model into a quadratic program.

    Note that this supports only basic functions of docplex as follows:
    - quadratic objective function
    - linear / quadratic constraints
    - binary / integer / continuous variables

    Args:
        model: The docplex.mp model to be loaded.

    Returns:
        The quadratic program corresponding to the model.

    Raises:
        QiskitOptimizationError: if the model contains unsupported elements.
    """
    if not isinstance(model, Model):
        raise QiskitOptimizationError(f"The model is not compatible: {model}")

    # pylint: disable=cyclic-import
    from qiskit_optimization.problems.quadratic_program import QuadraticProgram

    quadratic_program = QuadraticProgram()

    # get name
    quadratic_program.name = model.name

    # get variables
    # keep track of names separately, since docplex allows to have None names.
    var_names = {}
    for x in model.iter_variables():
        if isinstance(x.vartype, ContinuousVarType):
            x_new = quadratic_program.continuous_var(x.lb, x.ub, x.name)
        elif isinstance(x.vartype, BinaryVarType):
            x_new = quadratic_program.binary_var(x.name)
        elif isinstance(x.vartype, IntegerVarType):
            x_new = quadratic_program.integer_var(x.lb, x.ub, x.name)
        else:
            raise QiskitOptimizationError(
                "Unsupported variable type: {} {}".format(x.name, x.vartype)
            )
        var_names[x] = x_new.name

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

    # get linear constraints
    for constraint in model.iter_constraints():
        if isinstance(constraint, DocplexQuadraticConstraint):
            # ignore quadratic constraints here and process them later
            continue
        if not isinstance(constraint, DocplexLinearConstraint) or isinstance(
            constraint, NotEqualConstraint
        ):
            # If any constraint is not linear/quadratic constraints, it raises an error.
            # Notice that NotEqualConstraint is a subclass of Docplex's LinearConstraint,
            # but it cannot be handled by optimization.
            raise QiskitOptimizationError("Unsupported constraint: {}".format(constraint))
        name = constraint.name
        sense = constraint.sense

        left_expr = constraint.get_left_expr()
        right_expr = constraint.get_right_expr()
        # for linear constraints we may get an instance of Var instead of expression,
        # e.g. x + y = z
        if isinstance(left_expr, Var):
            left_expr = left_expr + 0
        if isinstance(right_expr, Var):
            right_expr = right_expr + 0

        rhs = right_expr.constant - left_expr.constant

        lhs = {}
        for x in left_expr.iter_variables():
            lhs[var_names[x]] = left_expr.get_coef(x)
        for x in right_expr.iter_variables():
            lhs[var_names[x]] = lhs.get(var_names[x], 0.0) - right_expr.get_coef(x)

        if sense == sense.EQ:
            quadratic_program.linear_constraint(lhs, "==", rhs, name)
        elif sense == sense.GE:
            quadratic_program.linear_constraint(lhs, ">=", rhs, name)
        elif sense == sense.LE:
            quadratic_program.linear_constraint(lhs, "<=", rhs, name)
        else:
            raise QiskitOptimizationError("Unsupported constraint sense: {}".format(constraint))

    # get quadratic constraints
    for constraint in model.iter_quadratic_constraints():
        name = constraint.name
        sense = constraint.sense

        left_expr = constraint.get_left_expr()
        right_expr = constraint.get_right_expr()

        rhs = right_expr.constant - left_expr.constant
        linear = {}
        quadratic = {}

        if left_expr.is_quad_expr():
            for x in left_expr.linear_part.iter_variables():
                linear[var_names[x]] = left_expr.linear_part.get_coef(x)
            for quad_triplet in left_expr.iter_quad_triplets():
                i = var_names[quad_triplet[0]]
                j = var_names[quad_triplet[1]]
                v = quad_triplet[2]
                quadratic[i, j] = v
        else:
            for x in left_expr.iter_variables():
                linear[var_names[x]] = left_expr.get_coef(x)

        if right_expr.is_quad_expr():
            for x in right_expr.linear_part.iter_variables():
                linear[var_names[x]] = linear.get(
                    var_names[x], 0.0
                ) - right_expr.linear_part.get_coef(x)
            for quad_triplet in right_expr.iter_quad_triplets():
                i = var_names[quad_triplet[0]]
                j = var_names[quad_triplet[1]]
                v = quad_triplet[2]
                quadratic[i, j] = quadratic.get((i, j), 0.0) - v
        else:
            for x in right_expr.iter_variables():
                linear[var_names[x]] = linear.get(var_names[x], 0.0) - right_expr.get_coef(x)

        if sense == sense.EQ:
            quadratic_program.quadratic_constraint(linear, quadratic, "==", rhs, name)
        elif sense == sense.GE:
            quadratic_program.quadratic_constraint(linear, quadratic, ">=", rhs, name)
        elif sense == sense.LE:
            quadratic_program.quadratic_constraint(linear, quadratic, "<=", rhs, name)
        else:
            raise QiskitOptimizationError("Unsupported constraint sense: {}".format(constraint))

    return quadratic_program
