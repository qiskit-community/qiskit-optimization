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

"""Model translator between QuadraticProgram and Docplex"""

from docplex.mp.constr import LinearConstraint as DocplexLinearConstraint
from docplex.mp.constr import NotEqualConstraint
from docplex.mp.constr import QuadraticConstraint as DocplexQuadraticConstraint

from docplex.mp.model import Model
from docplex.mp.quad import QuadExpr
from docplex.mp.vartype import BinaryVarType, ContinuousVarType, IntegerVarType

try:
    # new location since docplex 2.16.196
    from docplex.mp.dvar import Var
except ImportError:
    # old location until docplex 2.15.194
    from docplex.mp.linear import Var

from qiskit_optimization import QiskitOptimizationError
from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from qiskit_optimization.problems.variable import Variable
from qiskit_optimization.problems.constraint import Constraint
from qiskit_optimization.problems.quadratic_objective import QuadraticObjective
from .model_translator import ModelTranslator

from typing import cast


class DocplexTranslator(ModelTranslator[Model]):
    def qp_to_model(self, prog: QuadraticProgram) -> Model:
        """Returns a docplex model corresponding to this quadratic program.

        Returns:
            The docplex model corresponding to this quadratic program.

        Raises:
            QiskitOptimizationError: if non-supported elements (should never happen).
        """

        # initialize model
        mdl = Model(prog.name)

        # add variables
        var = {}
        for idx, x in enumerate(prog.variables):
            if x.vartype == Variable.Type.CONTINUOUS:
                var[idx] = mdl.continuous_var(lb=x.lowerbound, ub=x.upperbound, name=x.name)
            elif x.vartype == Variable.Type.BINARY:
                var[idx] = mdl.binary_var(name=x.name)
            elif x.vartype == Variable.Type.INTEGER:
                var[idx] = mdl.integer_var(lb=x.lowerbound, ub=x.upperbound, name=x.name)
            else:
                # should never happen
                raise QiskitOptimizationError('Unsupported variable type: {}'.format(x.vartype))

        # add objective
        objective = prog.objective.constant
        for i, v in prog.objective.linear.to_dict().items():
            objective += v * var[cast(int, i)]
        for (i, j), v in prog.objective.quadratic.to_dict().items():
            objective += v * var[cast(int, i)] * var[cast(int, j)]
        if prog.objective.sense == QuadraticObjective.Sense.MINIMIZE:
            mdl.minimize(objective)
        else:
            mdl.maximize(objective)

        # add linear constraints
        for i, l_constraint in enumerate(prog.linear_constraints):
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
                raise QiskitOptimizationError("Unsupported constraint sense: {}".format(sense))

        # add quadratic constraints
        for i, q_constraint in enumerate(prog.quadratic_constraints):
            name = q_constraint.name
            rhs = q_constraint.rhs
            if rhs == 0 \
                    and q_constraint.linear.coefficients.nnz == 0 \
                    and q_constraint.quadratic.coefficients.nnz == 0:
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
                raise QiskitOptimizationError("Unsupported constraint sense: {}".format(sense))
        return mdl

    def model_to_qp(self, model: Model) -> QuadraticProgram:
        """Loads this quadratic program from a docplex model.

        Note that this supports only basic functions of docplex as follows:
        - quadratic objective function
        - linear / quadratic constraints
        - binary / integer / continuous variables

        Args:
            model: The docplex model to be loaded.

        Raises:
            QiskitOptimizationError: if the model contains unsupported elements.
        """

        # clear current problem
        prog = QuadraticProgram()

        # get name
        prog.name = model.name

        # get variables
        # keep track of names separately, since docplex allows to have None names.
        var_names = {}
        for x in model.iter_variables():
            if isinstance(x.vartype, ContinuousVarType):
                x_new = prog.continuous_var(x.lb, x.ub, x.name)
            elif isinstance(x.vartype, BinaryVarType):
                x_new = prog.binary_var(x.name)
            elif isinstance(x.vartype, IntegerVarType):
                x_new = prog.integer_var(x.lb, x.ub, x.name)
            else:
                raise QiskitOptimizationError(
                    "Unsupported variable type: {} {}".format(x.name, x.vartype))
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
            prog.minimize(constant, linear, quadratic)
        else:
            prog.maximize(constant, linear, quadratic)

        # get linear constraints
        for constraint in model.iter_constraints():
            if isinstance(constraint, DocplexQuadraticConstraint):
                # ignore quadratic constraints here and process them later
                continue
            if not isinstance(constraint, DocplexLinearConstraint) or \
                    isinstance(constraint, NotEqualConstraint):
                # If any constraint is not linear/quadratic constraints, it raises an error.
                # Notice that NotEqualConstraint is a subclass of Docplex's LinearConstraint,
                # but it cannot be handled by optimization.
                raise QiskitOptimizationError(
                    'Unsupported constraint: {}'.format(constraint))
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
                prog.linear_constraint(lhs, '==', rhs, name)
            elif sense == sense.GE:
                prog.linear_constraint(lhs, '>=', rhs, name)
            elif sense == sense.LE:
                prog.linear_constraint(lhs, '<=', rhs, name)
            else:
                raise QiskitOptimizationError(
                    "Unsupported constraint sense: {}".format(constraint))

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
                    linear[var_names[x]] = linear.get(var_names[x], 0.0) - \
                                           right_expr.linear_part.get_coef(x)
                for quad_triplet in right_expr.iter_quad_triplets():
                    i = var_names[quad_triplet[0]]
                    j = var_names[quad_triplet[1]]
                    v = quad_triplet[2]
                    quadratic[i, j] = quadratic.get((i, j), 0.0) - v
            else:
                for x in right_expr.iter_variables():
                    linear[var_names[x]] = linear.get(var_names[x], 0.0) - right_expr.get_coef(
                        x)

            if sense == sense.EQ:
                prog.quadratic_constraint(linear, quadratic, '==', rhs, name)
            elif sense == sense.GE:
                prog.quadratic_constraint(linear, quadratic, '>=', rhs, name)
            elif sense == sense.LE:
                prog.quadratic_constraint(linear, quadratic, '<=', rhs, name)
            else:
                raise QiskitOptimizationError(
                    "Unsupported constraint sense: {}".format(constraint))
        return prog
