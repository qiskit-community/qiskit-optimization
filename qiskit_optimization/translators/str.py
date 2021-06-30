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

"""Translate QuadraticProgram into string"""

from io import StringIO
from math import isclose
from typing import List, Optional, Union, cast

from qiskit_optimization import INFINITY
from qiskit_optimization.problems import (
    LinearExpression,
    QuadraticExpression,
    QuadraticObjective,
    QuadraticProgram,
)
from qiskit_optimization.problems.constraint import ConstraintSense

SENSE = {ConstraintSense.EQ: "==", ConstraintSense.LE: "<=", ConstraintSense.GE: ">="}


def _f2i(val: Union[int, float]) -> Union[int, float]:
    """Convert a value into an integer if possible

    Note: if abs(val) >= 1e23, int(val) is not correct
          e.g., int(1e23) -> 99999999999999991611392
    """
    if isinstance(val, float) and abs(val) <= 1e20 and cast(float, val).is_integer():
        return int(val)
    return val


def _coeff2str(coeff: float, is_head: bool) -> List[str]:
    """Translate a coefficient to a list of strings.

    Args:
        coeff: The coefficient.
        is_head: Whether this coefficient appears in the head of the string or not.

    Returns:
        A list of strings consisting of the sign and the absolute value of the coefficient.
    """
    if coeff < 0.0:
        sign = "-"
    else:
        sign = "" if is_head else "+"
    ret = []
    if sign:
        ret.append(sign)
    abs_val = abs(coeff)
    if not isclose(abs_val, 1.0):
        ret.append(f"{_f2i(abs_val)}")
    return ret


def _expr2str(
    constant: float = 0.0,
    lin: Optional[LinearExpression] = None,
    quad: Optional[QuadraticExpression] = None,
) -> str:
    """Translate the sum of expressions into a string.

    Args:
        constant: The constant part.
        lin: The linear expression.
        quad: The quadratic expression.

    Returns:
        A string representing the sum of the expressions.
    """
    expr = []
    is_head = True
    lin_dict = lin.to_dict(use_name=True) if lin else {}
    quad_dict = quad.to_dict(use_name=True) if quad else {}

    # constant
    if not isclose(constant, 0.0, abs_tol=1e-10):
        expr.append(f"{_f2i(constant)}")
        is_head = False
    elif not lin_dict and not quad_dict:
        expr.append("0")
        is_head = False

    # quadratic expression
    for (var1, var2), coeff in quad_dict.items():
        expr.extend(_coeff2str(coeff, is_head))
        is_head = False
        if var1 == var2:
            expr.append(f"{var1}^2")
        else:
            expr.append(f"{var1} * {var2}")

    # linear expression
    for var, coeff in lin_dict.items():
        expr.extend(_coeff2str(coeff, is_head))
        expr.append(f"{var}")
        is_head = False

    return " ".join(expr)


def to_str(quadratic_program: QuadraticProgram) -> str:
    """Translate QuadraticProgram into a string

    Args:
        quadratic_program: The optimization problem to be translated into a string

    Returns:
        A string representing this problem.
    """

    with StringIO() as buf:
        buf.write(f"Problem name: {quadratic_program.name}\n\n")
        if quadratic_program.objective.sense == QuadraticObjective.Sense.MINIMIZE:
            buf.write("Minimize\n")
        else:
            buf.write("Maximize\n")
        buf.write("  ")
        buf.write(
            _expr2str(
                quadratic_program.objective.constant,
                quadratic_program.objective.linear,
                quadratic_program.objective.quadratic,
            )
        )
        buf.write("\n\nSubject to\n")
        if (
            quadratic_program.get_num_linear_constraints() == 0
            and quadratic_program.get_num_quadratic_constraints() == 0
        ):
            buf.write("  No constraints\n")
        for cst in quadratic_program.linear_constraints:
            buf.write(f"  {cst.name}: ")
            buf.write(_expr2str(lin=cst.linear))
            buf.write(f" {SENSE[cst.sense]} {_f2i(cst.rhs)}")
            buf.write("\n")
        for cst2 in quadratic_program.quadratic_constraints:
            buf.write(f"  {cst2.name}: ")
            buf.write(_expr2str(lin=cst2.linear, quad=cst2.quadratic))
            buf.write(f" {SENSE[cst2.sense]} {_f2i(cst2.rhs)}")
            buf.write("\n")
        buf.write("\nVariables\n")
        if quadratic_program.get_num_vars() == 0:
            buf.write("  No variables\n")
        for var in quadratic_program.variables:
            if var.lowerbound > -INFINITY:
                buf.write(f"  {_f2i(var.lowerbound)} <= ")
            buf.write(var.name)
            if var.upperbound < INFINITY:
                buf.write(f" <= {_f2i(var.upperbound)}")
            buf.write(f": {var.vartype.name.lower()}\n")
        ret = buf.getvalue()
    return ret
