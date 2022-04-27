# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Translate ``QuadraticProgram`` into a pretty-printed string"""

from io import StringIO
from math import isclose
from typing import Optional, Union, cast

import numpy as np

from qiskit_optimization import INFINITY, QiskitOptimizationError
from qiskit_optimization.problems import (
    LinearExpression,
    QuadraticExpression,
    QuadraticObjective,
    QuadraticProgram,
    VarType,
)

DEFAULT_TRUNCATE = 50


def _int_if_close(val: Union[int, float, np.integer, np.floating]) -> Union[int, float]:
    """Convert a value into an integer if possible

    Note: if abs(val) is too large, int(val) is not correct
          e.g., int(1e16 - 1) -> 10000000000000000
    """
    if isinstance(val, np.integer):
        val = int(val)
    elif isinstance(val, np.floating):
        val = float(val)

    if isinstance(val, int):
        return val
    if abs(val) <= 1e10 and val.is_integer():
        return int(val)
    return val


def _term2str(coeff: float, term: str, is_head: bool) -> str:
    """Translate a pair of a coefficient and a term to a string.

    Args:
        coeff: a coefficient.
        term: a term. This can be empty and `coeff` is treated as a constant.
        is_head: Whether this coefficient appears in the head of the string or not.

    Returns:
        A strings representing the term.
    """
    if term:
        if is_head:
            if isclose(coeff, 1.0):
                ret = term
            elif isclose(coeff, -1.0):
                ret = f"-{term}"
            else:
                ret = f"{_int_if_close(coeff)}*{term}"
        else:
            sign = "-" if coeff < 0.0 else "+"
            abs_val = abs(coeff)
            if isclose(abs_val, 1.0):
                ret = f" {sign} {term}"
            else:
                ret = f" {sign} {_int_if_close(abs_val)}*{term}"
    else:
        if is_head:
            ret = f"{_int_if_close(coeff)}"
        else:
            sign = "-" if coeff < 0.0 else "+"
            abs_val = abs(coeff)
            ret = f" {sign} {_int_if_close(abs_val)}"
    return ret


def _check_name(name: str) -> None:
    """Check a name is printable.

    Args:
        name: a variable name.

    Raises:
        QiskitOptimizationError: if the variable name is not printable.
    """
    if not name.isprintable():
        raise QiskitOptimizationError("Variable name is not printable")


def _varname(name: str) -> str:
    """Translate a variable name into a string.

    Args:
        name: a variable name.

    Returns:
        A string representing the variable name. If it contains " ", "+", "-" or, "*", the name
        is translated into ("{name}").

    Raises:
        QiskitOptimizationError: if the variable name is not printable.
    """
    _check_name(name)
    if {" ", "+", "-", "*"}.intersection(set(name)):
        return f'("{name}")'
    return name


def _expr2str(
    constant: float = 0.0,
    linear: Optional[LinearExpression] = None,
    quadratic: Optional[QuadraticExpression] = None,
    truncate: int = 0,
) -> str:
    """Translate a combination of a constant, a linear expression, and a quadratic expression
    into a string.

    Args:
        constant: a constant part.
        linear: a linear expression.
        quadratic: a quadratic expression.
        truncate: the threshold of the output string to be truncated. If a string is longer than
            the threshold, it is truncated and appended "...", e.g., "x^2 + y +...".
            The default value 0 means no truncation is carried out.

    Returns:
        A string representing the combination of the expressions.

    Raises:
        ValueError: if `truncate` is negative.
        QiskitOptimizationError: if the variable name is not printable.
    """
    if truncate < 0:
        raise ValueError(f"Invalid truncate value: {truncate}")

    expr = StringIO()
    is_head = True
    lin_dict = linear.to_dict(use_name=True) if linear else {}
    quad_dict = quadratic.to_dict(use_name=True) if quadratic else {}

    # quadratic expression
    for (var1, var2), coeff in sorted(quad_dict.items()):
        var1 = _varname(cast(str, var1))
        var2 = _varname(cast(str, var2))
        if var1 == var2:
            expr.write(_term2str(coeff, f"{var1}^2", is_head))
        else:
            expr.write(_term2str(coeff, f"{var1}*{var2}", is_head))
        is_head = False

    # linear expression
    for var, coeff in sorted(lin_dict.items()):
        var = _varname(cast(str, var))
        expr.write(_term2str(coeff, f"{var}", is_head))
        is_head = False

    # constant
    if not isclose(constant, 0.0, abs_tol=1e-10):
        expr.write(_term2str(constant, "", is_head))
    elif not lin_dict and not quad_dict:
        expr.write(_term2str(0, "", is_head))

    ret = expr.getvalue()
    if 0 < truncate < len(ret):
        ret = ret[:truncate] + "..."
    return ret


def prettyprint(quadratic_program: QuadraticProgram) -> str:
    """Translate a :class:`~qiskit_optimization.problem.QuadraticProgram` into a pretty-printed string.

    Args:
        quadratic_program: The optimization problem to be translated into a string

    Returns:
        A pretty-printed string representing the problem.

    Raises:
        QiskitOptimizationError: if the variable name is not printable.
    """

    with StringIO() as buf:
        _check_name(quadratic_program.name)
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
        buf.write("\n\nSubject to")
        num_lin_csts = quadratic_program.get_num_linear_constraints()
        num_quad_csts = quadratic_program.get_num_quadratic_constraints()
        if num_lin_csts == 0 and num_quad_csts == 0:
            buf.write("\n  No constraints\n")
        if num_lin_csts > 0:
            buf.write(f"\n  Linear constraints ({num_lin_csts})\n")
            for cst in quadratic_program.linear_constraints:
                _check_name(cst.name)
                buf.write(
                    f"    {_expr2str(linear=cst.linear)}"
                    f" {cst.sense.label} {_int_if_close(cst.rhs)}"
                    f"  '{cst.name}'\n"
                )
        if num_quad_csts > 0:
            buf.write(f"\n  Quadratic constraints ({num_quad_csts})\n")
            for cst2 in quadratic_program.quadratic_constraints:
                _check_name(cst2.name)
                buf.write(
                    f"    {_expr2str(linear=cst2.linear, quadratic=cst2.quadratic)}"
                    f" {cst2.sense.label} {_int_if_close(cst2.rhs)}"
                    f"  '{cst2.name}'\n"
                )
        if quadratic_program.get_num_vars() == 0:
            buf.write("\n  No variables\n")
        bin_vars = []
        int_vars = []
        con_vars = []
        for var in quadratic_program.variables:
            if var.vartype is VarType.BINARY:
                bin_vars.append(_varname(var.name))
            elif var.vartype is VarType.INTEGER:
                int_vars.append(var)
            else:
                con_vars.append(var)
        if int_vars:
            buf.write(f"\n  Integer variables ({len(int_vars)})\n")
            for var in int_vars:
                buf.write("    ")
                if var.lowerbound > -INFINITY:
                    buf.write(f"{_int_if_close(var.lowerbound)} <= ")
                buf.write(_varname(var.name))
                if var.upperbound < INFINITY:
                    buf.write(f" <= {_int_if_close(var.upperbound)}")
                buf.write("\n")
        if con_vars:
            buf.write(f"\n  Continuous variables ({len(con_vars)})\n")
            for var in con_vars:
                buf.write("    ")
                if var.lowerbound > -INFINITY:
                    buf.write(f"{_int_if_close(var.lowerbound)} <= ")
                buf.write(_varname(var.name))
                if var.upperbound < INFINITY:
                    buf.write(f" <= {_int_if_close(var.upperbound)}")
                buf.write("\n")
        if bin_vars:
            buf.write(f"\n  Binary variables ({len(bin_vars)})\n")
            buf.write(f"    {' '.join(bin_vars)}\n")
        ret = buf.getvalue()
    return ret
