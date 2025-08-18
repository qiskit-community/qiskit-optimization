# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Translate ``QuadraticProgram`` into a pretty-printed string"""
from __future__ import annotations

from io import StringIO
from math import isclose
from typing import cast

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


def _int_if_close(val: int | float | np.integer | np.floating) -> int | float:
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
                ret = f"{sign} {term}"
            else:
                ret = f"{sign} {_int_if_close(abs_val)}*{term}"
    else:
        if is_head:
            ret = f"{_int_if_close(coeff)}"
        else:
            sign = "-" if coeff < 0.0 else "+"
            abs_val = abs(coeff)
            ret = f"{sign} {_int_if_close(abs_val)}"
    return ret


def _check_name(name: str, name_type: str) -> None:
    """Check a name is printable or not.

    Args:
        name: a variable name.
        name_type: the type associated with the name.

    Raises:
        QiskitOptimizationError: if the name is not printable.
    """
    if not name.isprintable():
        raise QiskitOptimizationError(f"{name_type} name is not printable: {repr(name)}")


def _concatenate_terms(terms: list[str], wrap: int, indent: int) -> str:
    ind = " " * indent
    if wrap == 0:
        return ind + " ".join(terms)
    buf = ind
    cur = indent
    for term in terms:
        if cur + len(term) >= wrap:
            buf += "\n"
            buf += ind
            cur = indent
        if cur != indent:  # if the position is not the start of the line
            buf += " "
            cur += 1
        buf += term
        cur += len(term)
    return buf


def expr2str(  # pylint: disable=too-many-positional-arguments
    constant: float = 0.0,
    linear: LinearExpression | None = None,
    quadratic: QuadraticExpression | None = None,
    truncate: int = 0,
    suffix: str = "",
    wrap: int = 0,
    indent: int = 0,
) -> str:
    """Translate a combination of a constant, a linear expression, and a quadratic expression
    into a string.

    Args:
        constant: a constant part.
        linear: a linear expression.
        quadratic: a quadratic expression.
        truncate: the threshold of the output string to be truncated. If a string is longer than
            the threshold, it is truncated and appended "...", e.g., "x^2 + y +...".
            It is disabled by setting 0. The default value is 0.
        suffix: a suffix text.
        wrap: The text width to wrap the output strings. It is disabled by setting 0.
            Note that some strings might exceed this value, for example, a long variable
            name won't be wrapped. The default value is 0.
        indent: The indent size. The default value is 0.

    Returns:
        A string representing the combination of the expressions.

    Raises:
        ValueError: if ``truncate`` is negative.
        QiskitOptimizationError: if the variable name is not printable.
    """
    if truncate < 0:
        raise ValueError(f"Invalid truncate value: {truncate}")

    terms = []
    is_head = True
    lin_dict = linear.to_dict(use_name=True) if linear else {}
    quad_dict = quadratic.to_dict(use_name=True) if quadratic else {}

    # quadratic expression
    for (var1, var2), coeff in sorted(quad_dict.items()):
        _check_name(cast(str, var1), "Variable")
        _check_name(cast(str, var2), "Variable")
        if var1 == var2:
            terms.append(_term2str(coeff, f"{var1}^2", is_head))
        else:
            terms.append(_term2str(coeff, f"{var1}*{var2}", is_head))
        is_head = False

    # linear expression
    for var, coeff in sorted(lin_dict.items()):
        _check_name(cast(str, var), "Variable")
        terms.append(_term2str(coeff, f"{var}", is_head))
        is_head = False

    # constant
    if not isclose(constant, 0.0, abs_tol=1e-10):
        terms.append(_term2str(constant, "", is_head))
    elif not lin_dict and not quad_dict:
        terms.append(_term2str(0, "", is_head))

    # suffix
    if suffix:
        terms.append(suffix)

    ret = _concatenate_terms(terms, wrap, indent)
    if 0 < truncate < len(ret):
        ret = ret[:truncate] + "..."
    return ret


def prettyprint(quadratic_program: QuadraticProgram, wrap: int = 80) -> str:
    """Translate a :class:`~qiskit_optimization.problems.QuadraticProgram` into a pretty-printed string.

    Args:
        quadratic_program: The optimization problem to be translated into a string.
        wrap: The text width to wrap the output strings. It is disabled by setting 0.
            Note that some strings might exceed this value, for example, a long variable
            name won't be wrapped. The default value is 80.

    Returns:
        A pretty-printed string representing the problem.

    Raises:
        QiskitOptimizationError: if there is a non-printable name.
    """

    with StringIO() as buf:
        _check_name(quadratic_program.name, "Problem")
        buf.write(f"Problem name: {quadratic_program.name}\n\n")
        if quadratic_program.objective.sense == QuadraticObjective.Sense.MINIMIZE:
            buf.write("Minimize\n")
        else:
            buf.write("Maximize\n")
        buf.write(
            expr2str(
                quadratic_program.objective.constant,
                quadratic_program.objective.linear,
                quadratic_program.objective.quadratic,
                wrap=wrap,
                indent=2,
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
                _check_name(cst.name, "Linear constraint")
                suffix = f"{cst.sense.label} {_int_if_close(cst.rhs)}  '{cst.name}'\n"
                buf.write(expr2str(linear=cst.linear, suffix=suffix, wrap=wrap, indent=4))
        if num_quad_csts > 0:
            buf.write(f"\n  Quadratic constraints ({num_quad_csts})\n")
            for cst2 in quadratic_program.quadratic_constraints:
                _check_name(cst2.name, "Quadratic constraint")
                suffix = f"{cst2.sense.label} {_int_if_close(cst2.rhs)}  '{cst2.name}'\n"
                buf.write(
                    expr2str(
                        linear=cst2.linear,
                        quadratic=cst2.quadratic,
                        suffix=suffix,
                        wrap=wrap,
                        indent=4,
                    )
                )
        if quadratic_program.get_num_vars() == 0:
            buf.write("\n  No variables\n")
        bin_vars = []
        int_vars = []
        con_vars = []
        for var in quadratic_program.variables:
            if var.vartype is VarType.BINARY:
                _check_name(var.name, "Variable")
                bin_vars.append(var.name)
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
                _check_name(var.name, "Variable")
                buf.write(var.name)
                if var.upperbound < INFINITY:
                    buf.write(f" <= {_int_if_close(var.upperbound)}")
                buf.write("\n")
        if con_vars:
            buf.write(f"\n  Continuous variables ({len(con_vars)})\n")
            for var in con_vars:
                buf.write("    ")
                if var.lowerbound > -INFINITY:
                    buf.write(f"{_int_if_close(var.lowerbound)} <= ")
                _check_name(var.name, "Variable")
                buf.write(var.name)
                if var.upperbound < INFINITY:
                    buf.write(f" <= {_int_if_close(var.upperbound)}")
                buf.write("\n")
        if bin_vars:
            buf.write(f"\n  Binary variables ({len(bin_vars)})\n")
            buf.write(_concatenate_terms(bin_vars, wrap=wrap, indent=4))
            buf.write("\n")
        ret = buf.getvalue()
    return ret
