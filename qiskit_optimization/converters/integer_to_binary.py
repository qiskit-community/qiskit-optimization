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

"""The converter to map integer variables in a quadratic program to binary variables."""

import copy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .quadratic_program_converter import QuadraticProgramConverter
from ..exceptions import QiskitOptimizationError
from ..problems.quadratic_objective import QuadraticObjective
from ..problems.quadratic_program import QuadraticProgram
from ..problems.variable import Variable


class IntegerToBinary(QuadraticProgramConverter):
    """Convert a :class:`~qiskit_optimization.problems.QuadraticProgram` into new one by encoding
    integer with binary variables.

    This bounded-coefficient encoding used in this converted is proposed in [1], Eq. (5).

    Examples:
        >>> from qiskit_optimization.problems import QuadraticProgram
        >>> from qiskit_optimization.converters import IntegerToBinary
        >>> problem = QuadraticProgram()
        >>> var = problem.integer_var(name='x', lowerbound=0, upperbound=10)
        >>> conv = IntegerToBinary()
        >>> problem2 = conv.convert(problem)

    References:
        [1]: Sahar Karimi, Pooya Ronagh (2017), Practical Integer-to-Binary Mapping for Quantum
            Annealers. arxiv.org:1706.01945.
    """

    _delimiter = "@"  # users are supposed not to use this character in variable names

    def __init__(self) -> None:
        self._src: Optional[QuadraticProgram] = None
        self._dst: Optional[QuadraticProgram] = None
        self._conv: Dict[Variable, List[Tuple[str, int]]] = {}
        # e.g., self._conv = {x: [('x@1', 1), ('x@2', 2)]}

    def convert(self, problem: QuadraticProgram) -> QuadraticProgram:
        """Convert an integer problem into a new problem with binary variables.

        Args:
            problem: The problem to be solved, that may contain integer variables.

        Returns:
            The converted problem, that contains no integer variables.

        Raises:
            QiskitOptimizationError: if variable or constraint type is not supported.
        """

        # Copy original QP as reference.
        self._src = copy.deepcopy(problem)

        if self._src.get_num_integer_vars() > 0:

            # Initialize new QP
            self._dst = QuadraticProgram(name=problem.name)

            # Declare variables
            for x in self._src.variables:
                if x.vartype == Variable.Type.INTEGER:
                    new_vars = self._convert_var(x.name, x.lowerbound, x.upperbound)
                    self._conv[x] = new_vars
                    for (var_name, _) in new_vars:
                        self._dst.binary_var(var_name)
                else:
                    if x.vartype == Variable.Type.CONTINUOUS:
                        self._dst.continuous_var(x.lowerbound, x.upperbound, x.name)
                    elif x.vartype == Variable.Type.BINARY:
                        self._dst.binary_var(x.name)
                    else:
                        raise QiskitOptimizationError(f"Unsupported variable type {x.vartype}")

            self._substitute_int_var()

        else:
            # just copy the problem if no integer variables exist
            self._dst = copy.deepcopy(problem)

        return self._dst

    def _convert_var(
        self, name: str, lowerbound: float, upperbound: float
    ) -> List[Tuple[str, int]]:
        var_range = upperbound - lowerbound
        power = int(np.log2(var_range)) if var_range > 0 else 0
        bounded_coef = var_range - (2 ** power - 1)

        coeffs = [2 ** i for i in range(power)] + [bounded_coef]
        return [(name + self._delimiter + str(i), coef) for i, coef in enumerate(coeffs)]

    def _convert_linear_coefficients_dict(
        self, coefficients: Dict[str, float]
    ) -> Tuple[Dict[str, float], float]:
        constant = 0.0
        linear: Dict[str, float] = {}
        for name, v in coefficients.items():
            x = self._src.get_variable(name)
            if x in self._conv:
                for y, coeff in self._conv[x]:
                    linear[y] = v * coeff
                constant += v * x.lowerbound
            else:
                linear[x.name] = v

        return linear, constant

    def _convert_quadratic_coefficients_dict(
        self, coefficients: Dict[Tuple[str, str], float]
    ) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float], float]:
        constant = 0.0
        linear: Dict[str, float] = {}
        quadratic = {}
        for (name_i, name_j), v in coefficients.items():
            x = self._src.get_variable(name_i)
            y = self._src.get_variable(name_j)

            if x in self._conv and y not in self._conv:
                for z_x, coeff_x in self._conv[x]:
                    quadratic[z_x, y.name] = v * coeff_x
                linear[y.name] = linear.get(y.name, 0.0) + v * x.lowerbound

            elif x not in self._conv and y in self._conv:
                for z_y, coeff_y in self._conv[y]:
                    quadratic[x.name, z_y] = v * coeff_y
                linear[x.name] = linear.get(x.name, 0.0) + v * y.lowerbound

            elif x in self._conv and y in self._conv:
                for z_x, coeff_x in self._conv[x]:
                    for z_y, coeff_y in self._conv[y]:
                        quadratic[z_x, z_y] = v * coeff_x * coeff_y

                for z_x, coeff_x in self._conv[x]:
                    linear[z_x] = linear.get(z_x, 0.0) + v * coeff_x * y.lowerbound
                for z_y, coeff_y in self._conv[y]:
                    linear[z_y] = linear.get(z_y, 0.0) + v * coeff_y * x.lowerbound

                constant += v * x.lowerbound * y.lowerbound

            else:
                quadratic[x.name, y.name] = v

        return quadratic, linear, constant

    def _substitute_int_var(self):

        # set objective
        linear, linear_constant = self._convert_linear_coefficients_dict(
            self._src.objective.linear.to_dict(use_name=True)
        )
        quadratic, q_linear, q_constant, = self._convert_quadratic_coefficients_dict(
            self._src.objective.quadratic.to_dict(use_name=True)
        )

        constant = self._src.objective.constant + linear_constant + q_constant
        for i, v in q_linear.items():
            linear[i] = linear.get(i, 0) + v

        if self._src.objective.sense == QuadraticObjective.Sense.MINIMIZE:
            self._dst.minimize(constant, linear, quadratic)
        else:
            self._dst.maximize(constant, linear, quadratic)

        # set linear constraints
        for constraint in self._src.linear_constraints:
            linear, constant = self._convert_linear_coefficients_dict(
                constraint.linear.to_dict(use_name=True)
            )
            self._dst.linear_constraint(
                linear, constraint.sense, constraint.rhs - constant, constraint.name
            )

        # set quadratic constraints
        for constraint in self._src.quadratic_constraints:
            linear, linear_constant = self._convert_linear_coefficients_dict(
                constraint.linear.to_dict(use_name=True)
            )
            quadratic, q_linear, q_constant = self._convert_quadratic_coefficients_dict(
                constraint.quadratic.to_dict(use_name=True)
            )

            constant = linear_constant + q_constant
            for i, v in q_linear.items():
                linear[i] = linear.get(i, 0) + v

            self._dst.quadratic_constraint(
                linear,
                quadratic,
                constraint.sense,
                constraint.rhs - constant,
                constraint.name,
            )

    def interpret(self, x: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Convert back the converted problem (binary variables)
        to the original (integer variables).

        Args:
            x: The result of the converted problem or the given result in case of FAILURE.

        Returns:
            The result of the original problem.
        """
        # interpret integer values
        sol = {var.name: x[i] for i, var in enumerate(self._dst.variables)}
        new_x = np.zeros(self._src.get_num_vars())
        for i, var in enumerate(self._src.variables):
            if var in self._conv:
                new_x[i] = sum(sol[aux] * coef for aux, coef in self._conv[var]) + var.lowerbound
            else:
                new_x[i] = sol[var.name]
        return np.array(new_x)
