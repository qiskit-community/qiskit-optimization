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

"""Quadratic Constraint."""

from typing import Union, List, Dict, Any

from numpy import ndarray
from scipy.sparse import spmatrix

from ..exceptions import QiskitOptimizationError
from .constraint import Constraint, ConstraintSense
from .linear_expression import LinearExpression
from .variable import Variable


class IndicatorConstraint(Constraint):
    """Representation of an indicator constraint."""

    # Note: added, duplicating in effect that in Constraint, to avoid issues with Sphinx
    Sense = ConstraintSense

    def __init__(
        self,
        quadratic_program: Any,
        name: str,
        binary_var: Union[str, int, Variable],
        linear: Union[ndarray, spmatrix, List[float], Dict[Union[str, int], float]],
        sense: ConstraintSense,
        rhs: float,
        active_value: int = 1,
    ) -> None:
        """Constructs an indicator constraint, consisting of a binary indicator variable and
            linear terms.

        Args:
            quadratic_program: The parent quadratic program.
            name: The name of the constraint.
            binary_var: The binary indicator variable.
            linear: The coefficients specifying the linear part of the constraint.
            sense: The sense of the constraint.
            rhs: The right-hand-side of the constraint.
            active_value: The value of the binary variable is used to force the satisfaction of the
            linear constraint. Default is 1.

        Raises:
            QiskitOptimizationError: If given binary_var is not Variable, the index, or the name
                of the variable.
            QiskitOptimizationError: If binary_var is not a binary variable
            QiskitOptimizationError: If active_value is not 0 or 1.
        """
        # Type check for the arguments
        super().__init__(quadratic_program, name, sense, rhs)
        self._linear = LinearExpression(quadratic_program, linear)
        if isinstance(binary_var, Variable):
            binary_var_ = binary_var
        elif isinstance(binary_var, (int, str)):
            binary_var_ = quadratic_program.get_variable(binary_var)
        else:
            raise QiskitOptimizationError(
                "Unsupported format for binary_var. It must be \
                Variable, the index, or the name: {}".format(
                    type(binary_var)
                )
            )
        if binary_var_.vartype != Variable.Type.BINARY:
            raise QiskitOptimizationError(
                "binary_var must be a binary variable: {}".format(binary_var_.vartype)
            )
        self._binary_var = binary_var_
        if active_value not in (0, 1):
            raise QiskitOptimizationError("Active value must be 1 or 0: {}".format(active_value))
        self._active_value = active_value

    @property
    def active_value(self) -> int:
        """Returns the active value for the binary indicator variable of the constraint.

        Args:
            The active value of the binary indicator variable
        """
        return self._active_value

    @active_value.setter
    def active_value(self, active_value: int) -> None:
        """Set the active value for the binary indicator variable of the constraint.

        Returns:
            The active value
        """
        self._active_value = active_value

    @property
    def binary_var(self) -> Variable:
        """Returns the binary indicator variable of the constraint.

        Returns:
            The binary indicator variable
        """
        return self._binary_var

    @binary_var.setter
    def binary_var(self, binary_var: Variable) -> None:
        """Set the binary indicator variable of the constraint.

        Args:
            binary_var: The binary indicator variable of the constraint.
        """
        self._binary_var = binary_var

    @property
    def linear(self) -> LinearExpression:
        """Returns the linear expression corresponding to the left-hand-side of the constraint.

        Returns:
            The left-hand-side linear expression.
        """
        return self._linear

    @linear.setter
    def linear(
        self,
        linear: Union[ndarray, spmatrix, List[float], Dict[Union[str, int], float]],
    ) -> None:
        """Sets the linear expression corresponding to the left-hand-side of the constraint.
        The coefficients can either be given by an array, a (sparse) 1d matrix, a list or a
        dictionary.

        Args:
            linear: The linear coefficients of the left-hand-side.
        """

        self._linear = LinearExpression(self.quadratic_program, linear)

    def evaluate(self, x: Union[ndarray, List, Dict[Union[int, str], float]]) -> float:
        """Evaluate the left-hand-side of the constraint.

        Args:
            x: The values of the variables to be evaluated.

        Returns:
            The left-hand-side of the constraint given the variable values.
        """
        return self.linear.evaluate(x)

    def evaluate_indicator(self, x: Union[ndarray, List, Dict[Union[int, str], float]]) -> bool:
        """Evaluate the binary indicator var using the active value.

        Args:
            x: The values of the variables to be evaluated.

        Returns:
            If x equals to the active value, return True. Otherwise, return False.

        Raises:
            QiskitOptimizationError: if the given variable index does not match to the index of
            self.binary_var.
            QiskitOptimizationError: if x is given in unsupported format.
        """
        index = self.quadratic_program.variables_index[self.binary_var.name]
        if isinstance(x, (list, ndarray)):
            val = x[index]
        elif isinstance(x, dict):
            for ind, value in x.items():
                if isinstance(ind, str):
                    ind = self.quadratic_program.variables_index[ind]
                if index == ind:
                    val = value
        else:
            raise QiskitOptimizationError("Unsupported format for x.")

        return val == self.active_value
