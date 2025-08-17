# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quadratic Objective."""
from __future__ import annotations

from enum import Enum
from typing import Any

from numpy import ndarray
from scipy.sparse import spmatrix

from ..exceptions import QiskitOptimizationError
from .linear_constraint import LinearExpression
from .quadratic_expression import QuadraticExpression
from .quadratic_program_element import QuadraticProgramElement


class ObjSense(Enum):
    """Objective Sense Type."""

    MINIMIZE = 1
    MAXIMIZE = -1


class QuadraticObjective(QuadraticProgramElement):
    """Representation of quadratic objective function of the form:
    constant + linear * x + x * quadratic * x.
    """

    Sense = ObjSense

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        quadratic_program: Any,
        constant: float = 0.0,
        linear: ndarray | spmatrix | list[float] | dict[str | int, float] | None = None,
        quadratic: (
            ndarray | spmatrix | list[list[float]] | dict[tuple[int | str, int | str], float] | None
        ) = None,
        sense: ObjSense = ObjSense.MINIMIZE,
    ) -> None:
        """Constructs a quadratic objective function.

        Args:
            quadratic_program: The parent quadratic program.
            constant: The constant offset of the objective.
            linear: The coefficients of the linear part of the objective.
            quadratic: The coefficients of the quadratic part of the objective.
            sense: The optimization sense of the objective.
        """
        super().__init__(quadratic_program)
        self._constant = constant
        if linear is None:
            linear = {}
        self._linear = LinearExpression(quadratic_program, linear)
        if quadratic is None:
            quadratic = {}
        self._quadratic = QuadraticExpression(quadratic_program, quadratic)
        self._sense = sense

    @property
    def constant(self) -> float:
        """Returns the constant part of the objective function.

        Returns:
            The constant part of the objective function.
        """
        return self._constant

    @constant.setter
    def constant(self, constant: float) -> None:
        """Sets the constant part of the objective function.

        Args:
            constant: The constant part of the objective function.
        """
        self._constant = constant

    @property
    def linear(self) -> LinearExpression:
        """Returns the linear part of the objective function.

        Returns:
            The linear part of the objective function.
        """
        return self._linear

    @linear.setter
    def linear(
        self,
        linear: ndarray | spmatrix | list[float] | dict[str | int, float],
    ) -> None:
        """Sets the coefficients of the linear part of the objective function.

        Args:
            linear: The coefficients of the linear part of the objective function.

        """
        self._linear = LinearExpression(self.quadratic_program, linear)

    @property
    def quadratic(self) -> QuadraticExpression:
        """Returns the quadratic part of the objective function.

        Returns:
            The quadratic part of the objective function.
        """
        return self._quadratic

    @quadratic.setter
    def quadratic(
        self,
        quadratic: (
            ndarray | spmatrix | list[list[float]] | dict[tuple[int | str, int | str], float]
        ),
    ) -> None:
        """Sets the coefficients of the quadratic part of the objective function.

        Args:
            quadratic: The coefficients of the quadratic part of the objective function.

        """
        self._quadratic = QuadraticExpression(self.quadratic_program, quadratic)

    @property
    def sense(self) -> ObjSense:
        """Returns the sense of the objective function.

        Returns:
            The sense of the objective function.
        """
        return self._sense

    @sense.setter
    def sense(self, sense: ObjSense) -> None:
        """Sets the sense of the objective function.

        Args:
            sense: The sense of the objective function.
        """
        self._sense = sense

    def evaluate(self, x: ndarray | list | dict[int | str, float]) -> float:
        """Evaluate the quadratic objective for given variable values.

        Args:
            x: The values of the variables to be evaluated.

        Returns:
            The value of the quadratic objective given the variable values.

        Raises:
            QiskitOptimizationError: if the shape of the objective function does not match with
                the number of variables.
        """
        n = self.quadratic_program.get_num_vars()
        if self.linear.coefficients.shape != (1, n) or self.quadratic.coefficients.shape != (n, n):
            raise QiskitOptimizationError(
                "The shape of the objective function does not match with the number of variables. "
                "Need to define the objective function after defining all variables"
            )
        return self.constant + self.linear.evaluate(x) + self.quadratic.evaluate(x)

    def evaluate_gradient(self, x: ndarray | list | dict[int | str, float]) -> ndarray:
        """Evaluate the gradient of the quadratic objective for given variable values.

        Args:
            x: The values of the variables to be evaluated.

        Returns:
            The value of the gradient of the quadratic objective given the variable values.

        Raises:
            QiskitOptimizationError: if the shape of the objective function does not match with
                the number of variables.
        """
        n = self.quadratic_program.get_num_vars()
        if self.linear.coefficients.shape != (1, n) or self.quadratic.coefficients.shape != (n, n):
            raise QiskitOptimizationError(
                "The shape of the objective function does not match with the number of variables. "
                "Need to define the objective function after defining all variables"
            )
        return self.linear.evaluate_gradient(x) + self.quadratic.evaluate_gradient(x)

    def __repr__(self):
        # pylint: disable=cyclic-import
        from ..translators.prettyprint import expr2str, DEFAULT_TRUNCATE

        expr_str = expr2str(self.constant, self.linear, self.quadratic, DEFAULT_TRUNCATE)
        return f"<{self.__class__.__name__}: {self._sense.name.lower()} {expr_str}>"

    def __str__(self):
        # pylint: disable=cyclic-import
        from ..translators.prettyprint import expr2str

        expr_str = expr2str(self.constant, self.linear, self.quadratic)
        return f"{self._sense.name.lower()} {expr_str}"
