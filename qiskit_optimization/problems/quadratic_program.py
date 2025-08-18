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

"""Quadratic Program."""
from __future__ import annotations

import logging
from collections.abc import Sequence
from enum import Enum
from math import isclose
from typing import cast
from warnings import warn

import numpy as np
from docplex.mp.model_reader import ModelReader
from numpy import ndarray
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.utils import deprecate_func
from scipy.sparse import spmatrix

import qiskit_optimization.optionals as _optionals

from ..exceptions import QiskitOptimizationError
from ..infinity import INFINITY
from .constraint import Constraint, ConstraintSense
from .linear_constraint import LinearConstraint
from .quadratic_constraint import QuadraticConstraint
from .quadratic_objective import QuadraticObjective
from .quadratic_program_element import QuadraticProgramElement
from .variable import Variable, VarType

logger = logging.getLogger(__name__)


class QuadraticProgramStatus(Enum):
    """Status of QuadraticProgram"""

    VALID = 0
    INFEASIBLE = 1


class QuadraticProgram:
    """Quadratically Constrained Quadratic Program representation.

    This representation supports inequality and equality constraints,
    as well as continuous, binary, and integer variables.
    """

    Status = QuadraticProgramStatus

    def __init__(self, name: str = "") -> None:
        """
        Args:
            name: The name of the quadratic program.
        """
        if not name.isprintable():
            warn("Problem name is not printable")
        self._name = ""
        self.name = name
        self._status = QuadraticProgram.Status.VALID

        self._variables: list[Variable] = []
        self._variables_index: dict[str, int] = {}

        self._linear_constraints: list[LinearConstraint] = []
        self._linear_constraints_index: dict[str, int] = {}

        self._quadratic_constraints: list[QuadraticConstraint] = []
        self._quadratic_constraints_index: dict[str, int] = {}

        self._objective = QuadraticObjective(self)

    def __repr__(self) -> str:
        from ..translators.prettyprint import DEFAULT_TRUNCATE, expr2str

        objective = expr2str(
            constant=self._objective.constant,
            linear=self.objective.linear,
            quadratic=self._objective.quadratic,
            truncate=DEFAULT_TRUNCATE,
        )
        num_constraints = self.get_num_linear_constraints() + self.get_num_quadratic_constraints()
        return (
            f"<{self.__class__.__name__}: "
            f"{self.objective.sense.name.lower()} "
            f"{objective}, "
            f"{self.get_num_vars()} variables, "
            f"{num_constraints} constraints, "
            f"'{self._name}'>"
        )

    def __str__(self) -> str:
        num_constraints = self.get_num_linear_constraints() + self.get_num_quadratic_constraints()
        return (
            f"{str(self.objective)} "
            f"({self.get_num_vars()} variables, "
            f"{num_constraints} constraints, "
            f"'{self._name}')"
        )

    def clear(self) -> None:
        """Clears the quadratic program, i.e., deletes all variables, constraints, the
        objective function as well as the name.
        """
        self._name = ""
        self._status = QuadraticProgram.Status.VALID

        self._variables.clear()
        self._variables_index.clear()

        self._linear_constraints.clear()
        self._linear_constraints_index.clear()

        self._quadratic_constraints.clear()
        self._quadratic_constraints_index.clear()

        self._objective = QuadraticObjective(self)

    @property
    def name(self) -> str:
        """Returns the name of the quadratic program.

        Returns:
            The name of the quadratic program.
        """
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """Sets the name of the quadratic program.

        Args:
            name: The name of the quadratic program.
        """
        self._check_name(name, "Problem")
        self._name = name

    @property
    def status(self) -> QuadraticProgramStatus:
        """Status of the quadratic program.
        It can be infeasible due to variable substitution.

        Returns:
            The status of the quadratic program
        """
        return self._status

    @property
    def variables(self) -> list[Variable]:
        """Returns the list of variables of the quadratic program.

        Returns:
            List of variables.
        """
        return self._variables

    @property
    def variables_index(self) -> dict[str, int]:
        """Returns the dictionary that maps the name of a variable to its index.

        Returns:
            The variable index dictionary.
        """
        return self._variables_index

    def _add_variable(
        self,
        lowerbound: float | int,
        upperbound: float | int,
        vartype: VarType,
        name: str | None,
    ) -> Variable:
        if not name:
            name = "x"
            key_format = "{}"
        else:
            key_format = ""
        return self._add_variables(1, lowerbound, upperbound, vartype, name, key_format)[1][0]

    def _add_variables(  # pylint: disable=too-many-positional-arguments
        self,
        keys: int | Sequence,
        lowerbound: float | int,
        upperbound: float | int,
        vartype: VarType,
        name: str | None,
        key_format: str,
    ) -> tuple[list[str], list[Variable]]:
        if isinstance(keys, int) and keys < 1:
            raise QiskitOptimizationError(f"Cannot create non-positive number of variables: {keys}")
        if not name:
            name = "x"
        if "{{}}" in key_format:
            raise QiskitOptimizationError(
                f"Formatter cannot contain nested substitutions: {key_format}"
            )
        if key_format.count("{}") > 1:
            raise QiskitOptimizationError(
                f"Formatter cannot contain more than one substitution: {key_format}"
            )

        def _find_name(name, key_format, k):
            prev = None
            while True:
                new_name = name + key_format.format(k)
                if new_name == prev:
                    raise QiskitOptimizationError(f"Variable name already exists: {new_name}")
                if new_name in self._variables_index:
                    k += 1
                    prev = new_name
                else:
                    break
            return new_name, k + 1

        names = []
        variables = []
        k = self.get_num_vars()
        lst = keys if isinstance(keys, Sequence) else range(keys)
        for key in lst:
            if isinstance(keys, Sequence):
                indexed_name = name + key_format.format(key)
            else:
                indexed_name, k = _find_name(name, key_format, k)
            if indexed_name in self._variables_index:
                raise QiskitOptimizationError(f"Variable name already exists: {indexed_name}")
            self._check_name(indexed_name, "Variable")
            names.append(indexed_name)
            self._variables_index[indexed_name] = self.get_num_vars()
            variable = Variable(self, indexed_name, lowerbound, upperbound, vartype)
            self._variables.append(variable)
            variables.append(variable)
        return names, variables

    def _var_dict(  # pylint: disable=too-many-positional-arguments
        self,
        keys: int | Sequence,
        lowerbound: float | int,
        upperbound: float | int,
        vartype: VarType,
        name: str | None,
        key_format: str,
    ) -> dict[str, Variable]:
        """
        Adds a positive number of variables to the variable list and index and returns a
        dictionary mapping the variable names to their instances. If 'key_format' is present,
        the next 'var_count' available indices are substituted into 'key_format' and appended
        to 'name'.

        Args:
            lowerbound: The lower bound of the variable(s).
            upperbound: The upper bound of the variable(s).
            vartype: The type of the variable(s).
            name: The name(s) of the variable(s).
            key_format: The format used to name/index the variable(s).
            keys: If keys: int, it is interpreted as the number of variables to construct.
                  Otherwise, the elements of the sequence are converted to strings via 'str' and
                  substituted into `key_format`.

        Returns:
            A dictionary mapping the variable names to variable instances.

        Raises:
            QiskitOptimizationError: if the variable name is already taken.
            QiskitOptimizationError: if less than one variable instantiation is attempted.
            QiskitOptimizationError: if `key_format` has more than one substitution or a
                                     nested substitution.
        """
        return dict(
            zip(*self._add_variables(keys, lowerbound, upperbound, vartype, name, key_format))
        )

    def _var_list(  # pylint: disable=too-many-positional-arguments
        self,
        keys: int | Sequence,
        lowerbound: float | int,
        upperbound: float | int,
        vartype: VarType,
        name: str | None,
        key_format: str,
    ) -> list[Variable]:
        """
        Adds a positive number of variables to the variable list and index and returns a
        list of variable instances.

        Args:
            lowerbound: The lower bound of the variable(s).
            upperbound: The upper bound of the variable(s).
            vartype: The type of the variable(s).
            name: The name(s) of the variable(s).
            key_format: The format used to name/index the variable(s).
            keys: If keys: int, it is interpreted as the number of variables to construct.
                  Otherwise, the elements of the sequence are converted to strings via 'str' and
                  substituted into `key_format`.

        Returns:
            A dictionary mapping the variable names to variable instances.

        Raises:
            QiskitOptimizationError: if the variable name is already taken.
            QiskitOptimizationError: if less than one variable instantiation is attempted.
            QiskitOptimizationError: if `key_format` has more than one substitution or a
                                     nested substitution.
        """
        return self._add_variables(keys, lowerbound, upperbound, vartype, name, key_format)[1]

    def continuous_var(
        self,
        lowerbound: float | int = 0,
        upperbound: float | int = INFINITY,
        name: str | None = None,
    ) -> Variable:
        """Adds a continuous variable to the quadratic program.

        Args:
            lowerbound: The lowerbound of the variable.
            upperbound: The upperbound of the variable.
            name: The name of the variable.
                If it's ``None`` or empty ``""``, the default name, e.g., ``x0``, is used.

        Returns:
            The added variable.

        Raises:
            QiskitOptimizationError: if the variable name is already occupied.
        """
        return self._add_variable(lowerbound, upperbound, Variable.Type.CONTINUOUS, name)

    def continuous_var_dict(  # pylint: disable=too-many-positional-arguments
        self,
        keys: int | Sequence,
        lowerbound: float | int = 0,
        upperbound: float | int = INFINITY,
        name: str | None = None,
        key_format: str = "{}",
    ) -> dict[str, Variable]:
        """
        Uses 'var_dict' to construct a dictionary of continuous variables

        Args:
            lowerbound: The lower bound of the variable(s).
            upperbound: The upper bound of the variable(s).
            name: The name(s) of the variable(s).
                If it's ``None`` or empty ``""``, the default name, e.g., ``x0``, is used.
            key_format: The format used to name/index the variable(s).
            keys: If keys: int, it is interpreted as the number of variables to construct.
                  Otherwise, the elements of the sequence are converted to strings via 'str' and
                  substituted into `key_format`.

        Returns:
            A dictionary mapping the variable names to variable instances.

        Raises:
            QiskitOptimizationError: if the variable name is already taken.
            QiskitOptimizationError: if less than one variable instantiation is attempted.
            QiskitOptimizationError: if `key_format` has more than one substitution or a
                                     nested substitution.
        """
        return self._var_dict(
            keys=keys,
            lowerbound=lowerbound,
            upperbound=upperbound,
            vartype=Variable.Type.CONTINUOUS,
            name=name,
            key_format=key_format,
        )

    def continuous_var_list(  # pylint: disable=too-many-positional-arguments
        self,
        keys: int | Sequence,
        lowerbound: float | int = 0,
        upperbound: float | int = INFINITY,
        name: str | None = None,
        key_format: str = "{}",
    ) -> list[Variable]:
        """
        Uses 'var_list' to construct a list of continuous variables

        Args:
            lowerbound: The lower bound of the variable(s).
            upperbound: The upper bound of the variable(s).
            name: The name(s) of the variable(s).
                If it's ``None`` or empty ``""``, the default name, e.g., ``x0``, is used.
            key_format: The format used to name/index the variable(s).
            keys: If keys: int, it is interpreted as the number of variables to construct.
                  Otherwise, the elements of the sequence are converted to strings via 'str' and
                  substituted into `key_format`.

        Returns:
            A list of variable instances.

        Raises:
            QiskitOptimizationError: if the variable name is already taken.
            QiskitOptimizationError: if less than one variable instantiation is attempted.
            QiskitOptimizationError: if `key_format` has more than one substitution or a
                                     nested substitution.
        """
        return self._var_list(
            keys, lowerbound, upperbound, Variable.Type.CONTINUOUS, name, key_format
        )

    def binary_var(self, name: str | None = None) -> Variable:
        """Adds a binary variable to the quadratic program.

        Args:
            name: The name of the variable.
                If it's ``None`` or empty ``""``, the default name, e.g., ``x0``, is used.

        Returns:
            The added variable.

        Raises:
            QiskitOptimizationError: if the variable name is already occupied.
        """
        return self._add_variable(0, 1, Variable.Type.BINARY, name)

    def binary_var_dict(
        self,
        keys: int | Sequence,
        name: str | None = None,
        key_format: str = "{}",
    ) -> dict[str, Variable]:
        """
        Uses 'var_dict' to construct a dictionary of binary variables

        Args:
            name: The name(s) of the variable(s).
                If it's ``None`` or empty ``""``, the default name, e.g., ``x0``, is used.
            key_format: The format used to name/index the variable(s).
            keys: If keys: int, it is interpreted as the number of variables to construct.
                  Otherwise, the elements of the sequence are converted to strings via 'str' and
                  substituted into `key_format`.

        Returns:
            A dictionary mapping the variable names to variable instances.

        Raises:
            QiskitOptimizationError: if the variable name is already taken.
            QiskitOptimizationError: if less than one variable instantiation is attempted.
            QiskitOptimizationError: if `key_format` has more than one substitution or a
                                     nested substitution.
        """
        return self._var_dict(
            keys=keys,
            lowerbound=0,
            upperbound=1,
            vartype=Variable.Type.BINARY,
            name=name,
            key_format=key_format,
        )

    def binary_var_list(
        self,
        keys: int | Sequence,
        name: str | None = None,
        key_format: str = "{}",
    ) -> list[Variable]:
        """
        Uses 'var_list' to construct a list of binary variables

        Args:
            name: The name(s) of the variable(s).
                If it's ``None`` or empty ``""``, the default name, e.g., ``x0``, is used.
            key_format: The format used to name/index the variable(s).
            keys: If keys: int, it is interpreted as the number of variables to construct.
                  Otherwise, the elements of the sequence are converted to strings via 'str' and
                  substituted into `key_format`.

        Returns:
            A list of variable instances.

        Raises:
            QiskitOptimizationError: if the variable name is already taken.
            QiskitOptimizationError: if less than one variable instantiation is attempted.
            QiskitOptimizationError: if `key_format` has more than one substitution or a
                                     nested substitution.
        """
        return self._var_list(keys, 0, 1, Variable.Type.BINARY, name, key_format)

    def integer_var(
        self,
        lowerbound: float | int = 0,
        upperbound: float | int = INFINITY,
        name: str | None = None,
    ) -> Variable:
        """Adds an integer variable to the quadratic program.

        Args:
            lowerbound: The lowerbound of the variable.
            upperbound: The upperbound of the variable.
            name: The name of the variable.
                If it's ``None`` or empty ``""``, the default name, e.g., ``x0``, is used.

        Returns:
            The added variable.

        Raises:
            QiskitOptimizationError: if the variable name is already occupied.
        """
        return self._add_variable(lowerbound, upperbound, Variable.Type.INTEGER, name)

    def integer_var_dict(  # pylint: disable=too-many-positional-arguments
        self,
        keys: int | Sequence,
        lowerbound: float | int = 0,
        upperbound: float | int = INFINITY,
        name: str | None = None,
        key_format: str = "{}",
    ) -> dict[str, Variable]:
        """
        Uses 'var_dict' to construct a dictionary of integer variables

        Args:
            lowerbound: The lower bound of the variable(s).
            upperbound: The upper bound of the variable(s).
            name: The name(s) of the variable(s).
                If it's ``None`` or empty ``""``, the default name, e.g., ``x0``, is used.
            key_format: The format used to name/index the variable(s).
            keys: If keys: int, it is interpreted as the number of variables to construct.
                  Otherwise, the elements of the sequence are converted to strings via 'str' and
                  substituted into `key_format`.

        Returns:
            A dictionary mapping the variable names to variable instances.

        Raises:
            QiskitOptimizationError: if the variable name is already taken.
            QiskitOptimizationError: if less than one variable instantiation is attempted.
            QiskitOptimizationError: if `key_format` has more than one substitution or a
                                     nested substitution.
        """
        return self._var_dict(
            keys=keys,
            lowerbound=lowerbound,
            upperbound=upperbound,
            vartype=Variable.Type.INTEGER,
            name=name,
            key_format=key_format,
        )

    def integer_var_list(  # pylint: disable=too-many-positional-arguments
        self,
        keys: int | Sequence,
        lowerbound: float | int = 0,
        upperbound: float | int = INFINITY,
        name: str | None = None,
        key_format: str = "{}",
    ) -> list[Variable]:
        """
        Uses 'var_list' to construct a list of integer variables

        Args:
            lowerbound: The lower bound of the variable(s).
            upperbound: The upper bound of the variable(s).
            name: The name(s) of the variable(s).
                If it's ``None`` or empty ``""``, the default name, e.g., ``x0``, is used.
            key_format: The format used to name/index the variable(s).
            keys: If keys: int, it is interpreted as the number of variables to construct.
                  Otherwise, the elements of the sequence are converted to strings via 'str' and
                  substituted into `key_format`.

        Returns:
            A list of variable instances.

        Raises:
            QiskitOptimizationError: if the variable name is already taken.
            QiskitOptimizationError: if less than one variable instantiation is attempted.
            QiskitOptimizationError: if `key_format` has more than one substitution or a
                                     nested substitution.
        """
        return self._var_list(keys, lowerbound, upperbound, Variable.Type.INTEGER, name, key_format)

    def get_variable(self, i: int | str) -> Variable:
        """Returns a variable for a given name or index.

        Args:
            i: the index or name of the variable.

        Returns:
            The corresponding variable.
        """
        if isinstance(i, (int, np.integer)):
            return self.variables[i]
        else:
            return self.variables[self._variables_index[i]]

    def get_num_vars(self, vartype: VarType | None = None) -> int:
        """Returns the total number of variables or the number of variables of the specified type.

        Args:
            vartype: The type to be filtered on. All variables are counted if None.

        Returns:
            The total number of variables.
        """
        if vartype:
            return sum(variable.vartype == vartype for variable in self._variables)
        else:
            return len(self._variables)

    def get_num_continuous_vars(self) -> int:
        """Returns the total number of continuous variables.

        Returns:
            The total number of continuous variables.
        """
        return self.get_num_vars(Variable.Type.CONTINUOUS)

    def get_num_binary_vars(self) -> int:
        """Returns the total number of binary variables.

        Returns:
            The total number of binary variables.
        """
        return self.get_num_vars(Variable.Type.BINARY)

    def get_num_integer_vars(self) -> int:
        """Returns the total number of integer variables.

        Returns:
            The total number of integer variables.
        """
        return self.get_num_vars(Variable.Type.INTEGER)

    @property
    def linear_constraints(self) -> list[LinearConstraint]:
        """Returns the list of linear constraints of the quadratic program.

        Returns:
            List of linear constraints.
        """
        return self._linear_constraints

    @property
    def linear_constraints_index(self) -> dict[str, int]:
        """Returns the dictionary that maps the name of a linear constraint to its index.

        Returns:
            The linear constraint index dictionary.
        """
        return self._linear_constraints_index

    def linear_constraint(
        self,
        linear: ndarray | spmatrix | list[float] | dict[int | str, float] | None = None,
        sense: str | ConstraintSense = "<=",
        rhs: float = 0.0,
        name: str | None = None,
    ) -> LinearConstraint:
        """Adds a linear equality constraint to the quadratic program of the form:
            ``(linear * x) sense rhs``.

        Args:
            linear: The linear coefficients of the left-hand side of the constraint.
            sense: The sense of the constraint,

              - ``==``, ``=``, ``E``, and ``EQ`` denote 'equal to'.
              - ``>=``, ``>``, ``G``, and ``GE`` denote 'greater-than-or-equal-to'.
              - ``<=``, ``<``, ``L``, and ``LE`` denote 'less-than-or-equal-to'.

            rhs: The right-hand side of the constraint.
            name: The name of the constraint.
                If it's ``None`` or empty ``""``, the default name, e.g., ``c0``, is used.

        Returns:
            The added constraint.

        Raises:
            QiskitOptimizationError: if the constraint name already exists or the sense is not
                valid.
        """
        if name:
            if name in self.linear_constraints_index:
                raise QiskitOptimizationError(f"Linear constraint's name already exists: {name}")
            self._check_name(name, "Linear constraint")
        else:
            k = self.get_num_linear_constraints()
            while f"c{k}" in self.linear_constraints_index:
                k += 1
            name = f"c{k}"
        self.linear_constraints_index[name] = len(self.linear_constraints)
        if linear is None:
            linear = {}
        constraint = LinearConstraint(self, name, linear, Constraint.Sense.convert(sense), rhs)
        self.linear_constraints.append(constraint)
        return constraint

    def get_linear_constraint(self, i: int | str) -> LinearConstraint:
        """Returns a linear constraint for a given name or index.

        Args:
            i: the index or name of the constraint.

        Returns:
            The corresponding constraint.

        Raises:
            IndexError: if the index is out of the list size
            KeyError: if the name does not exist
        """
        if isinstance(i, int):
            return self._linear_constraints[i]
        else:
            return self._linear_constraints[self._linear_constraints_index[i]]

    def get_num_linear_constraints(self) -> int:
        """Returns the number of linear constraints.

        Returns:
            The number of linear constraints.
        """
        return len(self._linear_constraints)

    @property
    def quadratic_constraints(self) -> list[QuadraticConstraint]:
        """Returns the list of quadratic constraints of the quadratic program.

        Returns:
            List of quadratic constraints.
        """
        return self._quadratic_constraints

    @property
    def quadratic_constraints_index(self) -> dict[str, int]:
        """Returns the dictionary that maps the name of a quadratic constraint to its index.

        Returns:
            The quadratic constraint index dictionary.
        """
        return self._quadratic_constraints_index

    def quadratic_constraint(  # pylint: disable=too-many-positional-arguments
        self,
        linear: ndarray | spmatrix | list[float] | dict[int | str, float] | None = None,
        quadratic: (
            ndarray | spmatrix | list[list[float]] | dict[tuple[int | str, int | str], float] | None
        ) = None,
        sense: str | ConstraintSense = "<=",
        rhs: float = 0.0,
        name: str | None = None,
    ) -> QuadraticConstraint:
        """Adds a quadratic equality constraint to the quadratic program of the form:
            ``(x * quadratic * x + linear * x) sense rhs``.

        Args:
            linear: The linear coefficients of the constraint.
            quadratic: The quadratic coefficients of the constraint.
            sense: The sense of the constraint,

              - ``==``, ``=``, ``E``, and ``EQ`` denote 'equal to'.
              - ``>=``, ``>``, ``G``, and ``GE`` denote 'greater-than-or-equal-to'.
              - ``<=``, ``<``, ``L``, and ``LE`` denote 'less-than-or-equal-to'.

            rhs: The right-hand side of the constraint.
            name: The name of the constraint.
                If it's ``None`` or empty ``""``, the default name, e.g., ``q0``, is used.

        Returns:
            The added constraint.

        Raises:
            QiskitOptimizationError: if the constraint name already exists.
        """
        if name:
            if name in self.quadratic_constraints_index:
                raise QiskitOptimizationError(f"Quadratic constraint name already exists: {name}")
            self._check_name(name, "Quadratic constraint")
        else:
            k = self.get_num_quadratic_constraints()
            while f"q{k}" in self.quadratic_constraints_index:
                k += 1
            name = f"q{k}"
        self.quadratic_constraints_index[name] = len(self.quadratic_constraints)
        if linear is None:
            linear = {}
        if quadratic is None:
            quadratic = {}
        constraint = QuadraticConstraint(
            self, name, linear, quadratic, Constraint.Sense.convert(sense), rhs
        )
        self.quadratic_constraints.append(constraint)
        return constraint

    def get_quadratic_constraint(self, i: int | str) -> QuadraticConstraint:
        """Returns a quadratic constraint for a given name or index.

        Args:
            i: the index or name of the constraint.

        Returns:
            The corresponding constraint.

        Raises:
            IndexError: if the index is out of the list size
            KeyError: if the name does not exist
        """
        if isinstance(i, int):
            return self._quadratic_constraints[i]
        else:
            return self._quadratic_constraints[self._quadratic_constraints_index[i]]

    def get_num_quadratic_constraints(self) -> int:
        """Returns the number of quadratic constraints.

        Returns:
            The number of quadratic constraints.
        """
        return len(self._quadratic_constraints)

    def remove_linear_constraint(self, i: str | int) -> None:
        """Remove a linear constraint

        Args:
            i: an index or a name of a linear constraint

        Raises:
            KeyError: if name does not exist
            IndexError: if index is out of range
        """
        if isinstance(i, str):
            i = self._linear_constraints_index[i]
        del self._linear_constraints[i]
        self._linear_constraints_index = {
            cst.name: j for j, cst in enumerate(self._linear_constraints)
        }

    def remove_quadratic_constraint(self, i: str | int) -> None:
        """Remove a quadratic constraint

        Args:
            i: an index or a name of a quadratic constraint

        Raises:
            KeyError: if name does not exist
            IndexError: if index is out of range
        """
        if isinstance(i, str):
            i = self._quadratic_constraints_index[i]
        del self._quadratic_constraints[i]
        self._quadratic_constraints_index = {
            cst.name: j for j, cst in enumerate(self._quadratic_constraints)
        }

    @property
    def objective(self) -> QuadraticObjective:
        """Returns the quadratic objective.

        Returns:
            The quadratic objective.
        """
        return self._objective

    def minimize(
        self,
        constant: float = 0.0,
        linear: ndarray | spmatrix | list[float] | dict[str | int, float] | None = None,
        quadratic: (
            ndarray | spmatrix | list[list[float]] | dict[tuple[int | str, int | str], float] | None
        ) = None,
    ) -> None:
        """Sets a quadratic objective to be minimized.

        Args:
            constant: the constant offset of the objective.
            linear: the coefficients of the linear part of the objective.
            quadratic: the coefficients of the quadratic part of the objective.

        Returns:
            The created quadratic objective.
        """
        self._objective = QuadraticObjective(
            self, constant, linear, quadratic, QuadraticObjective.Sense.MINIMIZE
        )

    def maximize(
        self,
        constant: float = 0.0,
        linear: ndarray | spmatrix | list[float] | dict[str | int, float] | None = None,
        quadratic: (
            ndarray | spmatrix | list[list[float]] | dict[tuple[int | str, int | str], float] | None
        ) = None,
    ) -> None:
        """Sets a quadratic objective to be maximized.

        Args:
            constant: the constant offset of the objective.
            linear: the coefficients of the linear part of the objective.
            quadratic: the coefficients of the quadratic part of the objective.

        Returns:
            The created quadratic objective.
        """
        self._objective = QuadraticObjective(
            self, constant, linear, quadratic, QuadraticObjective.Sense.MAXIMIZE
        )

    def _copy_from(self, other: QuadraticProgram, include_name: bool) -> None:
        """Copy another QuadraticProgram to this updating QuadraticProgramElement

        Note: this breaks the consistency of `other`. You cannot use `other` after the copy.

        Args:
            other: The quadratic program to be copied from.
            include_name: Whether this method copies the problem name or not.
        """
        for attr, val in vars(other).items():
            if attr == "_name" and not include_name:
                continue
            if isinstance(val, QuadraticProgramElement):
                val.quadratic_program = self
            if isinstance(val, list):
                for elem in val:
                    if isinstance(elem, QuadraticProgramElement):
                        elem.quadratic_program = self
            setattr(self, attr, val)

    @deprecate_func(since="0.7.0", additional_msg="Use prettyprint instead.")
    def export_as_lp_string(self) -> str:
        """Returns the quadratic program as a string of LP format.

        Returns:
            A string representing the quadratic program.
        """
        # pylint: disable=cyclic-import
        from ..translators.docplex_mp import to_docplex_mp

        return to_docplex_mp(self).export_as_lp_string()

    @_optionals.HAS_CPLEX.require_in_call
    @deprecate_func(since="0.7.0", additional_msg="Use from_docplex_mp or from_gurobipy instead.")
    def read_from_lp_file(self, filename: str) -> None:
        """Loads the quadratic program from a LP file.

        Args:
            filename: The filename of the file to be loaded.

        Raises:
            FileNotFoundError: If the file does not exist.

        Note:
            This method requires CPLEX to be installed and present in ``PYTHONPATH``.
        """

        def _parse_problem_name(filename: str) -> str:
            # Because docplex model reader uses the base name as model name,
            # we parse the model name in the LP file manually.
            # https://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.model_reader.html
            prefix = "\\Problem name:"
            model_name = ""
            with open(filename, encoding="utf8") as file:
                for line in file:
                    if line.startswith(prefix):
                        model_name = line[len(prefix) :].strip()
                    if not line.startswith("\\"):
                        break
            return model_name

        # pylint: disable=cyclic-import
        from ..translators.docplex_mp import from_docplex_mp

        model = ModelReader().read(filename, model_name=_parse_problem_name(filename))
        other = from_docplex_mp(model)
        self._copy_from(other, include_name=True)

    @deprecate_func(since="0.7.0", additional_msg="Use to_docplex_mp or to_gurobipy instead.")
    def write_to_lp_file(self, filename: str) -> None:
        """Writes the quadratic program to an LP file.

        Args:
            filename: The filename of the file the model is written to.
              If filename is a directory, file name 'my_problem.lp' is appended.
              If filename does not end with '.lp', suffix '.lp' is appended.

        Raises:
            OSError: If this cannot open a file.
            DOcplexException: If filename is an empty string
        """
        # pylint: disable=cyclic-import
        from ..translators.docplex_mp import to_docplex_mp

        mdl = to_docplex_mp(self)
        mdl.export_as_lp(filename)

    def substitute_variables(
        self,
        constants: dict[str | int, float] | None = None,
        variables: dict[str | int, tuple[str | int, float]] | None = None,
    ) -> QuadraticProgram:
        """Substitutes variables with constants or other variables.

        Args:
            constants: replace variable by constant
                e.g., ``{'x': 2}`` means ``x`` is substituted with 2

            variables: replace variables by weighted other variable
                need to copy everything using name reference to make sure that indices are matched
                correctly. The lower and upper bounds are updated accordingly.
                e.g., ``{'x': ('y', 2)}`` means ``x`` is substituted with ``y * 2``

        Returns:
            An optimization problem by substituting variables with constants or other variables.
            If the substitution is valid, ``QuadraticProgram.status`` is still
            ``QuadraticProgram.Status.VALID``.
            Otherwise, it gets ``QuadraticProgram.Status.INFEASIBLE``.

        Raises:
            QiskitOptimizationError: if the substitution is invalid as follows.

                - Same variable is substituted multiple times.
                - Coefficient of variable substitution is zero.
        """
        # pylint: disable=cyclic-import
        from .substitute_variables import substitute_variables

        return substitute_variables(self, constants, variables)

    def to_ising(self) -> tuple[SparsePauliOp, float]:
        """Return the Ising Hamiltonian of this problem.

        Variables are mapped to qubits in the same order, i.e.,
        i-th variable is mapped to i-th qubit.
        See https://github.com/Qiskit/qiskit-terra/issues/1148 for details.

        Returns:
            qubit_op: The qubit operator for the problem
            offset: The constant value in the Ising Hamiltonian.

        Raises:
            QiskitOptimizationError: If a variable type is not binary.
            QiskitOptimizationError: If constraints exist in the problem.
        """
        # pylint: disable=cyclic-import
        from ..translators.ising import to_ising

        return to_ising(self)

    def from_ising(
        self,
        qubit_op: BaseOperator,
        offset: float = 0.0,
        linear: bool = False,
    ) -> None:
        r"""Create a quadratic program from a qubit operator and a shift value.

        Variables are mapped to qubits in the same order, i.e.,
        i-th variable is mapped to i-th qubit.
        See https://github.com/Qiskit/qiskit-terra/issues/1148 for details.

        Args:
            qubit_op: The qubit operator of the problem.
            offset: The constant value in the Ising Hamiltonian.
            linear: If linear is True, :math:`x^2` is treated as a linear term
                since :math:`x^2 = x` for :math:`x \in \{0,1\}`.
                Else, :math:`x^2` is treated as a quadratic term.
                The default value is False.

        Raises:
            QiskitOptimizationError: If there are Pauli Xs in any Pauli term
            QiskitOptimizationError: If there are more than 2 Pauli Zs in any Pauli term
            NotImplementedError: If the input operator is a ListOp
        """
        # pylint: disable=cyclic-import
        from ..translators.ising import from_ising

        other = from_ising(qubit_op, offset, linear)
        self._copy_from(other, include_name=False)

    def get_feasibility_info(
        self, x: list[float] | np.ndarray
    ) -> tuple[bool, list[Variable], list[Constraint]]:
        """Returns whether a solution is feasible or not along with the violations.
        Args:
            x: a solution value, such as returned in an optimizer result.
        Returns:
            feasible: Whether the solution provided is feasible or not.
            List[Variable]: List of variables which are violated.
            List[Constraint]: List of constraints which are violated.

        Raises:
            QiskitOptimizationError: If the input `x` is not same len as total vars
        """
        # if input `x` is not the same len as the total vars, raise an error
        if len(x) != self.get_num_vars():
            raise QiskitOptimizationError(
                f"The size of solution `x`: {len(x)}, does not match the number of problem variables: "
                f"{self.get_num_vars()}"
            )

        # check whether the input satisfy the bounds of the problem
        violated_variables = []
        for i, val in enumerate(x):
            variable = self.get_variable(i)
            if val < variable.lowerbound or variable.upperbound < val:
                violated_variables.append(variable)

        # check whether the input satisfy the constraints of the problem
        violated_constraints = []
        for constraint in cast(list[Constraint], self._linear_constraints) + cast(
            list[Constraint], self._quadratic_constraints
        ):
            lhs = constraint.evaluate(x)
            if constraint.sense == ConstraintSense.LE and lhs > constraint.rhs:
                violated_constraints.append(constraint)
            elif constraint.sense == ConstraintSense.GE and lhs < constraint.rhs:
                violated_constraints.append(constraint)
            elif constraint.sense == ConstraintSense.EQ and not isclose(lhs, constraint.rhs):
                violated_constraints.append(constraint)

        feasible = not violated_variables and not violated_constraints

        return feasible, violated_variables, violated_constraints

    def is_feasible(self, x: list[float] | np.ndarray) -> bool:
        """Returns whether a solution is feasible or not.

        Args:
            x: a solution value, such as returned in an optimizer result.

        Returns:
            ``True`` if the solution provided is feasible otherwise ``False``.

        """
        feasible, _, _ = self.get_feasibility_info(x)

        return feasible

    def prettyprint(self, wrap: int = 80) -> str:
        """Returns a pretty printed string of this problem.

        Args:
            wrap: The text width to wrap the output strings. It is disabled by setting 0.
                Note that some strings might exceed this value, for example, a long variable
                name won't be wrapped. The default value is 80.

        Returns:
            A pretty printed string representing the problem.

        Raises:
            QiskitOptimizationError: if there is a non-printable name.
        """
        # pylint: disable=cyclic-import
        from qiskit_optimization.translators.prettyprint import prettyprint

        return prettyprint(self, wrap)

    @staticmethod
    def _check_name(name: str, name_type: str) -> None:
        """Displays a warning message if a name string is not printable"""
        if not name.isprintable():
            warn(f"{name_type} name is not printable: {repr(name)}")
