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

"""The minimum eigensolver interface and result."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from qiskit.quantum_info.operators.base_operator import BaseOperator

from ..algorithm_result import AlgorithmResult
from ..list_or_dict import ListOrDict


class MinimumEigensolver(ABC):
    """The minimum eigensolver interface.

    Algorithms that can compute a minimum eigenvalue for an operator may implement this interface to
    allow different algorithms to be used interchangeably.
    """

    @abstractmethod
    def compute_minimum_eigenvalue(
        self,
        operator: BaseOperator,
        aux_operators: ListOrDict[BaseOperator] | None = None,
    ) -> "MinimumEigensolverResult":
        """
        Computes the minimum eigenvalue. The ``operator`` and ``aux_operators`` are supplied here.
        While an ``operator`` is required by algorithms, ``aux_operators`` are optional.

        Args:
            operator: Qubit operator of the observable.
            aux_operators: Optional list of auxiliary operators to be evaluated with the
                parameters of the minimum eigenvalue main result and their expectation values
                returned. For instance in chemistry these can be dipole operators and total particle
                count operators, so we can get values for these at the ground state.

        Returns:
            A minimum eigensolver result.
        """
        return MinimumEigensolverResult()

    @classmethod
    def supports_aux_operators(cls) -> bool:
        """Whether computing the expectation value of auxiliary operators is supported.

        If the minimum eigensolver computes an eigenvalue of the main ``operator`` then it can
        compute the expectation value of the ``aux_operators`` for that state. Otherwise they will
        be ignored.

        Returns:
            True if aux_operator expectations can be evaluated, False otherwise
        """
        return False


class MinimumEigensolverResult(AlgorithmResult):
    """Minimum eigensolver result."""

    def __init__(self) -> None:
        super().__init__()
        self._eigenvalue: complex | None = None
        self._aux_operators_evaluated: ListOrDict[tuple[complex, dict[str, Any]]] | None = None

    @property
    def eigenvalue(self) -> complex | None:
        """The computed minimum eigenvalue."""
        return self._eigenvalue

    @eigenvalue.setter
    def eigenvalue(self, value: complex) -> None:
        self._eigenvalue = value

    @property
    def aux_operators_evaluated(self) -> ListOrDict[tuple[complex, dict[str, Any]]] | None:
        """The aux operator expectation values.

        These values are in fact tuples formatted as (mean, (variance, shots)).
        """
        return self._aux_operators_evaluated

    @aux_operators_evaluated.setter
    def aux_operators_evaluated(self, value: ListOrDict[tuple[complex, dict[str, Any]]]) -> None:
        self._aux_operators_evaluated = value
