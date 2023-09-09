# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Common classes for rounding schemes"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from qiskit.circuit import QuantumCircuit

from qiskit_optimization.algorithms import SolutionSample

from .quantum_random_access_encoding import QuantumRandomAccessEncoding


@dataclass
class RoundingResult:
    """Result of rounding"""

    expectation_values: list[complex] | None
    """Expectation values"""
    samples: list[SolutionSample]
    """List of samples after rounding"""
    bases: np.ndarray | None = None
    """The bases used for the magic rounding"""
    basis_shots: np.ndarray | None = None
    """The number of shots used for each basis for the magic rounding"""
    basis_counts: list[dict[str, int]] | None = None
    """The basis_counts represents the resulting counts obtained by measuring with the bases
    corresponding to the number of shots specified in basis_shots for the magic rounding."""


@dataclass
class RoundingContext:
    """Information that is provided for rounding"""

    encoding: QuantumRandomAccessEncoding
    """Encoding containing the problem information."""
    expectation_values: list[complex] | None
    """Expectation values for the relaxed Hamiltonian."""
    circuit: QuantumCircuit | None = None
    """Circuit corresponding to the encoding and expectation values."""


class RoundingScheme(ABC):
    """Base class for a rounding scheme"""

    @abstractmethod
    def round(self, rounding_context: RoundingContext) -> RoundingResult:
        """Perform rounding

        Returns: an instance of RoundingResult
        """
