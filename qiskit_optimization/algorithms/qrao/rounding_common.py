# This code is part of Qiskit.
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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from qiskit.circuit import QuantumCircuit

from qiskit_optimization.algorithms import SolutionSample

from .quantum_random_access_encoding import QuantumRandomAccessEncoding


@dataclass
class RoundingResult:
    """Base class for a rounding result"""

    expectation_values: List[float]
    """Expectation values"""
    samples: List[SolutionSample]
    """List of samples after rounding"""


class RoundingContext:
    """Information that is provided for rounding"""

    def __init__(
        self,
        encoding: QuantumRandomAccessEncoding,
        expectation_values: List[float],
        circuit: Optional[QuantumCircuit] = None,
    ):
        """
        Args:
            encoding: Encoding containing the problem information.
            expectation_values: Expectation values of the encoding.
            circuit: circuit corresponding to the encoding and expectation values.
        """
        self.encoding = encoding
        self.expectation_values = expectation_values
        self.circuit = circuit


class RoundingScheme(ABC):
    """Base class for a rounding scheme"""

    @abstractmethod
    def round(self, ctx: RoundingContext) -> RoundingResult:
        """Perform rounding

        Returns: an instance of RoundingResult
        """
