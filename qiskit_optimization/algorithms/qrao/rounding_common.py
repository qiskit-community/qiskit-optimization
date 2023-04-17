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

from typing import List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from qiskit.circuit import QuantumCircuit

from .quantum_random_access_encoding import QuantumRandomAccessEncoding


# pylint: disable=too-few-public-methods


@dataclass
class RoundingSolutionSample:
    """Partial SolutionSample for use in rounding results"""

    x: np.ndarray
    probability: float


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


class RoundingResult:
    """Base class for a rounding result"""

    def __init__(
        self,
        samples: List[RoundingSolutionSample],
        expectation_values: List[float],
        time_taken: Optional[float] = None,
    ):
        """
        Args:
            samples: List of samples of the rounding.
            expectation_values: Expectation values of the encoding.
            time_taken: Time taken for rounding.
        """
        self._samples = samples
        self._expectation_values = expectation_values
        self.time_taken = time_taken

    @property
    def samples(self) -> List[RoundingSolutionSample]:
        """List of samples for the rounding"""
        return self._samples

    @property
    def expectation_values(self):
        """Expectation values"""
        return self._expectation_values


class RoundingScheme(ABC):
    """Base class for a rounding scheme"""

    @abstractmethod
    def round(self, ctx: RoundingContext) -> RoundingResult:
        """Perform rounding

        Returns: an instance of RoundingResult
        """
