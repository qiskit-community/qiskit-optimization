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

"""Common classes for rounding schemes"""

from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

# from qiskit.opflow import PrimitiveOp
from qiskit.circuit import QuantumCircuit

from .quantum_random_access_encoding import QuantumRandomAccessEncoding, q2vars_from_var2op


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
        *,
        encoding: QuantumRandomAccessEncoding,
        # var2op: Optional[Dict[int, Tuple[int, PrimitiveOp]]] = None,
        # q2vars: Optional[List[List[int]]] = None,
        expectation_values,
        circuit: Optional[QuantumCircuit] = None,
        # _vars_per_qubit: Optional[int] = None,
    ):
        # if encoding is not None:
        #     if var2op is not None or q2vars is not None:
        #         raise ValueError(
        #             "Neither var2op nor q2vars should be provided if encoding is"
        #         )
        #     if _vars_per_qubit is not None:
        #         raise ValueError(
        #             "_vars_per_qubit should not be provided if encoding is"
        #         )
        #     self.var2op = encoding.var2op
        #     self.q2vars = encoding.q2vars
        #     self._vars_per_qubit = encoding.max_vars_per_qubit
        # else:
        #     if var2op is None:
        #         raise ValueError("Either an encoding or var2ops must be provided")
        #     if _vars_per_qubit is None:
        #         raise ValueError(
        #             "_vars_per_qubit must be provided if encoding is not provided"
        #         )
        #     self.var2op = var2op
        #     self.q2vars = q2vars_from_var2op(var2op) if q2vars is None else q2vars
        #     self._vars_per_qubit = _vars_per_qubit

        self.encoding = encoding
        self.expectation_values = expectation_values
        self.circuit = circuit


class RoundingResult:
    """Base class for a rounding result"""

    def __init__(self, samples: List[RoundingSolutionSample], expectation_values,*, time_taken=None):
        self._samples = samples
        self._expectation_values = expectation_values
        self.time_taken = time_taken

    @property
    def samples(self) -> List[RoundingSolutionSample]:
        """List of samples"""
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
