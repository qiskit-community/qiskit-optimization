# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An abstract class for optimization application classes."""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.result import QuasiDistribution

from qiskit_optimization.algorithms import OptimizationResult
from qiskit_optimization.problems.quadratic_program import QuadraticProgram


class OptimizationApplication(ABC):
    """
    An abstract class for optimization applications.
    """

    @abstractmethod
    def to_quadratic_program(self) -> QuadraticProgram:
        """Convert a problem instance into a
        :class:`~qiskit_optimization.problems.QuadraticProgram`
        """
        pass

    @abstractmethod
    def interpret(self, result: OptimizationResult | np.ndarray):
        """Convert the calculation result of the problem
        (:class:`~qiskit_optimization.algorithms.OptimizationResult` or a binary array using
        np.ndarray) to the answer of the problem in an easy-to-understand format.

        Args:
            result: The calculated result of the problem
        """
        pass

    def _result_to_x(self, result: OptimizationResult | np.ndarray) -> np.ndarray:
        # Return result.x for OptimizationResult and return result itself for np.ndarray
        if isinstance(result, OptimizationResult):
            x = result.x
        elif isinstance(result, np.ndarray):
            x = result
        else:
            raise TypeError(
                "Unsupported format of result. Provide an　OptimizationResult or a",
                f" binary array using np.ndarray instead of {type(result)}",
            )
        return x

    @staticmethod
    def sample_most_likely(
        state_vector: QuasiDistribution | Statevector | np.ndarray | dict,
    ) -> np.ndarray:
        """Compute the most likely binary string from state vector.

        Args:
            state_vector: state vector or counts or quasi-probabilities.

        Returns:
            binary string as numpy.ndarray of ints.

        Raises:
            ValueError: if state_vector is not QuasiDistribution, Statevector,
                np.ndarray, or dict.
        """
        if isinstance(state_vector, QuasiDistribution):
            probabilities = state_vector.binary_probabilities()
            binary_string = max(probabilities.items(), key=lambda kv: kv[1])[0]
            x = np.asarray([int(y) for y in reversed(list(binary_string))])
            return x
        elif isinstance(state_vector, Statevector):
            probabilities = state_vector.probabilities()
            n = state_vector.num_qubits
            k = np.argmax(np.abs(probabilities))
            x = np.zeros(n)
            for i in range(n):
                x[i] = k % 2
                k >>= 1
            return x
        elif isinstance(state_vector, (OrderedDict, dict)):
            # get the binary string with the largest count
            binary_string = max(state_vector.items(), key=lambda kv: kv[1])[0]
            x = np.asarray([int(y) for y in reversed(list(binary_string))])
            return x
        elif isinstance(state_vector, np.ndarray):
            n = int(np.log2(state_vector.shape[0]))
            k = np.argmax(np.abs(state_vector))
            x = np.zeros(n)
            for i in range(n):
                x[i] = k % 2
                k >>= 1
            return x
        else:
            raise ValueError(
                "state vector should be QuasiDistribution, Statevector, ndarray, or dict. "
                f"But it is {type(state_vector)}."
            )
