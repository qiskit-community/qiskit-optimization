# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An abstract class for optimization application classes."""
from typing import Union, Dict
from collections import OrderedDict
from abc import ABC, abstractmethod

import numpy as np

from qiskit.opflow import StateFn
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
    def interpret(self, result: Union[OptimizationResult, np.ndarray]):
        """Convert the calculation result of the problem
        (:class:`~qiskit_optimization.algorithms.OptimizationResult` or a binary array using
        np.ndarray) to the answer of the problem in an easy-to-understand format.

        Args:
            result: The calculated result of the problem
        """
        pass

    def _result_to_x(self, result: Union[OptimizationResult, np.ndarray]) -> np.ndarray:
        # Return result.x for OptimizationResult and return result itself for np.ndarray
        if isinstance(result, OptimizationResult):
            x = result.x
        elif isinstance(result, np.ndarray):
            x = result
        else:
            raise TypeError(
                "Unsupported format of result. Provide an　OptimizationResult or a",
                "binary array using np.ndarray instead of {}".format(type(result)),
            )
        return x

    @staticmethod
    def sample_most_likely(state_vector: Union[np.ndarray, Dict]) -> np.ndarray:
        """Compute the most likely binary string from state vector.

        Args:
            state_vector: state vector or counts.

        Returns:
            binary string as numpy.ndarray of ints.
        """
        if isinstance(state_vector, (OrderedDict, dict)):
            # get the binary string with the largest count
            binary_string = sorted(state_vector.items(), key=lambda kv: kv[1])[-1][0]
            x = np.asarray([int(y) for y in reversed(list(binary_string))])
            return x
        elif isinstance(state_vector, StateFn):
            binary_string = list(state_vector.sample().keys())[0]
            x = np.asarray([int(y) for y in reversed(list(binary_string))])
            return x
        else:
            n = int(np.log2(state_vector.shape[0]))
            k = np.argmax(np.abs(state_vector))
            x = np.zeros(n)
            for i in range(n):
                x[i] = k % 2
                k >>= 1
            return x
