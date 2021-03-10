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
from typing import Union
from abc import ABC, abstractmethod

import numpy as np

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
            raise TypeError("Unsupported format of result. Provide an　OptimizationResult or a",
                            "binary array using np.ndarray instead of {}".format(type(result)))
        return x
