# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2020, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An abstract class for optimization algorithms in Qiskit optimization module."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..problems.quadratic_program import QuadraticProgram


class QuadraticProgramConverter(ABC):
    """
    An abstract class for converters of quadratic programs in Qiskit optimization module.
    """

    @abstractmethod
    def convert(self, problem: QuadraticProgram) -> QuadraticProgram:
        """Convert a QuadraticProgram into another form
        and keep the information required to interpret the result.
        """
        raise NotImplementedError

    @abstractmethod
    def interpret(self, x: np.ndarray | list[float]) -> np.ndarray:
        """Interpret a result into another form using the information of conversion"""
        raise NotImplementedError
