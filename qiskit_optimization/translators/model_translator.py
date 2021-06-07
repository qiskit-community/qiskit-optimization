# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Abstract class for optimization model translators"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # pylint: disable=cyclic-import
    from qiskit_optimization.problems.quadratic_program import QuadraticProgram


class ModelTranslator(ABC):
    """Translator between a quadratic program and another object."""

    @abstractmethod
    def is_installed(self) -> bool:
        """Checks whether the dependent module for this translator is installed or not.

        Returns:
            Returns True if necessary modules are installed, False otherwise.
        """
        pass

    @abstractmethod
    def is_compatible(self, source: Any) -> bool:
        """Checks whether a source can be translated with this translator.

        Args:
            source: The external source to be translated into ``QuadraticProgram``.

        Returns:
            Returns True if the model is compatible, False otherwise.
        """
        pass

    @abstractmethod
    def from_qp(self, quadratic_program: "QuadraticProgram") -> Any:
        """Returns an optimization model corresponding to a quadratic program.

        Args:
            quadratic_program: The quadratic program to be translated.

        Returns:
            The optimization model corresponding to a quadratic program.
        """
        pass

    @abstractmethod
    def to_qp(self, source: Any) -> "QuadraticProgram":
        """Translate an optimization model into a quadratic program.

        Args:
            source: The external source to be translated into ``QuadraticProgram``.

        Returns:
            The quadratic program corresponding to the model.
        """
        pass
