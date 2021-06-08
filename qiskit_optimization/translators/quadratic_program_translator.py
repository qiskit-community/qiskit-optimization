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

"""Abstract class for quadratic program translators"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # pylint: disable=cyclic-import
    from qiskit_optimization.problems.quadratic_program import QuadraticProgram


class QuadraticProgramTranslator(ABC):
    """Translator between a quadratic program and other objects.

    Translators allow users to load a quadratic program from an external source
    such as optimization models and files.
    They also allow users to export a quadratic program to optimization models
    and save a quadratic program to a file in a particular format (e.g., LP format).
    """

    @classmethod
    @abstractmethod
    def is_installed(cls) -> bool:
        """Checks whether the dependent module is installed or not.

        Returns:
            Returns ``True`` if necessary modules are installed, ``False`` otherwise.
        """
        pass

    @classmethod
    @abstractmethod
    def is_compatible(cls, source: Any) -> bool:
        """Checks whether the supplied source is supported by translator.

        Args:
            source: The external source to be translated into ``QuadraticProgram``.

        Returns:
            Returns True if the model is compatible, False otherwise.
        """
        pass

    @abstractmethod
    def from_qp(self, quadratic_program: "QuadraticProgram") -> Any:
        """Translates a quadratic program into the target object.

        Args:
            quadratic_program: The quadratic program to be translated.

        Returns:
            The target object corresponding to a quadratic program.
        """
        pass

    @abstractmethod
    def to_qp(self, source: Any) -> "QuadraticProgram":
        """Translates an external source into a quadratic program.

        Args:
            source: The external source to be translated into ``QuadraticProgram``.

        Returns:
            The quadratic program corresponding to the source.
        """
        pass
