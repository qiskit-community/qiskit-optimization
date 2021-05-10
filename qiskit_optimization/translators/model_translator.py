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
from typing import Any


class ModelTranslator(ABC):
    """Translator between an optimization model and a quadratic program

    Note:
        The types of quadratic_program is `Any` because we need to avoid cyclic import.
    """

    @abstractmethod
    def is_compatible(self, model: Any) -> bool:
        """Checks whether a given model can be translated with this translator.

        Args:
            model: The optimization model to check compatibility.

        Returns:
            Returns True if the model is compatible, False otherwise.
        """
        pass

    @abstractmethod
    def qp_to_model(self, quadratic_program: Any) -> Any:
        """Returns an optimization model corresponding to a quadratic program.

        Args:
            quadratic_program: The quadratic program to be translated

        Returns:
            The optimization model corresponding to a quadratic program.
        """
        pass

    @abstractmethod
    def model_to_qp(self, model: Any) -> Any:
        """Translate an optimization model into a quadratic program.

        Args:
            model: The optimization model to be loaded.

        Returns:
            The quadratic program corresponding to the model.
        """
        pass
