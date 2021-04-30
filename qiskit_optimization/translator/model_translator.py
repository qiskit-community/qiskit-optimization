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
from typing import Generic, TypeVar

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit_optimization.problems.quadratic_program import QuadraticProgram

Model = TypeVar('Model')


class ModelTranslator(ABC, Generic[Model]):
    """Translator between an optimization model and a quadratic program
    """

    @abstractmethod
    def qp_to_model(self, quadratic_program: 'QuadraticProgram') -> Model:
        """Returns an optimization model corresponding to a quadratic program.

        Args:
            quadratic_program: The quadratic program to be translated

        Returns:
            The optimization model corresponding to a quadratic program.
        """
        pass

    @abstractmethod
    def model_to_qp(self, model: Model, quadratic_program: 'QuadraticProgram') -> None:
        """Translate an optimization model into a quadratic program.

        Args:
            model: The optimization model to be loaded.
            quadratic_program: The quadratic program to be stored.
        """
        pass
