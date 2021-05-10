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

"""Utilities for optimization model translators"""

from typing import Any

from qiskit_optimization.exceptions import QiskitOptimizationError
from .docplex import DocplexMpTranslator

translators = [DocplexMpTranslator()]


def load_model(model: Any) -> Any:
    """Returns a quadratic program corresponding to the model.

    Args:
        model: The optimization model to be translated

    Returns:
        The quadratic program corresponding to the model.

    Raises:
        QiskitOptimizationError: if no model translator can handle the model.

    Note:
        The return type is `Any` not `QuadraticProgram` because we need to avoid cyclic import.

    """
    for trans in translators:
        if trans.is_compatible(model):
            return trans.model_to_qp(model)
    raise QiskitOptimizationError(
        "There is no compatible translator to this model: {}".format(type(model))
    )
