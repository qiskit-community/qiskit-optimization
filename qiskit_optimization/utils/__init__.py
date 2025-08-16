# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Utility functions (:mod:`qiskit_optimization.utils`)
====================================================

.. currentmodule:: qiskit_optimization.utils

Utility functions
-----------------
.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   algorithm_globals
"""

from .algorithm_globals import algorithm_globals
from .validate_bounds import validate_bounds
from .validate_initial_point import validate_initial_point

__all__ = [
    "algorithm_globals",
    "validate_initial_point",
    "validate_bounds",
]
