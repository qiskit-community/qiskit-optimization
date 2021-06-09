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

"""
Quadratic program translators (:mod:`qiskit_optimization.translators`)
======================================================================

.. currentmodule:: qiskit_optimization.translators

Translators between :class:`~qiskit_optimization.problems.QuadraticProgram` and
other optimization models.

Base class for translators
=======================================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QuadraticProgramTranslator

Translators
======================
.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   DocplexMpTranslator
   GurobiTranslator
"""

from .docplex_mp import DocplexMpTranslator
from .gurobi import GurobiTranslator
from .quadratic_program_translator import QuadraticProgramTranslator

_all = ["QuadraticProgramTranslator", "DocplexMpTranslator", "GurobiTranslator"]
