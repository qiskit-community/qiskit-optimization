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
other objects such as optimization models and files.
Translators allow users to load a quadratic program from an external source
such as optimization models and files.
They also allow users to export a quadratic program to optimization models
and save a quadratic program to a file in a particular format (e.g., LP format).

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
   LPFileTranslator
"""

from .docplex_mp import DocplexMpTranslator
from .gurobi import GurobiTranslator
from .lp_file import LPFileTranslator
from .quadratic_program_translator import QuadraticProgramTranslator

_all = ["ModelTranslator", "DocplexMpTranslator", "GurobiTranslator", "LPFileTranslator"]
