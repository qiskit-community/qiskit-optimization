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
Optimization model translators (:mod:`qiskit_optimization.translators`)
===================================================================

.. currentmodule:: qiskit_optimization.translators

Translators between an optimization model and a quadratic program

Base classes for applications
=======================================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ModelTranslator
   DocplexTranslator
   GurobiTranslator

Applications
======================
.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   DocplexTranslator
   GurobiTranslator
"""

from .model_translator import ModelTranslator
from .docplex import DocplexMpTranslator
from .utils import load_model

_all = ["ModelTranslator", "DocplexTranslator", "load_model"]
