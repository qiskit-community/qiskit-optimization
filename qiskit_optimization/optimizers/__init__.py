# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Optimizers (:mod:`qiskit_optimization.optimizers`)
==================================================
Classical Optimizers.

This package contains a variety of classical optimizers and were designed for use by
qiskit_optimization's quantum variational algorithms, such as
:class:`~qiskit_optimization.minimum_eigensolvers.SamplingVQE`.

.. currentmodule:: qiskit_optimization.optimizers

Optimizer Base Classes
----------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Optimizer
   OptimizerResult
   Minimizer

Optimizers
----------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   COBYLA
   NELDER_MEAD
   SPSA
   SciPyOptimizer

"""

from .cobyla import COBYLA
from .nelder_mead import NELDER_MEAD
from .optimizer import Minimizer, Optimizer, OptimizerResult, OptimizerSupportLevel
from .scipy_optimizer import SciPyOptimizer
from .spsa import SPSA

__all__ = [
    "Optimizer",
    "OptimizerSupportLevel",
    "OptimizerResult",
    "Minimizer",
    "SPSA",
    "COBYLA",
    "NELDER_MEAD",
    "SciPyOptimizer",
]
