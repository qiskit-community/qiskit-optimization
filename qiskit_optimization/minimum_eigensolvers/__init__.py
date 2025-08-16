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
Minimum Eigensolvers (:mod:`qiskit_optimization.minimum_eigensolvers`)
======================================================================
Algorithms to find the minimum eigenvalue of an operator.

.. currentmodule:: qiskit_optimization.minimum_eigensolvers

Minimum Eigensolvers
--------------------
.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   NumPyMinimumEigensolver
   NumPyMinimumEigensolverResult
   SamplingMinimumEigensolver
   SamplingMinimumEigensolverResult
   SamplingVQE
   SamplingVQEResult
   QAOA
   VQE
   VQEResult
"""

from .numpy_minimum_eigensolver import NumPyMinimumEigensolver, NumPyMinimumEigensolverResult
from .qaoa import QAOA
from .sampling_mes import SamplingMinimumEigensolver, SamplingMinimumEigensolverResult
from .sampling_vqe import SamplingVQE, SamplingVQEResult
from .vqe import VQE, VQEResult

__all__ = [
    "NumPyMinimumEigensolver",
    "NumPyMinimumEigensolverResult",
    "SamplingMinimumEigensolver",
    "SamplingMinimumEigensolverResult",
    "SamplingVQE",
    "SamplingVQEResult",
    "QAOA",
    "VQE",
    "VQEResult",
]
