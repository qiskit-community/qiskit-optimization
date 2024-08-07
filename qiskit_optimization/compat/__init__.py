# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Compatibility module (:mod:`qiskit_optimization.compat`)
=======================================================

Algorithms copied from qiskit-algorithms, which are compatible with Sampler V2.

.. currentmodule:: qiskit_optimization.compat

Algorithms
----------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   SamplingVQE
   QAOA

"""

from .qaoa import QAOA
from .sampling_vqe import SamplingVQE

__all__ = ["SamplingVQE", "QAOA"]
