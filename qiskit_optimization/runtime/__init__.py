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
Qiskit Optimization Runtime (:mod:`qiskit_optimization.runtime`)
================================================================

.. currentmodule:: qiskit_optimization.runtime

Programs that embed Qiskit Runtime in the algorithmic interfaces and facilitate usage of
algorithms and scripts in the cloud.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   VQEProgram
   VQEProgramResult
   QAOAProgram

"""

from .vqe_program import VQEProgram, VQEProgramResult
from .qaoa_program import QAOAProgram

__all__ = ["VQEProgram", "VQEProgramResult", "QAOAProgram"]
