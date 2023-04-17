# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Quantum Random Access Optimization (:mod:`qiskit_optimization.algorithms.qrao`)
===============================================================================

.. currentmodule:: qiskit_optimization.algorithms.qrao


Quantum Random Access Encoding and Optimization
===============================================
.. autosummary::
    :toctree: ../stubs/
    :nosignatures:

    EncodingCommutationVerifier
    QuantumRandomAccessEncoding
    QuantumRandomAccessOptimizer
    QuantumRandomAccessOptimizationResult

Rounding schemes
================

.. autosummary::
    :toctree: ../stubs/
    :nosignatures:

    MagicRounding
    MagicRoundingResult
    RoundingScheme
    RoundingContext
    RoundingResult
    SemideterministicRounding
    SemideterministicRoundingResult

"""

from .encoding_commutation_verifier import EncodingCommutationVerifier
from .quantum_random_access_encoding import QuantumRandomAccessEncoding
from .magic_rounding import MagicRounding, MagicRoundingResult
from .quantum_random_access_optimizer import (
    QuantumRandomAccessOptimizationResult,
    QuantumRandomAccessOptimizer,
)
from .rounding_common import RoundingContext, RoundingResult, RoundingScheme
from .semideterministic_rounding import SemideterministicRounding, SemideterministicRoundingResult

__all__ = [
    "EncodingCommutationVerifier",
    "QuantumRandomAccessEncoding",
    "RoundingScheme",
    "RoundingContext",
    "RoundingResult",
    "SemideterministicRounding",
    "SemideterministicRoundingResult",
    "MagicRounding",
    "MagicRoundingResult",
    "QuantumRandomAccessOptimizer",
    "QuantumRandomAccessOptimizationResult",
]
