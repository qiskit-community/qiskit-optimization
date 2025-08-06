# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

r"""
Quantum Random Access Optimization (:mod:`qiskit_optimization.algorithms.qrao`)
===============================================================================

.. currentmodule:: qiskit_optimization.algorithms.qrao

The Quantum Random Access Optimization (QRAO) module is designed to enable users to leverage a new
quantum method for combinatorial optimization problems [1]. This approach incorporates
Quantum Random Access Codes (QRACs) as a tool to encode multiple classical binary variables into a
single qubit, thereby saving quantum resources and enabling exploration of larger problem instances
on a quantum computer. The encoding produce a local quantum Hamiltonian whose ground state can be
approximated with standard algorithms such as VQE, and then rounded to yield approximation solutions
of the original problem.

QRAO through a series of 3 classes:

* The encoding class (:class:`~.QuantumRandomAccessEncoding`): This class encodes the original
  problem into a relaxed problem that requires fewer resources to solve.
* The rounding schemes (:class:`~.SemideterministicRounding` and :class:`~.MagicRounding`): This
  scheme is used to round the solution obtained from the relaxed problem back to a solution of
  the original problem.
* The optimizer class (:class:`~.QuantumRandomAccessOptimizer`): This class performs the high-level
  optimization algorithm, utilizing the capabilities of the encoding class and the rounding scheme.

:class:`~.QuantumRandomAccessOptimizer` has two methods for solving problems,
:meth:`~.QuantumRandomAccessOptimizer.solve` and
:meth:`~.QuantumRandomAccessOptimizer.solve_relaxed`. The solve method provides a seamless
workflow by automatically managing the encoding and rounding procedures, as demonstrated in the
example below. This allows for a simplified and streamlined user experience.
On the other hand, the solve_relaxed method offers the flexibility to break the computation
process into distinct steps. This feature can be  advantageous when we need to compare solutions
obtained from different rounding schemes applied to a potential ground state.


For example:

.. code-block:: python

    from qiskit_optimization.optimizers import COBYLA
    from qiskit_optimization.minimum_eigensolvers import VQE
    from qiskit.circuit.library import RealAmplitudes
    from qiskit.primitives import Estimator

    from qiskit_optimization.algorithms.qrao import (
        QuantumRandomAccessOptimizer,
        QuantumRandomAccessEncoding,
        SemideterministicRounding,
    )
    from qiskit_optimization.problems import QuadraticProgram

    problem = QuadraticProgram()
    problem.binary_var("x")
    problem.binary_var("y")
    problem.binary_var("z")
    problem.minimize(linear={"x": 1, "y": 2, "z": 3})

    ansatz = RealAmplitudes(1)
    vqe = VQE(
        ansatz=ansatz,
        optimizer=COBYLA(),
        estimator=Estimator(),
    )
    # solve() automatically performs the encoding, optimization, and rounding
    qrao = QuantumRandomAccessOptimizer(min_eigen_solver=vqe)
    result = qrao.solve(problem)

    # solve_relaxed() only performs the optimization. The encoding and rounding must be done manually.
    # encoding
    encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=3)
    encoding.encode(problem)
    # optimization
    qrao = QuantumRandomAccessOptimizer(min_eigen_solver=vqe)
    relaxed_results, rounding_context = qrao.solve_relaxed(encoding=encoding)
    # rounding
    rounding = SemideterministicRounding()
    result = rounding.round(rounding_context)


[1] Bryce Fuller et al., Approximate Solutions of Combinatorial Problems via Quantum Relaxations,
`arXiv:2111.03167 <https://arxiv.org/abs/2111.03167>`_


Quantum Random Access Encoding and Optimization
-----------------------------------------------
.. autosummary::
    :toctree: ../stubs/
    :nosignatures:

    EncodingCommutationVerifier
    QuantumRandomAccessEncoding
    QuantumRandomAccessOptimizer
    QuantumRandomAccessOptimizationResult

Rounding schemes
----------------

.. autosummary::
    :toctree: ../stubs/
    :nosignatures:

    MagicRounding
    RoundingScheme
    RoundingContext
    RoundingResult
    SemideterministicRounding

"""

from .encoding_commutation_verifier import EncodingCommutationVerifier
from .magic_rounding import MagicRounding
from .quantum_random_access_encoding import QuantumRandomAccessEncoding
from .quantum_random_access_optimizer import (
    QuantumRandomAccessOptimizationResult,
    QuantumRandomAccessOptimizer,
    SemideterministicRounding,
)
from .rounding_common import RoundingContext, RoundingResult, RoundingScheme

__all__ = [
    "EncodingCommutationVerifier",
    "QuantumRandomAccessEncoding",
    "RoundingScheme",
    "RoundingContext",
    "RoundingResult",
    "SemideterministicRounding",
    "MagicRounding",
    "QuantumRandomAccessOptimizer",
    "QuantumRandomAccessOptimizationResult",
]
