# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
============================================
Qiskit Algorithms (:mod:`qiskit_algorithms`)
============================================
Qiskit Algorithms is a library of quantum algorithms for quantum computing with
`Qiskit <https://www.ibm.com/quantum/qiskit>`__.
These algorithms can be used to carry out research and investigate how to solve
problems in different domains on simulators and near-term real quantum devices
using shallow circuits.

The library includes some algorithms, for example the :class:`.NumPyMinimumEigensolver`, which take
the same input as their quantum counterpart but solve the problem classically. This has utility in
the near-term, where problems are still tractable classically, to validate and/or act as a reference.
There are also classical :mod:`.optimizers` for use with variational algorithms such as :class:`.VQE`.

This package also provides common building blocks for algorithms, such quantum circuit
gradients (:mod:`.gradients`) and fidelities of quantum states (:mod:`.state_fidelities`).
These elements are frequently used in a variety of applications, such as variational optimization,
time evolution and quantum machine learning.

The quantum algorithms here all use
`Primitives <https://docs.quantum.ibm.com/run/primitives>`__
to execute quantum circuits. This can be an
``Estimator``, which computes expectation values, or a ``Sampler`` which computes
probability distributions. Refer to the specific algorithm for more information in this regard.

.. currentmodule:: qiskit_algorithms

Algorithms
==========

The algorithms now presented are grouped by logical function, such
as minimum eigensolvers, amplitude amplifiers, time evolvers etc. Within each group, the
algorithms conform to an interface that allows them to be used interchangeably
by different applications. E.g. a Qiskit Nature application may take a minimum
eigensolver to solve a ground state problem, and require it to
conform to the :class:`.MinimumEigensolver` interface. Any algorithm that conforms to
the interface, for example :class:`.VQE`, can be used by this application.

Amplitude Amplifiers
--------------------
Algorithms based on amplitude amplification.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   AmplificationProblem
   AmplitudeAmplifier
   Grover
   GroverResult


Amplitude Estimators
--------------------
Algorithms based on amplitude estimation.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   AmplitudeEstimator
   AmplitudeEstimatorResult
   AmplitudeEstimation
   AmplitudeEstimationResult
   EstimationProblem
   FasterAmplitudeEstimation
   FasterAmplitudeEstimationResult
   IterativeAmplitudeEstimation
   IterativeAmplitudeEstimationResult
   MaximumLikelihoodAmplitudeEstimation
   MaximumLikelihoodAmplitudeEstimationResult


Eigensolvers
------------
Algorithms to find eigenvalues of an operator. For chemistry these can be used to find excited
states of a molecule, and ``qiskit-nature`` has some algorithms that leverage chemistry specific
knowledge to do this in that application domain.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Eigensolver
   EigensolverResult
   NumPyEigensolver
   NumPyEigensolverResult
   VQD
   VQDResult


Gradients
---------
Algorithms to calculate the gradient of a quantum circuit.

.. autosummary::
   :toctree:

   gradients


Minimum Eigensolvers
--------------------
Algorithms to find the minimum eigenvalue of an operator.

This set of these algorithms take an ``Estimator`` primitive and can
solve for a general Hamiltonian.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   MinimumEigensolver
   MinimumEigensolverResult
   NumPyMinimumEigensolver
   NumPyMinimumEigensolverResult
   VQE
   VQEResult
   AdaptVQE
   AdaptVQEResult

This set of algorithms take a ``Sampler`` primitive and can only
solve for a diagonal Hamiltonian, such as an Ising Hamiltonian of an optimization problem.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   SamplingMinimumEigensolver
   SamplingMinimumEigensolverResult
   SamplingVQE
   SamplingVQEResult
   QAOA


Optimizers
----------
Classical optimizers designed for use by quantum variational algorithms.

.. autosummary::
   :toctree:

   optimizers


Phase Estimators
----------------
Algorithms that estimate the phases of eigenstates of a unitary.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   HamiltonianPhaseEstimation
   HamiltonianPhaseEstimationResult
   PhaseEstimationScale
   PhaseEstimation
   PhaseEstimationResult
   IterativePhaseEstimation


State Fidelities
----------------
Algorithms that compute the fidelity of pairs of quantum states.

.. autosummary::
   :toctree:

   state_fidelities


Time Evolvers
-------------
Algorithms to evolve quantum states in time. Both real and imaginary time evolution is possible
with algorithms that support them. For machine learning, Quantum Imaginary Time Evolution might be
used to train Quantum Boltzmann Machine Neural Networks for example.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   RealTimeEvolver
   ImaginaryTimeEvolver
   TimeEvolutionResult
   TimeEvolutionProblem
   PVQD
   PVQDResult
   SciPyImaginaryEvolver
   SciPyRealEvolver
   TrotterQRTE
   VarQITE
   VarQRTE
   VarQTEResult

Variational Quantum Time Evolution
++++++++++++++++++++++++++++++++++
Classes used by variational quantum time evolution algorithms -
:class:`.VarQITE` and :class:`.VarQRTE`.

.. autosummary::
   :toctree:

   time_evolvers.variational


Miscellaneous
=============
Various classes used by qiskit-algorithms that are part of and exposed
by the public API.


Exceptions
----------

.. autosummary::
   :toctree:
   :nosignatures:

   AlgorithmError


Utility classes
---------------

Utility classes and function used by algorithms.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   AlgorithmJob

.. autosummary::
   :toctree:

   utils.algorithm_globals

"""
from .algorithm_job import AlgorithmJob
from .algorithm_result import AlgorithmResult
from .variational_algorithm import VariationalAlgorithm, VariationalResult

from .exceptions import AlgorithmError
from .observables_evaluator import estimate_observables

from .minimum_eigensolvers import (
    SamplingMinimumEigensolver,
    SamplingMinimumEigensolverResult,
    MinimumEigensolver,
    MinimumEigensolverResult,
    NumPyMinimumEigensolver,
    NumPyMinimumEigensolverResult,
)

__all__ = [
    "AlgorithmJob",
    "AlgorithmResult",
    "VariationalAlgorithm",
    "VariationalResult",
    "AlgorithmError",
    "estimate_observables",
    "SamplingMinimumEigensolver",
    "SamplingMinimumEigensolverResult",
    "MinimumEigensolver",
    "MinimumEigensolverResult",
    "NumPyMinimumEigensolver",
    "NumPyMinimumEigensolverResult",
]
