Qiskit Optimization v0.7 Migration Guide
=========================================

This tutorial will guide you through the process of migrating your code from Qiskit Optimization v0.6 to v0.7,
which includes two major changes:

1. **Migration from qiskit-algorithms dependency**: Essential components have been copied into qiskit-optimization
2. **Migration from V1 to V2 Primitives**: V1 Primitives usage is deprecated in favor of V2 Primitives

Overview
--------

Qiskit Optimization v0.7 introduces significant changes to improve independence and performance:

**qiskit-algorithms Independence:**

- Removed dependency on ``qiskit-algorithms`` package
- Migrated essential minimum eigensolvers, optimizers, and utilities directly into qiskit-optimization

**V2 Primitives Support:**
Qiskit Optimization v0.7 introduces support for Qiskit's V2 Primitives (``BaseEstimatorV2`` and ``BaseSamplerV2``)
while deprecating support for V1 Primitives (``BaseEstimatorV1`` and ``BaseSamplerV1``). This migration
guide provides comprehensive examples for updating your code to use the new V2 Primitives interface.

The main differences between V1 and V2 Primitives are:

- **Different input/output formats**: V2 Primitives use a different interface for submitting jobs and retrieving results
- **Pass manager requirement**: V2 Primitives (except ``StatevectorEstimator`` and ``StatevectorSampler``) require explicit pass managers for circuit transpilation

Key Changes
-----------

V1 Primitive usage is deprecated as of Qiskit Optimization v0.7.0 and will be removed in a future release.
All algorithms now support both V1 and V2 Primitives with automatic detection and appropriate warnings.

Migration from qiskit-algorithms to qiskit-optimization
-------------------------------------------------------

As part of Qiskit Optimization v0.7, the dependency on ``qiskit-algorithms`` has been removed.
Essential components previously imported from ``qiskit-algorithms`` have been migrated directly
into ``qiskit-optimization``. This section covers the key migration patterns.

Optimizers Migration
~~~~~~~~~~~~~~~~~~~~

**Before (with qiskit-algorithms dependency):**

.. code-block:: python

    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_algorithms.utils import algorithm_globals

    optimizer = COBYLA()
    algorithm_globals.random_seed = 42

**After (qiskit-optimization v0.7):**

.. code-block:: python

    from qiskit_optimization.optimizers import COBYLA
    from qiskit_optimization.utils import algorithm_globals

    optimizer = COBYLA()
    algorithm_globals.random_seed = 42

**Available migrated optimizers:**

- ``COBYLA``
- ``NELDER_MEAD``
- ``SciPyOptimizer``
- ``SPSA``

Algorithm Globals Migration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Before:**

.. code-block:: python

    from qiskit_algorithms.utils import algorithm_globals

    algorithm_globals.random_seed = 42

**After:**

.. code-block:: python

    from qiskit_optimization.utils import algorithm_globals

    algorithm_globals.random_seed = 42

Minimum Eigensolvers
~~~~~~~~~~~~~~~~~~~~

**Before:**

.. code-block:: python

    from qiskit_algorithms import QAOA
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

**After:**

.. code-block:: python

    from qiskit_optimization.minimum_eigensolvers import QAOA
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

**Available migrated minimum eigensolvers:**

- ``SamplingVQE``
- ``QAOA``
- ``VQE``
- ``NumPyMinimumEigensolver``

V2 Primitives Migration
-----------------------

VQE with EstimatorV2
~~~~~~~~~~~~~~~~~~~~

**V1 Primitive (Deprecated):**

.. code-block:: python

    from qiskit.circuit.library import RealAmplitudes
    from qiskit.primitives import Estimator
    from qiskit_optimization.minimum_eigensolvers import VQE
    from qiskit_optimization.optimizers import COBYLA

    # V1 Estimator - deprecated
    estimator = Estimator(seed=123, shots=1000)
    ansatz = RealAmplitudes(num_qubits=2, reps=1)
    optimizer = COBYLA()

    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer
    )

**V2 Primitive (Recommended):**

.. code-block:: python

    from qiskit import generate_preset_pass_manager
    from qiskit.circuit.library import RealAmplitudes
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import EstimatorV2
    from qiskit_optimization.minimum_eigensolvers import VQE
    from qiskit_optimization.optimizers import COBYLA

    # V2 Estimator with pass_manager (for hardware/simulators)
    backend = AerSimulator()
    pass_manager = generate_preset_pass_manager(
        optimization_level=2,
        target=backend.target
    )
    estimator = EstimatorV2(options={"default_precision": 0.01, "backend_options": {"seed_simulator": 123}})
    ansatz = RealAmplitudes(num_qubits=2, reps=1)
    optimizer = COBYLA()

    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        pass_manager=pass_manager  # Required for V2 Primitives (except StatevectorEstimator)
    )

QAOA with SamplerV2
~~~~~~~~~~~~~~~~~~~

**V1 Primitive (Deprecated):**

.. code-block:: python

    from qiskit.primitives import Sampler
    from qiskit_optimization.minimum_eigensolvers import QAOA
    from qiskit_optimization.optimizers import COBYLA

    # V1 Sampler - deprecated
    sampler = Sampler(seed=123, shots=1000)
    optimizer = COBYLA()

    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=1
    )

**V2 Primitive (Recommended):**

.. code-block:: python

    from qiskit import generate_preset_pass_manager
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2
    from qiskit_optimization.minimum_eigensolvers import QAOA
    from qiskit_optimization.optimizers import COBYLA

    # V2 Sampler with pass_manager (for hardware/simulators)
    backend = AerSimulator()
    pass_manager = generate_preset_pass_manager(
        optimization_level=2,
        backend=backend
    )
    sampler = SamplerV2(seed=123, default_shots=1000)
    optimizer = COBYLA()

    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=1,
        pass_manager=pass_manager  # Required for V2 Primitives (except StatevectorSampler)
    )

SamplingVQE Migration
~~~~~~~~~~~~~~~~~~~~~

**V1 Primitive (Deprecated):**

.. code-block:: python

    from qiskit.circuit.library import RealAmplitudes
    from qiskit.primitives import Sampler
    from qiskit_optimization.minimum_eigensolvers import SamplingVQE
    from qiskit_optimization.optimizers import COBYLA

    # V1 Sampler - deprecated
    sampler = Sampler(seed=123, shots=1000)
    ansatz = RealAmplitudes(num_qubits=2, reps=1)
    optimizer = COBYLA()

    sampling_vqe = SamplingVQE(
        sampler=sampler,
        ansatz=ansatz,
        optimizer=optimizer
    )

**V2 Primitive (Recommended):**

.. code-block:: python

    from qiskit import generate_preset_pass_manager
    from qiskit.circuit.library import RealAmplitudes
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2
    from qiskit_optimization.minimum_eigensolvers import SamplingVQE
    from qiskit_optimization.optimizers import COBYLA

    # V2 Sampler with pass_manager (for hardware/simulators)
    backend = AerSimulator()
    pass_manager = generate_preset_pass_manager(
        optimization_level=2,
        backend=backend
    )
    sampler = SamplerV2(seed=123, default_shots=1000)
    ansatz = RealAmplitudes(num_qubits=2, reps=1)
    optimizer = COBYLA()

    sampling_vqe = SamplingVQE(
        sampler=sampler,
        ansatz=ansatz,
        optimizer=optimizer,
        pass_manager=pass_manager  # Required for V2 Primitives (except StatevectorSampler)
    )

QRAO (Quantum Random Access Optimization) Migration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``MagicRounding`` requires a Sampler and must be updated to use V2 Primitives.
``SemideterministicRounding`` does not require a Sampler and does not need changes.

**V1 Primitive (Deprecated):**

.. code-block:: python

    from qiskit.circuit.library import RealAmplitudes
    from qiskit.primitives import Estimator, Sampler
    from qiskit_optimization.algorithms.qrao import (
        MagicRounding,
        QuantumRandomAccessOptimizer
    )
    from qiskit_optimization.minimum_eigensolvers import VQE
    from qiskit_optimization.optimizers import COBYLA

    # V1 Primitives - deprecated
    estimator = Estimator(seed=123, shots=10000)
    sampler = Sampler(seed=123, shots=10000)

    ansatz = RealAmplitudes(1)
    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=COBYLA())
    magic_rounding = MagicRounding(sampler=sampler)

    qrao = QuantumRandomAccessOptimizer(
        min_eigen_solver=vqe,
        rounding_scheme=magic_rounding
    )

**V2 Primitive (Recommended):**

.. code-block:: python

    from qiskit import generate_preset_pass_manager
    from qiskit.circuit.library import RealAmplitudes
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import EstimatorV2, SamplerV2
    from qiskit_optimization.algorithms.qrao import (
        MagicRounding,
        QuantumRandomAccessOptimizer
    )
    from qiskit_optimization.minimum_eigensolvers import VQE
    from qiskit_optimization.optimizers import COBYLA

    # V2 Primitives with pass_manager (for hardware/simulators)
    backend = AerSimulator()
    pass_manager = generate_preset_pass_manager(
        optimization_level=2,
        backend=backend
    )

    estimator = EstimatorV2(options={"default_precision": 0.01, "backend_options": {"seed_simulator": 123}})
    sampler = SamplerV2(seed=123, default_shots=10000)
    ansatz = RealAmplitudes(1)

    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=COBYLA(),
        pass_manager=pass_manager
    )
    magic_rounding = MagicRounding(
        sampler=sampler,
        pass_manager=pass_manager
    )

    qrao = QuantumRandomAccessOptimizer(
        min_eigen_solver=vqe,
        rounding_scheme=magic_rounding
    )

Grover Optimizer with SamplerV2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**V1 Primitive (Deprecated):**

.. code-block:: python

    from qiskit.primitives import Sampler
    from qiskit_optimization.algorithms import GroverOptimizer

    # V1 Sampler - deprecated
    sampler = Sampler(seed=123, shots=1000)

    grover_optimizer = GroverOptimizer(
        num_value_qubits=3,
        num_iterations=3,
        sampler=sampler
    )

**V2 Primitive (Recommended):**

.. code-block:: python

    from qiskit import generate_preset_pass_manager
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2
    from qiskit_optimization.algorithms import GroverOptimizer

    # V2 Sampler with pass_manager (for hardware/simulators)
    backend = AerSimulator()
    pass_manager = generate_preset_pass_manager(
        optimization_level=2,
        backend=backend
    )
    sampler = SamplerV2(seed=123, default_shots=1000)

    grover_optimizer = GroverOptimizer(
        num_value_qubits=3,
        num_iterations=3,
        sampler=sampler,
        pass_manager=pass_manager  # Required for V2 Primitives (except StatevectorSampler)
    )

Migration Example: Complete Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Before (v0.6 with qiskit-algorithms):**

.. code-block:: python

    from qiskit.primitives import Sampler  # V1 Primitive
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_algorithms.utils import algorithm_globals
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_optimization.minimum_eigensolvers import QAOA

    # Set global random seed
    algorithm_globals.random_seed = 42

    # Create problem
    problem = QuadraticProgram()
    problem.binary_var("x")
    problem.binary_var("y")
    problem.minimize(linear={"x": 1, "y": 2})

    # Create QAOA with V1 Primitives and qiskit-algorithms optimizer
    sampler = Sampler(seed=42, shots=1000)
    optimizer = COBYLA()
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=1)

    # Solve
    meo = MinimumEigenOptimizer(qaoa)
    result = meo.solve(problem)

**After (v0.7 with internal components and V2 Primitives):**

.. code-block:: python

    from qiskit import generate_preset_pass_manager
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2  # V2 Primitive
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_optimization.minimum_eigensolvers import QAOA
    from qiskit_optimization.optimizers import COBYLA  # Now internal
    from qiskit_optimization.utils import algorithm_globals  # Now internal

    # Set global random seed - same API
    algorithm_globals.random_seed = 42

    # Create problem - unchanged
    problem = QuadraticProgram()
    problem.binary_var("x")
    problem.binary_var("y")
    problem.minimize(linear={"x": 1, "y": 2})

    # Create QAOA with V2 Primitives, internal optimizer, and pass_manager
    backend = AerSimulator()
    pass_manager = generate_preset_pass_manager(
        optimization_level=2,
        backend=backend,
        seed_transpiler=42
    )
    sampler = SamplerV2(seed=42, default_shots=1000)
    optimizer = COBYLA()  # Same class, now from qiskit_optimization

    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=1,
        pass_manager=pass_manager  # Required for V2 Primitives (except StatevectorSampler)
    )

    # Solve - unchanged
    meo = MinimumEigenOptimizer(qaoa)
    result = meo.solve(problem)

Migration Checklist
-------------------

To migrate your code to Qiskit Optimization v0.7:

**qiskit-algorithms Migration:**

☐ **Update optimizer imports**:
   - ``from qiskit_algorithms.optimizers import COBYLA`` → ``from qiskit_optimization.optimizers import COBYLA``
   - ``from qiskit_algorithms.optimizers import NELDER_MEAD`` → ``from qiskit_optimization.optimizers import NELDER_MEAD``
   - ``from qiskit_algorithms.optimizers import SciPyOptimizer`` → ``from qiskit_optimization.optimizers import SciPyOptimizer``
   - ``from qiskit_algorithms.optimizers import SPSA`` → ``from qiskit_optimization.optimizers import SPSA``

☐ **Update minimum_eigensolver imports**:
   - ``from qiskit_algorithms import QAOA`` → ``from qiskit_optimization.minimum_eigensolvers import QAOA``
   - ``from qiskit_algorithms import VQE`` → ``from qiskit_optimization.minimum_eigensolvers import VQE``
   - ``from qiskit_algorithms import SamplingVQE`` → ``from qiskit_optimization.minimum_eigensolvers import SamplingVQE``
   - ``from qiskit_algorithms import NumPyMinimumEigensolver`` → ``from qiskit_optimization.minimum_eigensolvers import NumPyMinimumEigensolver``

☐ **Update algorithm_globals import**:
   - ``from qiskit_algorithms.utils import algorithm_globals`` → ``from qiskit_optimization.utils import algorithm_globals``

☐ **Remove qiskit-algorithms dependency** (if only used for the above components):
   - Remove from ``requirements.txt`` or ``pyproject.toml`` dependencies

**V2 Primitives Migration:**

☐ **Replace primitive imports for simulators and hardware**:
   - For simulators: ``from qiskit.primitives import Estimator`` → ``from qiskit.primitives import StatevectorEstimator`` or ``from qiskit_aer.primitives import EstimatorV2``
   - For simulators: ``from qiskit.primitives import Sampler`` → ``from qiskit.primitives import StatevectorSampler`` or ``from qiskit_aer.primitives import SamplerV2``
   - For hardware: use the appropriate provider's primitives (e.g., ``from qiskit_ibm_runtime import EstimatorV2, SamplerV2`` for IBM Quantum hardware)

☐ **Update primitive initialization**:
   - Add ``default_shots`` parameter for V2 Sampler
   - Add ``default_precision`` or ``default_shots`` parameter for V2 Estimator
       - Refer to `Introduction to options <https://quantum.cloud.ibm.com/docs/en/guides/runtime-options-overview>`_ for details of options
   - Use ``seed`` parameter for SamplerV2 of Qiskit Aer (to ensure deterministic results for simulations)
   - Use ``options={"backend_options": {"seed_simulator": seed}}`` for EstimatorV2 of Qiskit Aer (to ensure deterministic results for simulations)

☐ **Add pass manager** (if not using Statevector primitives):
   - Import: ``from qiskit import generate_preset_pass_manager`` or ``from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager``
   - Create: ``pass_manager = generate_preset_pass_manager(optimization_level=2, backend=backend)``
   - Pass to algorithm constructor: ``pass_manager=pass_manager``

☐ **Test your code** to ensure results remain consistent

☐ **Remove any V1-specific result access patterns** if you have custom result processing

By following this migration guide, you'll successfully transition your Qiskit Optimization code to v0.7,
eliminating the qiskit-algorithms dependency and adopting the V2 Primitives interface for future compatibility.
