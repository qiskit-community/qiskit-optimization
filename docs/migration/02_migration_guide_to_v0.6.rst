Qiskit Optimization v0.6 Migration Guide
========================================

This tutorial will guide you through the process of migrating your code
from Qiskit Optimization v0.5 to v0.6.

Overview
--------

Qiskit Terra v0.25 deprecated the ``qiskit.algorithms`` module. It has been
superseded by a new standalone library
`Qiskit Algorithms <https://github.com/qiskit-community/qiskit_algorithms>`__.

Qiskit Optimization v0.6 supports only the new algorithms of Qiskit Algorithms.

It is not the intention to provide detailed explanations of the
new Qiskit Algorithms in this migration guide. We suggest that you read the
`corresponding
resources <https://qiskit-community.github.io/qiskit-algorithms/index.html>`__
of the Qiskit Algorithms documentation instead.

We can basically use the existing codes by replacing ``qiskit.algorithms``
with ``qiskit_algorithms``.


``MinimumEigenOptimizer``
-------------------------

The former algorithms exist in
``qiskit.algorithms.minimum_eigensolvers``.
On the other hand, the new algorithms exist in
``qiskit_algorithms.minimum_eigensolvers`` and we can access them by
``qiskit_algorithms.*``.

``MinimumEigenOptimizer`` of Qiskit Optimization can use
``qiskit_algorithms.SamplingMinimumEigensolver``
interface of the new algorithms. Note that ``MinimumEigenOptimizer``
cannot basically handle
``qiskit_algorithms.MinimumEigensolver`` of the new
algorithms. But there is an exception. ``MinimumEigenOptimizer`` can
handle ``qiskit_algorithms.NumPyMinimumEigensolver``
because ``qiskit_algorithms.NumPyMinimumEigensolver`` has
an extension that allows users to access the eigen states.

The following is the corresponding table.

.. csv-table::
    :header: Former algorithm, New algorithm

    ``qiskit.algorithms.minimum_eigensolvers.SamplingMinimumEigensolver``, ``qiskit_algorithms.SamplingMinimumEigensolver``
    ``qiskit.algorithms.minimum_eigensolver.NumPyMinimumEigensolver``, ``qiskit_algorithms.NumPyMinimumEigensolver``
    ``qiskit.algorithms.minimum_eigensolvers.QAOA``, ``qiskit_algorithms.QAOA``
    ``qiskit.algorithms.minimum_eigensolvers.SamplingVQE``, ``qiskit_algorithms.SamplingVQE``



NumPyMinimumEigensolver
~~~~~~~~~~~~~~~~~~~~~~~

Previously

.. code:: python

    from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver

    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    mes = NumPyMinimumEigensolver()
    meo = MinimumEigenOptimizer(min_eigen_solver=mes)
    result = meo.solve(problem)
    print(result)


New

.. code:: python

    from qiskit_algorithms import NumPyMinimumEigensolver

    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    mes = NumPyMinimumEigensolver()
    meo = MinimumEigenOptimizer(min_eigen_solver=mes)
    result = meo.solve(problem)
    print(result)



QAOA
~~~~

Previously

.. code:: python

    from qiskit.algorithms.minimum_eigensolvers import QAOA
    from qiskit.algorithms.optimizers import COBYLA
    from qiskit.primitives import Sampler

    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    shots = 1000
    mes = QAOA(sampler=Sampler(), optimizer=COBYLA())
    meo = MinimumEigenOptimizer(min_eigen_solver=mes)
    result = meo.solve(problem)
    print(result)


New

.. code:: python

    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit.primitives import Sampler

    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    shots = 1000
    mes = QAOA(sampler=Sampler(), optimizer=COBYLA())
    meo = MinimumEigenOptimizer(min_eigen_solver=mes)
    result = meo.solve(problem)
    print(result)



SamplingVQE
~~~~~~~~~~~

Previously

.. code:: python

    from qiskit.algorithms.minimum_eigensolvers import SamplingVQE
    from qiskit.algorithms.optimizers import COBYLA
    from qiskit.circuit.library import RealAmplitudes
    from qiskit.primitives import Sampler

    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    mes = SamplingVQE(sampler=Sampler(), ansatz=RealAmplitudes(), optimizer=COBYLA())
    meo = MinimumEigenOptimizer(min_eigen_solver=mes)
    result = meo.solve(problem)
    print(result)


New

.. code:: python

    from qiskit_algorithms import SamplingVQE
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit.circuit.library import RealAmplitudes
    from qiskit.primitives import Sampler

    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    mes = SamplingVQE(sampler=Sampler(), ansatz=RealAmplitudes(), optimizer=COBYLA())
    meo = MinimumEigenOptimizer(min_eigen_solver=mes)
    result = meo.solve(problem)
    print(result)



``WarmStartQAOAOptimizer``
--------------------------


Previously

.. code:: python

    from qiskit.algorithms.minimum_eigensolvers import QAOA
    from qiskit.algorithms.optimizers import COBYLA
    from qiskit.primitives import Sampler

    from qiskit_optimization.algorithms import WarmStartQAOAOptimizer, SlsqpOptimizer

    qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA())
    optimizer = WarmStartQAOAOptimizer(
        pre_solver=SlsqpOptimizer(), relax_for_pre_solver=True, qaoa=qaoa, epsilon=0.25
    )
    result = optimizer.solve(problem)
    print(result)


New

.. code:: python

    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit.primitives import Sampler

    from qiskit_optimization.algorithms import WarmStartQAOAOptimizer, SlsqpOptimizer

    qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA())
    optimizer = WarmStartQAOAOptimizer(
        pre_solver=SlsqpOptimizer(), relax_for_pre_solver=True, qaoa=qaoa, epsilon=0.25
    )
    result = optimizer.solve(problem)
    print(result)



``GroverOptimizer``
-------------------


Previously

.. code:: python

    from qiskit.algorithms.optimizers import COBYLA
    from qiskit.primitives import Sampler

    from qiskit_optimization.algorithms import GroverOptimizer

    optimizer = GroverOptimizer(num_value_qubits=3, num_iterations=3, sampler=Sampler())
    result = optimizer.solve(problem)
    print(result)


New

.. code:: python

    from qiskit_algorithms.optimizers import COBYLA
    from qiskit.primitives import Sampler

    from qiskit_optimization.algorithms import GroverOptimizer

    optimizer = GroverOptimizer(num_value_qubits=3, num_iterations=3, sampler=Sampler())
    result = optimizer.solve(problem)
    print(result)
