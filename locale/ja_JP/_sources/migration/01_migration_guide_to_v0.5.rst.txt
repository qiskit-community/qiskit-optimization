Qiskit Optimization v0.5 Migration Guide
========================================

This tutorial will guide you through the process of migrating your code
from Qiskit Optimization v0.4 to v0.5.

Overview
--------

Qiskit Terra v0.22 introduces new algorithm implementations that
leverage `Qiskit
Primitives <https://qiskit.org/documentation/apidoc/primitives.html>`__
(Estimator and Sampler). The former algorithm implementations that
leverage opflow will be deprecated in the future release.

Qiskit Optimization v0.5 supports both the new and the former algorithms
of Qiskit Terra v0.22 until the former algorithms are deprecated.

It is not the intention to provide detailed explanations of the
primitives in this migration guide. We suggest that you read the
`corresponding
resources <https://qiskit.org/documentation/apidoc/primitives.html>`__
of the Qiskit Terra documentation instead.

We use ``qiskit.primitives.Sampler`` in this guide as an example of
Sampler implementation, which follows ``qiskit.primitives.BaseSampler``
interface. Users can also use other Sampler implementations such as
``BackendSampler`` (qiskit-terra), ``AerSampler`` (qiskit-aer), and
Qiskit Runtime Sampler (qiskit-ibm-runtime).

``MinimumEigenOptimizer``
-------------------------

The former algorithms exist in
``qiskit.algorithms.minimum_eigen_solvers`` and we can access them by
``qiskit.algorithms.*``. On the other hand, the new algorithms exist in
``qiskit.algorithms.minimum_eigensolvers`` and we can access them by
``qiskit.algorithms.minimum_eigensolvers.*``. Note that the difference
is ``minimum_eigen_solvers`` (former) and ``minimum_eigensolvers``
(new).

``MinimumEigenOptimizer`` of Qiskit Optimization can use
``qiskit.algorithms.MinimumEigenSolver`` interface of the former
algorithms and
``qiskit.algorithms.minimum_eigensolvers.SamplingMinimumEigensolver``
interface of the new algorithms. Note that ``MinimumEigenOptimizer``
cannot basically handle
``qiskit.algorithms.minimum_eigensolvers.MinimumEigensolver`` of the new
algorithms. But there is an exception. ``MinimumEigenOptimizer`` can
handle ``algorithms.minimum_eigensolver.NumPyMinimumEigensolver`` though
it inherits
``qiskit.algorithms.minimum_eigensolvers.MinimumEigensolver``. It is
because ``algorithms.minimum_eigensolver.NumPyMinimumEigensolver`` has
an extension that allows users to access the eigen states.

The following is the corresponding table.

.. csv-table::
    :header: Former algorithm, New algorithm

    ``qiskit.algorithms.MinimumEigenSolver``, ``qiskit.algorithms.minimum_eigensolvers.SamplingMinimumEigensolver``
    ``qiskit.algorithms.NumPyMinimumEigensolver``, ``qiskit.algorithms.minimum_eigensolver.NumPyMinimumEigensolver``
    ``qiskit.algorithms.QAOA``, ``qiskit.algorithms.minimum_eigensolvers.QAOA``
    ``qiskit.algorithms.VQE``, ``qiskit.algorithms.minimum_eigensolvers.SamplingVQE``


Setup of a problem

.. code:: python

    from qiskit_optimization import QuadraticProgram
    
    problem = QuadraticProgram("sample")
    problem.binary_var("x")
    problem.binary_var("y")
    problem.maximize(linear={"x": 1, "y": -2})
    print(problem.prettyprint())


.. parsed-literal::

    Problem name: sample
    
    Maximize
      x - 2*y
    
    Subject to
      No constraints
    
      Binary variables (2)
        x y
    


NumPyMinimumEigensolver
~~~~~~~~~~~~~~~~~~~~~~~

Previously

.. code:: python

    from qiskit.algorithms import NumPyMinimumEigensolver
    
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    
    mes = NumPyMinimumEigensolver()
    meo = MinimumEigenOptimizer(min_eigen_solver=mes)
    result = meo.solve(problem)
    print(result)


.. parsed-literal::

    fval=1.0, x=1.0, y=0.0, status=SUCCESS


New

.. code:: python

    from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
    
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    
    mes = NumPyMinimumEigensolver()
    meo = MinimumEigenOptimizer(min_eigen_solver=mes)
    result = meo.solve(problem)
    print(result)


.. parsed-literal::

    fval=1.0, x=1.0, y=0.0, status=SUCCESS


QAOA
~~~~

Previously

.. code:: python

    from qiskit import BasicAer
    from qiskit.algorithms import QAOA
    from qiskit.algorithms.optimizers import COBYLA
    from qiskit.utils import QuantumInstance
    
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    
    backend = BasicAer.get_backend("qasm_simulator")
    shots = 1000
    qins = QuantumInstance(backend=backend, shots=shots)
    mes = QAOA(optimizer=COBYLA(), quantum_instance=qins)
    meo = MinimumEigenOptimizer(min_eigen_solver=mes)
    result = meo.solve(problem)
    print(result)


.. parsed-literal::

    fval=1.0, x=1.0, y=0.0, status=SUCCESS


New

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


.. parsed-literal::

    fval=1.0, x=1.0, y=0.0, status=SUCCESS


VQE (former) → SamplingVQE (new)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously

.. code:: python

    from qiskit import BasicAer
    from qiskit.algorithms import VQE
    from qiskit.algorithms.optimizers import COBYLA
    from qiskit.circuit.library import RealAmplitudes
    from qiskit.utils import QuantumInstance
    
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    
    backend = BasicAer.get_backend("qasm_simulator")
    shots = 1000
    qins = QuantumInstance(backend=backend, shots=shots)
    mes = VQE(ansatz=RealAmplitudes(), optimizer=COBYLA(), quantum_instance=qins)
    meo = MinimumEigenOptimizer(min_eigen_solver=mes)
    result = meo.solve(problem)
    print(result)


.. parsed-literal::

    fval=1.0, x=1.0, y=0.0, status=SUCCESS


New

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


.. parsed-literal::

    fval=1.0, x=1.0, y=0.0, status=SUCCESS


An error occurs due to ``VQE`` with ``Estimator``. You can use
``SamplingVQE`` with ``Sampler`` instead (see the previous cell).

.. code:: python

    from qiskit.algorithms.minimum_eigensolvers import VQE
    from qiskit.algorithms.optimizers import COBYLA
    from qiskit.circuit.library import RealAmplitudes
    from qiskit.primitives import Estimator
    
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    
    mes = VQE(estimator=Estimator(), ansatz=RealAmplitudes(), optimizer=COBYLA())
    try:
        meo = MinimumEigenOptimizer(min_eigen_solver=mes)
    except TypeError as ex:
        print(ex)


.. parsed-literal::

    MinimumEigenOptimizer does not support this VQE. You can use  qiskit.algorithms.minimum_eigensolvers.SamplingVQE instead.


``WarmStartQAOAOptimizer``
--------------------------

``WarmStartQAOAOptimizer`` can use both the former
``qiskit.algorithms.QAOA`` and the new
``qiskit.algorithms.minimum_eigensolvers.QAOA`` as follows.

Previously

.. code:: python

    from qiskit import BasicAer
    from qiskit.algorithms import QAOA
    from qiskit.algorithms.optimizers import COBYLA
    from qiskit.utils import QuantumInstance
    
    from qiskit_optimization.algorithms import WarmStartQAOAOptimizer, SlsqpOptimizer
    
    backend = BasicAer.get_backend("qasm_simulator")
    shots = 1000
    qins = QuantumInstance(backend=backend, shots=shots)
    qaoa = QAOA(optimizer=COBYLA(), quantum_instance=qins)
    optimizer = WarmStartQAOAOptimizer(
        pre_solver=SlsqpOptimizer(), relax_for_pre_solver=True, qaoa=qaoa, epsilon=0.25
    )
    result = optimizer.solve(problem)
    print(result)


.. parsed-literal::

    fval=1.0, x=1.0, y=0.0, status=SUCCESS


New

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


.. parsed-literal::

    fval=1.0, x=1.0, y=0.0, status=SUCCESS


``GroverOptimizer``
-------------------

``GroverOptimizer`` supports both ``QuantumInstance`` and
``BaseSampler``. But users must specify one of them.

Previously

.. code:: python

    from qiskit import BasicAer
    from qiskit.algorithms.optimizers import COBYLA
    from qiskit.utils import QuantumInstance
    
    from qiskit_optimization.algorithms import GroverOptimizer
    
    backend = BasicAer.get_backend("qasm_simulator")
    shots = 1000
    qins = QuantumInstance(backend=backend, shots=shots)
    optimizer = GroverOptimizer(num_value_qubits=3, num_iterations=3, quantum_instance=qins)
    result = optimizer.solve(problem)
    print(result)


.. parsed-literal::

    fval=1.0, x=1.0, y=0.0, status=SUCCESS


New

.. code:: python

    from qiskit.algorithms.optimizers import COBYLA
    from qiskit.primitives import Sampler
    
    from qiskit_optimization.algorithms import GroverOptimizer
    
    optimizer = GroverOptimizer(num_value_qubits=3, num_iterations=3, sampler=Sampler())
    result = optimizer.solve(problem)
    print(result)


.. parsed-literal::

    fval=0.0, x=0.0, y=0.0, status=SUCCESS


An error occurs because both ``quantum_instance`` and ``sampler`` are
set. You can set only one of them.

.. code:: python

    from qiskit import BasicAer
    from qiskit.algorithms.optimizers import COBYLA
    from qiskit.utils import QuantumInstance
    from qiskit.primitives import Sampler
    
    from qiskit_optimization.algorithms import GroverOptimizer
    
    backend = BasicAer.get_backend("qasm_simulator")
    shots = 1000
    qins = QuantumInstance(backend=backend, shots=shots)
    try:
        optimizer = GroverOptimizer(
            num_value_qubits=3, num_iterations=3, quantum_instance=qins, sampler=Sampler()
        )
        # raises an error because both quantum_instance and sampler are set.
    except ValueError as ex:
        print(ex)


.. parsed-literal::

    Only one of quantum_instance or sampler can be passed, not both!


.. code:: python

    import qiskit.tools.jupyter
    
    %qiskit_version_table
    %qiskit_copyright



.. raw:: html

    <h3>Version Information</h3><table><tr><th>Qiskit Software</th><th>Version</th></tr><tr><td><code>qiskit-terra</code></td><td>0.23.0</td></tr><tr><td><code>qiskit-aer</code></td><td>0.11.1</td></tr><tr><td><code>qiskit-optimization</code></td><td>0.5.0</td></tr><tr><td><code>qiskit-machine-learning</code></td><td>0.6.0</td></tr><tr><th>System information</th></tr><tr><td>Python version</td><td>3.9.15</td></tr><tr><td>Python compiler</td><td>Clang 14.0.0 (clang-1400.0.29.102)</td></tr><tr><td>Python build</td><td>main, Oct 11 2022 22:27:25</td></tr><tr><td>OS</td><td>Darwin</td></tr><tr><td>CPUs</td><td>4</td></tr><tr><td>Memory (Gb)</td><td>16.0</td></tr><tr><td colspan='2'>Tue Dec 06 22:08:13 2022 JST</td></tr></table>



.. raw:: html

    <div style='width: 100%; background-color:#d5d9e0;padding-left: 10px; padding-bottom: 10px; padding-right: 10px; padding-top: 5px'><h3>This code is a part of Qiskit</h3><p>&copy; Copyright IBM 2017, 2022.</p><p>This code is licensed under the Apache License, Version 2.0. You may<br>obtain a copy of this license in the LICENSE.txt file in the root directory<br> of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.<p>Any modifications or derivative works of this code must retain this<br>copyright notice, and modified files need to carry a notice indicating<br>that they have been altered from the originals.</p></div>

