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

from itertools import product
from timeit import timeit

import networkx as nx
from qiskit import Aer
from qiskit.algorithms import QAOA, VQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import EfficientSU2
from qiskit.utils import algorithm_globals, QuantumInstance

from qiskit_optimization.algorithms import MinimumEigenOptimizer, GroverOptimizer
from qiskit_optimization.applications import Maxcut
from qiskit_optimization.converters import QuadraticProgramToQubo


class MaxcutBenchmarks:
    params = ([2, 4, 8, 12], [3, 5, 7, 9])
    param_names = ['number of nodes', 'degree']

    def setup(self, n, d):
        seed = 123
        algorithm_globals.random_seed = seed
        qasm_sim = Aer.get_backend('qasm_simulator')
        sv_sim = Aer.get_backend('statevector_simulator')
        self._qins = QuantumInstance(backend=qasm_sim, shots=1, seed_simulator=seed,
                                     seed_transpiler=seed)
        if n >= d:
            graph = nx.random_regular_graph(n=n, d=d)
            self._maxcut = Maxcut(graph=graph)
            self._qp = self._maxcut.to_quadratic_program()
        else:
            raise NotImplementedError

    @staticmethod
    def _generate_qubo(maxcut: Maxcut):
        q_p = maxcut.to_quadratic_program()
        conv = QuadraticProgramToQubo()
        qubo = conv.convert(q_p)
        return qubo

    def time_generate_qubo(self, _, __):
        self._generate_qubo(self._maxcut)

    def time_qaoa(self, _, __):
        meo = MinimumEigenOptimizer(
            min_eigen_solver=QAOA(optimizer=COBYLA(maxiter=1), quantum_instance=self._qins))
        meo.solve(self._qp)

    def time_vqe(self, _, __):
        meo = MinimumEigenOptimizer(
            min_eigen_solver=VQE(optimizer=COBYLA(maxiter=1), ansatz=EfficientSU2(),
                                 quantum_instance=self._qins))
        meo.solve(self._qp)

    def time_grover(self, _, __):
        meo = GroverOptimizer(num_value_qubits=self._qp.get_num_vars() // 2,
                              num_iterations=1, quantum_instance=self._qins)
        meo.solve(self._qp)


if __name__ == '__main__':
    for n, d in product(*MaxcutBenchmarks.params):
        if n < d:
            continue
        bench = MaxcutBenchmarks()
        try:
            bench.setup(n=n, d=d)
        except NotImplementedError:
            continue
        for method in set(dir(MaxcutBenchmarks)):
            if method.startswith('time_'):
                elapsed = timeit(f'bench.{method}(None, None)', number=10, globals=globals())
                print(f'n={n}, d={d}, {method}:\t{elapsed}')
