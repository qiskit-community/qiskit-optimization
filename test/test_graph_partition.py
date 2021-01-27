# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Graph Partition """

import unittest
from test import QiskitOptimizationTestCase
import numpy as np
from qiskit import BasicAer
from qiskit.circuit.library import RealAmplitudes
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms import NumPyMinimumEigensolver, VQE
from qiskit.algorithms.optimizers import SPSA
from qiskit_optimization.applications.ising import graph_partition
from qiskit_optimization.applications.ising.common import random_graph, sample_most_likely


class TestGraphPartition(QiskitOptimizationTestCase):
    """Cplex Ising tests."""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 100
        self.num_nodes = 4
        self.w = random_graph(self.num_nodes, edge_prob=0.8, weight_range=10)
        self.qubit_op, self.offset = graph_partition.get_operator(self.w)

    def _brute_force(self):
        # use the brute-force way to generate the oracle
        def bitfield(n, length):
            result = np.binary_repr(n, length)
            return [int(digit) for digit in result]  # [2:] to chop off the "0b" part

        nodes = self.num_nodes
        maximum = 2 ** nodes
        minimal_v = np.inf
        for i in range(maximum):
            cur = bitfield(i, nodes)

            how_many_nonzero = np.count_nonzero(cur)
            if how_many_nonzero * 2 != nodes:  # not balanced
                continue

            cur_v = graph_partition.objective_value(np.array(cur), self.w)
            if cur_v < minimal_v:
                minimal_v = cur_v
        return minimal_v

    def test_graph_partition(self):
        """ Graph Partition test """
        algo = NumPyMinimumEigensolver()
        result = algo.compute_minimum_eigenvalue(operator=self.qubit_op, aux_operators=[])
        x = sample_most_likely(result.eigenstate)
        # check against the oracle
        ising_sol = graph_partition.get_graph_solution(x)
        # solutions are equivalent
        self.assertEqual(graph_partition.objective_value(np.array([0, 1, 0, 1]), self.w),
                         graph_partition.objective_value(ising_sol, self.w))
        oracle = self._brute_force()
        self.assertEqual(graph_partition.objective_value(x, self.w), oracle)

    def test_graph_partition_vqe(self):
        """ Graph Partition VQE test """
        algorithm_globals.random_seed = 10213
        wavefunction = RealAmplitudes(self.qubit_op.num_qubits, insert_barriers=True,
                                      reps=5, entanglement='linear')
        q_i = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                              seed_simulator=algorithm_globals.random_seed,
                              seed_transpiler=algorithm_globals.random_seed)
        result = VQE(wavefunction,
                     SPSA(maxiter=300),
                     max_evals_grouped=2,
                     quantum_instance=q_i).compute_minimum_eigenvalue(operator=self.qubit_op)

        x = sample_most_likely(result.eigenstate)
        # check against the oracle
        ising_sol = graph_partition.get_graph_solution(x)
        self.assertEqual(graph_partition.objective_value(np.array([0, 1, 0, 1]), self.w),
                         graph_partition.objective_value(ising_sol, self.w))
        oracle = self._brute_force()
        self.assertEqual(graph_partition.objective_value(x, self.w), oracle)


if __name__ == '__main__':
    unittest.main()
