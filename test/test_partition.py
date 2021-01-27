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

""" Test Partition """

import unittest
from test import QiskitOptimizationTestCase
import numpy as np
from qiskit import BasicAer
from qiskit.circuit.library import RealAmplitudes
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms import NumPyMinimumEigensolver, VQE
from qiskit.algorithms.optimizers import SPSA
from qiskit_optimization.applications.ising import partition
from qiskit_optimization.applications.ising.common import read_numbers_from_file, sample_most_likely


class TestSetPacking(QiskitOptimizationTestCase):
    """Cplex Ising tests."""

    def setUp(self):
        super().setUp()
        input_file = self.get_resource_path('sample.partition')
        number_list = read_numbers_from_file(input_file)
        self.qubit_op, _ = partition.get_operator(number_list)

    def test_partition(self):
        """ Partition test """
        algo = NumPyMinimumEigensolver()
        result = algo.compute_minimum_eigenvalue(operator=self.qubit_op, aux_operators=[])
        x = sample_most_likely(result.eigenstate)
        if x[0] != 0:
            x = np.logical_not(x) * 1
        np.testing.assert_array_equal(x, [0, 1, 0])

    def test_partition_vqe(self):
        """ Partition VQE test """
        algorithm_globals.random_seed = 100
        q_i = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                              seed_simulator=algorithm_globals.random_seed,
                              seed_transpiler=algorithm_globals.random_seed)
        result = VQE(RealAmplitudes(reps=5, entanglement='linear'),
                     SPSA(maxiter=200),
                     max_evals_grouped=2,
                     quantum_instance=q_i).compute_minimum_eigenvalue(operator=self.qubit_op)
        x = sample_most_likely(result.eigenstate)
        self.assertNotEqual(x[0], x[1])
        self.assertNotEqual(x[2], x[1])  # hardcoded oracle


if __name__ == '__main__':
    unittest.main()
