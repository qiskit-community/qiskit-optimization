# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the QAOA algorithm."""

import unittest
from functools import partial
from test import QiskitAlgorithmsTestCase

import numpy as np
import rustworkx as rx
from ddt import data, ddt, idata, unpack
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.result import QuasiDistribution
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.utils.optionals import HAS_AER
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler, SamplerV2
from scipy.optimize import minimize as scipy_minimize

from qiskit_optimization.minimum_eigensolvers import QAOA
from qiskit_optimization.optimizers import COBYLA, NELDER_MEAD
from qiskit_optimization.utils import algorithm_globals

W1 = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
P1 = 1
M1 = SparsePauliOp.from_list(
    [
        ("IIIX", 1),
        ("IIXI", 1),
        ("IXII", 1),
        ("XIII", 1),
    ]
)
S1 = {"0101", "1010"}


W2 = np.array(
    [
        [0.0, 8.0, -9.0, 0.0],
        [8.0, 0.0, 7.0, 9.0],
        [-9.0, 7.0, 0.0, -8.0],
        [0.0, 9.0, -8.0, 0.0],
    ]
)
P2 = 1
M2 = None
S2 = {"1011", "0100"}

CUSTOM_SUPERPOSITION = [1 / np.sqrt(15)] * 15 + [0]


@ddt
class TestQAOA(QiskitAlgorithmsTestCase):
    """Test QAOA with MaxCut."""

    @unittest.skipUnless(HAS_AER, "qiskit-aer is required to run this test")
    def setUp(self):
        super().setUp()
        self.seed = 10598
        algorithm_globals.random_seed = self.seed
        self.sampler = {
            "v1": Sampler(run_options={"seed_simulator": self.seed, "shots": 10000}),
            "v2": SamplerV2(seed=self.seed, default_shots=10000),
        }
        self.pass_manager = generate_preset_pass_manager(
            optimization_level=1, target=AerSimulator().target, seed_transpiler=self.seed
        )

    @idata(
        [
            [W1, P1, M1, S1, "v1"],
            [W1, P1, M1, S1, "v2"],
            [W2, P2, M2, S2, "v1"],
            [W2, P2, M2, S2, "v2"],
        ]
    )
    @unpack
    # pylint: disable=too-many-positional-arguments
    def test_qaoa(self, w, reps, mixer, solutions, version):
        """QAOA test"""
        self.log.debug("Testing %s-step QAOA with MaxCut on graph\n%s", reps, w)

        qubit_op, _ = self._get_operator(w)

        qaoa = QAOA(
            self.sampler[version], COBYLA(), reps=reps, mixer=mixer, pass_manager=self.pass_manager
        )
        result = qaoa.compute_minimum_eigenvalue(operator=qubit_op)

        x = self._sample_most_likely(result.eigenstate)
        graph_solution = self._get_graph_solution(x)
        self.assertIn(graph_solution, solutions)

    @idata(
        [
            [W1, P1, S1, "v1"],
            [W1, P1, S1, "v2"],
            [W2, P2, S2, "v1"],
            [W2, P2, S2, "v2"],
        ]
    )
    @unpack
    def test_qaoa_qc_mixer(self, w, prob, solutions, version):
        """QAOA test with a mixer as a parameterized circuit"""
        self.log.debug(
            "Testing %s-step QAOA with MaxCut on graph with a mixer as a parameterized circuit\n%s",
            prob,
            w,
        )

        optimizer = COBYLA()
        qubit_op, _ = self._get_operator(w)

        num_qubits = qubit_op.num_qubits
        mixer = QuantumCircuit(num_qubits)
        theta = Parameter("θ")
        mixer.rx(theta, range(num_qubits))

        qaoa = QAOA(
            self.sampler[version], optimizer, reps=prob, mixer=mixer, pass_manager=self.pass_manager
        )
        result = qaoa.compute_minimum_eigenvalue(operator=qubit_op)
        x = self._sample_most_likely(result.eigenstate)
        graph_solution = self._get_graph_solution(x)
        self.assertIn(graph_solution, solutions)

    @data("v1", "v2")
    def test_qaoa_qc_mixer_many_parameters(self, version):
        """QAOA test with a mixer as a parameterized circuit with the num of parameters > 1."""
        optimizer = COBYLA()
        qubit_op, _ = self._get_operator(W1)

        num_qubits = qubit_op.num_qubits
        mixer = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            theta = Parameter("θ" + str(i))
            mixer.rx(theta, range(num_qubits))

        qaoa = QAOA(
            self.sampler[version], optimizer, reps=2, mixer=mixer, pass_manager=self.pass_manager
        )
        result = qaoa.compute_minimum_eigenvalue(operator=qubit_op)
        x = self._sample_most_likely(result.eigenstate)
        self.log.debug(x)
        graph_solution = self._get_graph_solution(x)
        self.assertIn(graph_solution, S1)

    @data("v1", "v2")
    def test_qaoa_qc_mixer_no_parameters(self, version):
        """QAOA test with a mixer as a parameterized circuit with zero parameters."""
        qubit_op, _ = self._get_operator(W1)

        num_qubits = qubit_op.num_qubits
        mixer = QuantumCircuit(num_qubits)
        # just arbitrary circuit
        mixer.rx(np.pi / 2, range(num_qubits))

        qaoa = QAOA(
            self.sampler[version], COBYLA(), reps=1, mixer=mixer, pass_manager=self.pass_manager
        )
        result = qaoa.compute_minimum_eigenvalue(operator=qubit_op)
        # we just assert that we get a result, it is not meaningful.
        self.assertIsNotNone(result.eigenstate)

    @data("v1", "v2")
    def test_change_operator_size(self, version):
        """QAOA change operator size test"""
        qubit_op, _ = self._get_operator(
            np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
        )
        qaoa = QAOA(self.sampler[version], COBYLA(), reps=1, pass_manager=self.pass_manager)
        result = qaoa.compute_minimum_eigenvalue(operator=qubit_op)
        x = self._sample_most_likely(result.eigenstate)
        graph_solution = self._get_graph_solution(x)
        with self.subTest(msg="QAOA 4x4"):
            self.assertIn(graph_solution, {"0101", "1010"})

        qubit_op, _ = self._get_operator(
            np.array(
                [
                    [0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1, 0],
                ]
            )
        )
        result = qaoa.compute_minimum_eigenvalue(operator=qubit_op)
        x = self._sample_most_likely(result.eigenstate)
        graph_solution = self._get_graph_solution(x)
        with self.subTest(msg="QAOA 6x6"):
            self.assertIn(graph_solution, {"010101", "101010"})

    @idata(
        [
            [W2, S2, None, "v1"],
            [W2, S2, None, "v2"],
            [W2, S2, [0.001, 0.0], "v1"],
            [W2, S2, [0.001, 0.0], "v2"],
            [W2, S2, [1.0, 0.8], "v1"],
            [W2, S2, [1.0, 0.8], "v2"],
        ]
    )
    @unpack
    def test_qaoa_initial_point(self, w, solutions, init_pt, version):
        """Check first parameter value used is initial point as expected"""
        qubit_op, _ = self._get_operator(w)

        first_pt = []

        def cb_callback(eval_count, parameters, mean, metadata):
            nonlocal first_pt
            if eval_count == 1:
                first_pt = list(parameters)

        qaoa = QAOA(
            self.sampler[version],
            COBYLA(),
            initial_point=init_pt,
            callback=cb_callback,
            pass_manager=self.pass_manager,
        )
        result = qaoa.compute_minimum_eigenvalue(operator=qubit_op)

        x = self._sample_most_likely(result.eigenstate)
        graph_solution = self._get_graph_solution(x)

        with self.subTest("Initial Point"):
            # If None the preferred random initial point of QAOA variational form
            if init_pt is None:
                self.assertLess(result.eigenvalue, -0.97)
            else:
                self.assertListEqual(init_pt, first_pt)

        with self.subTest("Solution"):
            self.assertIn(graph_solution, solutions)

    @data("v1", "v2")
    def test_qaoa_random_initial_point(self, version):
        """QAOA random initial point"""
        # the function undirected_gnp_random_graph() does exist in
        # rustworkx packagebut the linter can't see it
        w = rx.adjacency_matrix(
            rx.undirected_gnp_random_graph(  # pylint: disable=no-member
                5, 0.5, seed=algorithm_globals.random_seed
            )
        )
        qubit_op, _ = self._get_operator(w)
        qaoa = QAOA(self.sampler[version], NELDER_MEAD(), reps=2, pass_manager=self.pass_manager)
        result = qaoa.compute_minimum_eigenvalue(operator=qubit_op)
        self.assertLess(result.eigenvalue, -0.97)

    @data("v1", "v2")
    def test_optimizer_scipy_callable(self, version):
        """Test passing a SciPy optimizer directly as callable."""
        w = rx.adjacency_matrix(
            rx.undirected_gnp_random_graph(  # pylint: disable=no-member
                5, 0.5, seed=algorithm_globals.random_seed
            )
        )
        qubit_op, _ = self._get_operator(w)
        qaoa = QAOA(
            self.sampler[version],
            partial(scipy_minimize, method="Nelder-Mead", options={"maxiter": 2}),
            pass_manager=self.pass_manager,
        )
        result = qaoa.compute_minimum_eigenvalue(qubit_op)
        self.assertEqual(result.cost_function_evals, 5)

    def _get_operator(self, weight_matrix):
        """Generate Hamiltonian for the max-cut problem of a graph.

        Args:
            weight_matrix (numpy.ndarray) : adjacency matrix.

        Returns:
            PauliSumOp: operator for the Hamiltonian
            float: a constant shift for the obj function.

        """
        num_nodes = weight_matrix.shape[0]
        pauli_list = []
        shift = 0
        for i in range(num_nodes):
            for j in range(i):
                if weight_matrix[i, j] != 0:
                    x_p = np.zeros(num_nodes, dtype=bool)
                    z_p = np.zeros(num_nodes, dtype=bool)
                    z_p[i] = True
                    z_p[j] = True
                    pauli_list.append([0.5 * weight_matrix[i, j], Pauli((z_p, x_p))])
                    shift -= 0.5 * weight_matrix[i, j]
        lst = [(pauli[1].to_label(), pauli[0]) for pauli in pauli_list]
        return SparsePauliOp.from_list(lst), shift

    def _get_graph_solution(self, x: np.ndarray) -> str:
        """Get graph solution from binary string.

        Args:
            x : binary string as numpy array.

        Returns:
            a graph solution as string.
        """

        return "".join([str(int(i)) for i in 1 - x])

    def _sample_most_likely(self, state_vector: QuasiDistribution) -> np.ndarray:
        """Compute the most likely binary string from state vector.
        Args:
            state_vector: Quasi-distribution.

        Returns:
            Binary string as numpy.ndarray of ints.
        """
        max_key = max(state_vector.keys())
        n = max_key.bit_length()
        values = np.zeros(max_key + 1)
        for k, v in state_vector.items():
            values[k] = v
        k = np.argmax(np.abs(values))
        x = np.zeros(n)
        for i in range(n):
            x[i] = k % 2
            k >>= 1
        return x


if __name__ == "__main__":
    unittest.main()
