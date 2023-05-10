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

"""Tests for QuantumRandomAccessEncoding"""
import itertools
import unittest
from test.optimization_test_case import QiskitOptimizationTestCase

from ddt import ddt, data, unpack
import numpy as np
import networkx as nx

from qiskit.circuit import QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp

from qiskit_optimization.algorithms.qrao import (
    EncodingCommutationVerifier,
    QuantumRandomAccessEncoding,
)
from qiskit_optimization.problems import QuadraticProgram, QuadraticObjective
from qiskit_optimization.applications import Maxcut


class TestQuantumRandomAccessEncoding(QiskitOptimizationTestCase):
    """QuantumRandomAccessEncoding tests."""

    def setUp(self):
        super().setUp()
        self.problem = QuadraticProgram()
        self.problem.binary_var("x")
        self.problem.binary_var("y")
        self.problem.binary_var("z")
        self.problem.minimize(linear={"x": 1, "y": 2, "z": 3})

    def test_31p_qrac_encoding(self):
        """Test (3,1,p) QRAC"""
        encoding = QuantumRandomAccessEncoding(3)
        self.assertFalse(encoding.frozen)  # frozen is False
        encoding.encode(self.problem)
        expected_op = PauliSumOp(
            SparsePauliOp(
                ["X", "Y", "Z"], coeffs=[-np.sqrt(3) / 2, 2 * -np.sqrt(3) / 2, 3 * -np.sqrt(3) / 2]
            ),
            coeff=1.0,
        )

        self.assertTrue(encoding.frozen)  # frozen is True
        self.assertEqual(encoding.qubit_op, expected_op)
        self.assertEqual(encoding.num_vars, 3)
        self.assertEqual(encoding.num_qubits, 1)
        self.assertEqual(encoding.offset, 3)
        self.assertEqual(encoding.max_vars_per_qubit, 3)
        self.assertEqual(encoding.q2vars, [[0, 1, 2]])
        self.assertEqual(
            encoding.var2op,
            {
                0: (0, SparsePauliOp(["X"], coeffs=[1.0])),
                1: (0, SparsePauliOp(["Y"], coeffs=[1.0])),
                2: (0, SparsePauliOp(["Z"], coeffs=[1.0])),
            },
        )
        self.assertEqual(encoding.compression_ratio, 3)
        self.assertEqual(encoding.minimum_recovery_probability, (1 + 1 / np.sqrt(3)) / 2)
        self.assertEqual(encoding.problem, self.problem)

    def test_21p_qrac_encoding(self):
        """Test (2,1,p) QRAC"""
        encoding = QuantumRandomAccessEncoding(2)
        self.assertFalse(encoding.frozen)  # frozen is False
        encoding.encode(self.problem)
        expected_op = PauliSumOp(
            SparsePauliOp(
                ["XI", "ZI", "IX"],
                coeffs=[-np.sqrt(2) / 2, 2 * -np.sqrt(2) / 2, 3 * -np.sqrt(2) / 2],
            ),
            coeff=1.0,
        )

        self.assertTrue(encoding.frozen)  # frozen is True
        self.assertEqual(encoding.qubit_op, expected_op)
        self.assertEqual(encoding.num_vars, 3)
        self.assertEqual(encoding.num_qubits, 2)
        self.assertEqual(encoding.offset, 3)
        self.assertEqual(encoding.max_vars_per_qubit, 2)
        self.assertEqual(encoding.q2vars, [[0, 1], [2]])
        self.assertEqual(
            encoding.var2op,
            {
                0: (0, SparsePauliOp(["X"], coeffs=[1.0])),
                1: (0, SparsePauliOp(["Z"], coeffs=[1.0])),
                2: (1, SparsePauliOp(["X"], coeffs=[1.0])),
            },
        )
        self.assertEqual(encoding.compression_ratio, 1.5)
        self.assertEqual(encoding.minimum_recovery_probability, (1 + 1 / np.sqrt(2)) / 2)
        self.assertEqual(encoding.problem, self.problem)

    def test_11p_qrac_encoding(self):
        """Test (1,1,p) QRAC"""
        encoding = QuantumRandomAccessEncoding(1)
        self.assertFalse(encoding.frozen)  # frozen is False
        encoding.encode(self.problem)
        expected_op = PauliSumOp(
            SparsePauliOp(["ZII", "IZI", "IIZ"], coeffs=[-0.5, -1.0, -1.5]),
            coeff=1.0,
        )

        self.assertTrue(encoding.frozen)  # frozen is True
        self.assertEqual(encoding.qubit_op, expected_op)
        self.assertEqual(encoding.num_vars, 3)
        self.assertEqual(encoding.num_qubits, 3)
        self.assertEqual(encoding.offset, 3)
        self.assertEqual(encoding.max_vars_per_qubit, 1)
        self.assertEqual(encoding.q2vars, [[0], [1], [2]])
        self.assertEqual(
            encoding.var2op,
            {
                0: (0, SparsePauliOp(["Z"], coeffs=[1.0])),
                1: (1, SparsePauliOp(["Z"], coeffs=[1.0])),
                2: (2, SparsePauliOp(["Z"], coeffs=[1.0])),
            },
        )
        self.assertEqual(encoding.compression_ratio, 1)
        self.assertEqual(encoding.minimum_recovery_probability, 1)
        self.assertEqual(encoding.problem, self.problem)

    def test_qrac_state_prep(self):
        """Test that state preparation circuit is correct"""
        dvars = [0, 1, 1]
        with self.subTest(msg="(3,1,p) QRAC"):
            encoding = QuantumRandomAccessEncoding(3)
            encoding.encode(self.problem)
            state_prep_circ = encoding.state_preparation_circuit(dvars=dvars)
            circ = QuantumCircuit(1)
            BETA = np.arccos(1 / np.sqrt(3))
            circ.r(np.pi - BETA, np.pi / 4, 0)
            self.assertEqual(state_prep_circ, circ)

        with self.subTest(msg="(2,1,p) QRAC"):
            encoding = QuantumRandomAccessEncoding(2)
            encoding.encode(self.problem)
            state_prep_circ = encoding.state_preparation_circuit(dvars=dvars)
            circ = QuantumCircuit(2)
            circ.x(0)
            circ.r(-3 * np.pi / 4, -np.pi / 2, 0)
            circ.r(-3 * np.pi / 4, -np.pi / 2, 1)
            self.assertEqual(state_prep_circ, circ)

        with self.subTest(msg="(1,1,p) QRAC"):
            encoding = QuantumRandomAccessEncoding(1)
            encoding.encode(self.problem)
            state_prep_circ = encoding.state_preparation_circuit(dvars=dvars)
            circ = QuantumCircuit(3)
            circ.x(0)
            circ.x(1)
            self.assertEqual(state_prep_circ, circ)

    def test_qrac_unsupported_encoding(self):
        """Test that exception is raised if ``max_vars_per_qubit`` is invalid"""
        with self.assertRaises(ValueError):
            QuantumRandomAccessEncoding(4)
        with self.assertRaises(ValueError):
            QuantumRandomAccessEncoding(0)


@ddt
class TestEncodingCommutationVerifier(QiskitOptimizationTestCase):
    """Tests for EncodingCommutationVerifier."""

    def test_encoding_commutation_verifier(self):
        """Test EncodingCommutationVerifier"""
        problem = QuadraticProgram()
        problem.binary_var("x")
        problem.binary_var("y")
        problem.binary_var("z")
        problem.minimize(linear={"x": 1, "y": 2, "z": 3})

        encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=3)
        encoding.encode(problem)
        verifier = EncodingCommutationVerifier(encoding)
        self.assertEqual(len(verifier), 2**encoding.num_vars)
        for _, obj_val, encoded_obj_val in verifier:
            self.assertAlmostEqual(obj_val, encoded_obj_val)

    @data(*itertools.product([1, 2, 3], ["minimize", "maximize"]))
    @unpack
    def test_one_qubit_qrac(self, max_vars_per_qubit, task):
        """Test commutation of single qubit QRAC with non-uniform weights, degree 1 terms"""

        problem = QuadraticProgram()
        nodes = list(range(max_vars_per_qubit))
        _ = [problem.binary_var(name=f"x{i}") for i in nodes]
        obj = {f"x{i}": 2 * (i + 1) for i in nodes}
        if task == "minimize":
            problem.minimize(linear=obj)
        else:
            problem.maximize(linear=obj)
        check_problem_commutation(problem, max_vars_per_qubit)

    @data(
        *itertools.product(
            [1, 2, 3], [QuadraticObjective.Sense.MINIMIZE, QuadraticObjective.Sense.MAXIMIZE]
        )
    )
    @unpack
    def test_uniform_weights_degree_2(self, max_vars_per_qubit, task):
        """Test problem commutation with degree 2 terms"""
        # Note that the variable embedding has some qubits with 1, 2, and 3 qubits
        elist = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (1, 4), (2, 4)]
        graph = nx.from_edgelist(elist)
        for u, v in elist:
            graph[u][v]["weight"] = (u + 1) * (v + 2)

        maxcut = Maxcut(graph)
        problem = maxcut.to_quadratic_program()
        problem.objective.sense = task
        check_problem_commutation(problem, max_vars_per_qubit)

    @data(1, 2, 3)
    def test_random_unweighted_maxcut(self, max_vars_per_qubit):
        """Test problem commutation with random unweighted MaxCut"""
        graph = nx.random_regular_graph(3, 8)
        maxcut = Maxcut(graph)
        problem = maxcut.to_quadratic_program()
        check_problem_commutation(problem, max_vars_per_qubit)

    @data(1, 2, 3)
    def test_random_weighted_maxcut(self, max_vars_per_qubit):
        """Test problem commutation with random weighted MaxCut"""
        graph = nx.random_regular_graph(3, 8)
        for u, v in graph.edges:
            graph[u][v]["weight"] = np.random.randint(1, 10)
        maxcut = Maxcut(graph)
        problem = maxcut.to_quadratic_program()
        check_problem_commutation(problem, max_vars_per_qubit)


def check_problem_commutation(problem: QuadraticProgram, max_vars_per_qubit: int):
    """Utility function to check that the problem commutes with its encoding"""
    encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=max_vars_per_qubit)
    encoding.encode(problem)
    verifier = EncodingCommutationVerifier(encoding)
    assert len(verifier) == 2**encoding.num_vars
    assert all(np.isclose(obj_val, encoded_obj_val) for _, obj_val, encoded_obj_val in verifier)


if __name__ == "__main__":
    unittest.main()
