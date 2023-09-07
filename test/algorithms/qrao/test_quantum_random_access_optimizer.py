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

"""Tests for QuantumRandomAccessOptimizer"""
import unittest
from test.optimization_test_case import QiskitOptimizationTestCase

import numpy as np
from qiskit_algorithms import (
    NumPyMinimumEigensolver,
    NumPyMinimumEigensolverResult,
    VQE,
    VQEResult,
)
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Estimator
from qiskit.utils import algorithm_globals

from qiskit_optimization.algorithms import SolutionSample
from qiskit_optimization.algorithms.optimization_algorithm import OptimizationResultStatus
from qiskit_optimization.algorithms.qrao import (
    QuantumRandomAccessEncoding,
    QuantumRandomAccessOptimizationResult,
    QuantumRandomAccessOptimizer,
    RoundingContext,
    RoundingResult,
)
from qiskit_optimization.problems import QuadraticProgram


class TestQuantumRandomAccessOptimizer(QiskitOptimizationTestCase):
    """QuantumRandomAccessOptimizer tests."""

    def setUp(self):
        super().setUp()
        self.problem = QuadraticProgram()
        self.problem.binary_var("x")
        self.problem.binary_var("y")
        self.problem.binary_var("z")
        self.problem.minimize(linear={"x": 1, "y": 2, "z": 3})
        self.encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=3)
        self.encoding.encode(self.problem)
        self.ansatz = RealAmplitudes(self.encoding.num_qubits)  # for VQE
        algorithm_globals.random_seed = 50

    def test_solve_relaxed_numpy(self):
        """Test QuantumRandomAccessOptimizer with NumPyMinimumEigensolver."""
        np_solver = NumPyMinimumEigensolver()
        qrao = QuantumRandomAccessOptimizer(min_eigen_solver=np_solver)
        relaxed_results, rounding_context = qrao.solve_relaxed(encoding=self.encoding)
        self.assertIsInstance(relaxed_results, NumPyMinimumEigensolverResult)
        self.assertAlmostEqual(relaxed_results.eigenvalue, -3.24037, places=5)
        self.assertEqual(len(relaxed_results.aux_operators_evaluated), 3)
        self.assertAlmostEqual(relaxed_results.aux_operators_evaluated[0][0], 0.26726, places=5)
        self.assertAlmostEqual(relaxed_results.aux_operators_evaluated[1][0], 0.53452, places=5)
        self.assertAlmostEqual(relaxed_results.aux_operators_evaluated[2][0], 0.80178, places=5)
        self.assertIsInstance(rounding_context, RoundingContext)
        self.assertEqual(rounding_context.circuit.num_qubits, self.ansatz.num_qubits)
        self.assertEqual(rounding_context.encoding, self.encoding)
        self.assertAlmostEqual(rounding_context.expectation_values[0], 0.26726, places=5)
        self.assertAlmostEqual(rounding_context.expectation_values[1], 0.53452, places=5)
        self.assertAlmostEqual(rounding_context.expectation_values[2], 0.80178, places=5)

    def test_solve_relaxed_vqe(self):
        """Test QuantumRandomAccessOptimizer with VQE."""
        vqe = VQE(
            ansatz=self.ansatz,
            optimizer=COBYLA(),
            estimator=Estimator(),
        )
        qrao = QuantumRandomAccessOptimizer(min_eigen_solver=vqe)
        relaxed_results, rounding_context = qrao.solve_relaxed(encoding=self.encoding)
        self.assertIsInstance(relaxed_results, VQEResult)
        self.assertAlmostEqual(relaxed_results.eigenvalue, -2.73861, delta=1e-4)
        self.assertEqual(len(relaxed_results.aux_operators_evaluated), 3)
        self.assertAlmostEqual(relaxed_results.aux_operators_evaluated[0][0], 0.31632, delta=1e-4)
        self.assertAlmostEqual(relaxed_results.aux_operators_evaluated[1][0], 0, delta=1e-4)
        self.assertAlmostEqual(relaxed_results.aux_operators_evaluated[2][0], 0.94865, delta=1e-4)
        self.assertIsInstance(rounding_context, RoundingContext)
        self.assertEqual(rounding_context.circuit.num_qubits, self.ansatz.num_qubits)
        self.assertEqual(rounding_context.encoding, self.encoding)
        self.assertAlmostEqual(rounding_context.expectation_values[0], 0.31632, delta=1e-4)
        self.assertAlmostEqual(rounding_context.expectation_values[1], 0, delta=1e-4)
        self.assertAlmostEqual(rounding_context.expectation_values[2], 0.94865, delta=1e-4)

    def test_require_aux_operator_support(self):
        """Test whether the eigensolver supports auxiliary operator.
        If auxiliary operators are not supported, a TypeError should be raised.
        """

        class ModifiedVQE(VQE):
            """Modified VQE method without auxiliary operator support.
            Since no existing eigensolver seems to lack auxiliary operator support,
            we have created one that claims to lack it.
            """

            @classmethod
            def supports_aux_operators(cls) -> bool:
                return False

        vqe = ModifiedVQE(
            ansatz=self.ansatz,
            optimizer=COBYLA(),
            estimator=Estimator(),
        )
        with self.assertRaises(TypeError):
            QuantumRandomAccessOptimizer(min_eigen_solver=vqe)

    def test_solve_numpy(self):
        """Test QuantumRandomAccessOptimizer with NumPyMinimumEigensolver."""
        np_solver = NumPyMinimumEigensolver()
        qrao = QuantumRandomAccessOptimizer(min_eigen_solver=np_solver)
        results = qrao.solve(problem=self.problem)
        self.assertIsInstance(results, QuantumRandomAccessOptimizationResult)
        self.assertEqual(results.fval, 0)
        self.assertEqual(len(results.samples), 1)
        np.testing.assert_array_almost_equal(results.samples[0].x, [0, 0, 0])
        self.assertAlmostEqual(results.samples[0].fval, 0)
        self.assertAlmostEqual(results.samples[0].probability, 1.0)
        self.assertEqual(results.samples[0].status, OptimizationResultStatus.SUCCESS)
        self.assertAlmostEqual(results.relaxed_fval, -0.24037, places=5)
        self.assertIsInstance(results.relaxed_result, NumPyMinimumEigensolverResult)
        self.assertAlmostEqual(results.relaxed_result.eigenvalue, -3.24037, places=5)
        self.assertEqual(len(results.relaxed_result.aux_operators_evaluated), 3)
        self.assertAlmostEqual(
            results.relaxed_result.aux_operators_evaluated[0][0], 0.26726, places=5
        )
        self.assertAlmostEqual(
            results.relaxed_result.aux_operators_evaluated[1][0], 0.53452, places=5
        )
        self.assertAlmostEqual(
            results.relaxed_result.aux_operators_evaluated[2][0], 0.80178, places=5
        )
        self.assertIsInstance(results.rounding_result, RoundingResult)
        self.assertAlmostEqual(results.rounding_result.expectation_values[0], 0.26726, places=5)
        self.assertAlmostEqual(results.rounding_result.expectation_values[1], 0.53452, places=5)
        self.assertAlmostEqual(results.rounding_result.expectation_values[2], 0.80178, places=5)
        self.assertIsInstance(results.rounding_result.samples[0], SolutionSample)

    def test_solve_quadratic(self):
        """Test QuantumRandomAccessOptimizer with a quadratic objective function."""
        # quadratic objective
        problem2 = QuadraticProgram()
        problem2.binary_var("x")
        problem2.binary_var("y")
        problem2.binary_var("z")
        problem2.maximize(linear={"x": 1, "y": 2, "z": 3}, quadratic={("y", "z"): -4})
        np_solver = NumPyMinimumEigensolver()
        qrao = QuantumRandomAccessOptimizer(min_eigen_solver=np_solver)
        results = qrao.solve(problem2)
        self.assertIsInstance(results, QuantumRandomAccessOptimizationResult)
        self.assertEqual(results.fval, 4)
        self.assertEqual(len(results.samples), 1)
        np.testing.assert_array_almost_equal(results.samples[0].x, [1, 0, 1])
        self.assertAlmostEqual(results.samples[0].fval, 4)
        self.assertAlmostEqual(results.samples[0].probability, 1.0)
        self.assertEqual(results.samples[0].status, OptimizationResultStatus.SUCCESS)
        self.assertAlmostEqual(results.relaxed_fval, -5.98852, places=5)
        self.assertIsInstance(results.relaxed_result, NumPyMinimumEigensolverResult)
        self.assertAlmostEqual(results.relaxed_result.eigenvalue, -3.98852, places=5)
        self.assertEqual(len(results.relaxed_result.aux_operators_evaluated), 3)
        self.assertAlmostEqual(
            results.relaxed_result.aux_operators_evaluated[0][0], -0.27735, places=5
        )
        self.assertAlmostEqual(
            results.relaxed_result.aux_operators_evaluated[1][0], 0.96077, places=5
        )
        self.assertAlmostEqual(results.relaxed_result.aux_operators_evaluated[2][0], -1, places=5)
        self.assertIsInstance(results.rounding_result, RoundingResult)
        self.assertAlmostEqual(results.rounding_result.expectation_values[0], -0.27735, places=5)
        self.assertAlmostEqual(results.rounding_result.expectation_values[1], 0.96077, places=5)
        self.assertAlmostEqual(results.rounding_result.expectation_values[2], -1, places=5)
        self.assertIsInstance(results.rounding_result.samples[0], SolutionSample)

    def test_empty_encoding(self):
        """Test the encoding is empty."""
        np_solver = NumPyMinimumEigensolver()
        encoding = QuantumRandomAccessEncoding(3)
        with self.assertRaises(ValueError):
            qrao = QuantumRandomAccessOptimizer(min_eigen_solver=np_solver)
            qrao.solve_relaxed(encoding=encoding)


if __name__ == "__main__":
    unittest.main()
