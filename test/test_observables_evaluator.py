# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Observables Evaluators"""

import unittest
from unittest.mock import MagicMock

from test import QiskitAlgorithmsTestCase

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.primitives import BaseEstimator

from qiskit_optimization.exceptions import AlgorithmError
from qiskit_optimization.observables_evaluator import (
    estimate_observables,
    _handle_zero_ops,
    _prepare_result,
)


class TestEstimateObservables(QiskitAlgorithmsTestCase):
    """Observables Evaluators tests"""

    def setUp(self):
        """Set up a basic quantum circuit and estimator for testing."""
        super().setUp()
        self.estimator = MagicMock(spec=BaseEstimator)
        self.quantum_state = QuantumCircuit(2)  # Simple 2-qubit circuit
        self.observable = SparsePauliOp.from_list([("Z", 1)])
        self.observable_2 = SparsePauliOp.from_list([("X", 1)])
        self.observable_3 = SparsePauliOp.from_list([("Z", 1)])

    def test_estimate_observables_success_with_list(self):
        """Test estimation with a list of observables and successful estimator."""
        self.estimator.run.return_value.result.return_value.values = np.array([1.0, 0.5])
        self.estimator.run.return_value.result.return_value.metadata = [{} for _ in range(2)]
        observables = [self.observable, self.observable_2]
        result = estimate_observables(self.estimator, self.quantum_state, observables)
        # Verify results
        expected_results = [(1.0, {}), (0.5, {})]
        self.assertEqual(result, expected_results)
        self.estimator.run.assert_called_once()

    def test_estimate_observables_success_with_dict(self):
        """Test estimation with a dictionary of observables."""
        self.estimator.run = unittest.mock.MagicMock()
        self.estimator.run.return_value.result.return_value.values = np.array([1.0, 0.5])
        self.estimator.run.return_value.result.return_value.metadata = [{} for _ in range(2)]
        # Use valid BaseOperator instances for testing
        observables = {"obs1": self.observable, "obs2": self.observable_2}
        result = estimate_observables(self.estimator, self.quantum_state, observables)
        # Verify results
        expected_results = [(1.0, {}), (0.5, {})]
        self.assertEqual(
            result, _prepare_result(expected_results, observables)
        )  # Adjust according to your expected output
        # Assert that the estimator was called correctly
        self.estimator.run.assert_called_once()

    def test_estimate_observables_below_threshold(self):
        """Test estimation with values below threshold."""
        self.estimator.run.return_value.result.return_value.values = np.array([0.1, 0.0])
        self.estimator.run.return_value.result.return_value.metadata = [{} for _ in range(2)]
        observables = [self.observable, self.observable_2]
        result = estimate_observables(
            self.estimator, self.quantum_state, observables, threshold=0.2
        )
        # Verify that the results are filtered below the threshold
        expected_results = [(0.0, {}), (0.0, {})]  # Both should be ignored (mean <= threshold)
        self.assertEqual(result, expected_results)
        self.estimator.run.assert_called_once()

    def test_estimate_observables_empty_observables(self):
        """Test estimation with an empty list of observables."""
        observables = []
        result = estimate_observables(self.estimator, self.quantum_state, observables)
        # Verify that the result is an empty list
        self.assertEqual(result, [])
        self.estimator.run.assert_not_called()

    def test_estimate_observables_algorithm_error(self):
        """Test handling of AlgorithmError when estimator fails."""
        self.estimator.run.side_effect = Exception("Failed job")
        observables = [self.observable, self.observable_2]
        with self.assertRaises(AlgorithmError):
            estimate_observables(self.estimator, self.quantum_state, observables)

    def test_handle_zero_ops(self):
        """Test replacing zero operators with SparsePauliOp."""
        observables_list = [self.observable_3, 0, self.observable_3]
        # Ensure num_qubits is accessible and valid
        num_qubits = self.observable_3.num_qubits
        self.assertIsInstance(num_qubits, int)
        result = _handle_zero_ops(observables_list)
        # Check if the zero operator was replaced correctly
        zero_op = SparsePauliOp.from_list([("I" * num_qubits, 0)])
        # Validate that the zero operator was replaced
        self.assertEqual(result[1], zero_op)

    def test_prepare_result_with_list(self):
        """Test the _prepare_result function with a list of observables."""
        observables_results = [(1.0, {"meta1": "data1"}), (0.5, {"meta2": "data2"})]
        observables = [self.observable, self.observable_2]
        result = _prepare_result(observables_results, observables)
        # Verify the output is as expected
        expected_results = [(1.0, {"meta1": "data1"}), (0.5, {"meta2": "data2"})]
        self.assertEqual(result, expected_results)

    def test_prepare_result_with_dict(self):
        """Test the _prepare_result function with a dictionary of observables."""
        observables_results = [(1.0, {"meta1": "data1"}), (0.5, {"meta2": "data2"})]
        observables = {"obs1": self.observable, "obs2": self.observable_2}
        result = _prepare_result(observables_results, observables)
        # Verify the output is as expected
        expected_results = {"obs1": (1.0, {"meta1": "data1"}), "obs2": (0.5, {"meta2": "data2"})}
        self.assertEqual(result, expected_results)


if __name__ == "__main__":
    unittest.main()
