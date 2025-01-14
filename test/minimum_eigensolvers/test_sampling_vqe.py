# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the Sampler VQE."""


import unittest
from functools import partial
from test import QiskitAlgorithmsTestCase

import numpy as np
from ddt import ddt
from scipy.optimize import minimize as scipy_minimize

from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Pauli

from qiskit_optimization.minimum_eigensolvers import SamplingVQE
from qiskit_optimization.optimizers import OptimizerResult
from qiskit_optimization.utils import algorithm_globals


# pylint: disable=invalid-name
def _mock_optimizer(fun, x0, jac=None, bounds=None, inputs=None):
    """A mock of a callable that can be used as minimizer in the VQE.

    If ``inputs`` is given as a dictionary, stores the inputs in that dictionary.
    """
    result = OptimizerResult()
    result.x = np.zeros_like(x0)
    result.fun = fun(result.x)
    result.nit = 0

    if inputs is not None:
        inputs.update({"fun": fun, "x0": x0, "jac": jac, "bounds": bounds})
    return result


@ddt
class TestSamplerVQE(QiskitAlgorithmsTestCase):
    """Test VQE"""

    def setUp(self):
        super().setUp()
        self.optimal_value = -1.38
        self.optimal_bitstring = "10"
        algorithm_globals.random_seed = 42

    def test_optimizer_scipy_callable(self):
        """Test passing a SciPy optimizer directly as callable."""
        vqe = SamplingVQE(
            Sampler(),
            RealAmplitudes(),
            partial(scipy_minimize, method="COBYLA", options={"maxiter": 2}),
        )
        result = vqe.compute_minimum_eigenvalue(Pauli("Z"))
        self.assertEqual(result.cost_function_evals, 2)

    def test_optimizer_callable(self):
        """Test passing a optimizer directly as callable."""
        ansatz = RealAmplitudes(1, reps=1)
        vqe = SamplingVQE(Sampler(), ansatz, _mock_optimizer)
        result = vqe.compute_minimum_eigenvalue(Pauli("Z"))
        self.assertTrue(np.all(result.optimal_point == np.zeros(ansatz.num_parameters)))

    def test_aggregation(self):
        """Test the aggregation works."""

        # test a custom aggregation that just uses the best measurement
        def best_measurement(measurements):
            res = min(measurements, key=lambda meas: meas[1])[1]
            return res

        # test CVaR with alpha of 0.4 (i.e. 40% of the best measurements)
        alpha = 0.4

        ansatz = RealAmplitudes(1, reps=0)
        ansatz.h(0)

        for aggregation in [alpha, best_measurement]:
            with self.subTest(aggregation=aggregation):
                vqe = SamplingVQE(Sampler(), ansatz, _mock_optimizer, aggregation=best_measurement)
                result = vqe.compute_minimum_eigenvalue(Pauli("Z"))

                # evaluation at x0=0 samples -1 and 1 with 50% probability, and our aggregation
                # takes the smallest value
                self.assertAlmostEqual(result.optimal_value, -1)


if __name__ == "__main__":
    unittest.main()
