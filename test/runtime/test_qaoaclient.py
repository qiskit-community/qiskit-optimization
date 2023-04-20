# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the QAOA client."""

import unittest
import warnings
from test import QiskitOptimizationTestCase

import numpy as np
from ddt import data, ddt
from qiskit.algorithms.optimizers import COBYLA
from qiskit.opflow import I, Z
from qiskit.providers.basicaer import QasmSimulatorPy

from qiskit_optimization.runtime import QAOAClient, VQERuntimeResult

from .fake_vqeruntime import FakeQAOARuntimeProvider


@ddt
class TestQAOAClient(QiskitOptimizationTestCase):
    """Test the QAOA client."""

    def setUp(self):
        super().setUp()
        self.provider = FakeQAOARuntimeProvider()

    @data(
        {"name": "SPSA", "maxiter": 100},
        COBYLA(),
    )
    def test_standard_case(self, optimizer):
        """Test a standard use case."""
        reps = 2
        initial_point = np.random.RandomState(42).random(2 * reps)
        backend = QasmSimulatorPy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            operator = Z ^ I ^ Z
            qaoa = QAOAClient(
                optimizer=optimizer,
                reps=reps,
                initial_point=initial_point,
                backend=backend,
                provider=self.provider,
            )
            result = qaoa.compute_minimum_eigenvalue(operator)

        self.assertIsInstance(result, VQERuntimeResult)


if __name__ == "__main__":
    unittest.main()
