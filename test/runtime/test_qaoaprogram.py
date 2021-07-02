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

"""Test the QAOA program."""

from test import QiskitOptimizationTestCase

import unittest
import numpy as np
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit.opflow import I, Z

from qiskit_optimization.runtime import QAOAProgram, VQEProgramResult

from .fake_vqeruntime import FakeRuntimeProvider


class TestQAOAProgram(QiskitOptimizationTestCase):
    """Test the QAOA program."""

    def setUp(self):
        super().setUp()
        self.provider = FakeRuntimeProvider()

    def test_standard_case(self):
        """Test a standard use case."""
        operator = Z ^ I ^ Z
        reps = 2
        initial_point = np.random.RandomState(42).random(2 * reps)
        optimizer = {"name": "SPSA", "maxiter": 100}
        backend = QasmSimulatorPy()

        qaoa = QAOAProgram(
            optimizer=optimizer,
            reps=reps,
            initial_point=initial_point,
            backend=backend,
            provider=self.provider,
        )
        result = qaoa.compute_minimum_eigenvalue(operator)

        self.assertIsInstance(result, VQEProgramResult)


if __name__ == "__main__":
    unittest.main()
