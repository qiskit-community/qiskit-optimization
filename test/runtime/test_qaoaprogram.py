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

import warnings
import unittest
from ddt import ddt, data
import numpy as np
from qiskit.algorithms.optimizers import COBYLA
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit.opflow import I, Z

from qiskit_optimization.runtime import (
    QAOAClient,
    QAOAProgram,
    VQERuntimeResult,
    VQEProgramResult,
)

from .fake_vqeruntime import FakeQAOARuntimeProvider


@ddt
class TestQAOAProgram(QiskitOptimizationTestCase):
    """Test the QAOA program."""

    def setUp(self):
        super().setUp()
        self.provider = FakeQAOARuntimeProvider()

    @data(
        {"name": "SPSA", "maxiter": 100},
        COBYLA(),
    )
    def test_standard_case(self, optimizer):
        """Test a standard use case."""
        operator = Z ^ I ^ Z
        reps = 2
        initial_point = np.random.RandomState(42).random(2 * reps)
        backend = QasmSimulatorPy()

        for use_deprecated in [False, True]:
            if use_deprecated:
                qaoa_cls = QAOAProgram
                result_cls = VQEProgramResult
                warnings.filterwarnings("ignore", category=DeprecationWarning)
            else:
                qaoa_cls = QAOAClient
                result_cls = VQERuntimeResult

            qaoa = qaoa_cls(
                optimizer=optimizer,
                reps=reps,
                initial_point=initial_point,
                backend=backend,
                provider=self.provider,
            )

            if use_deprecated:
                warnings.filterwarnings("always", category=DeprecationWarning)

            result = qaoa.compute_minimum_eigenvalue(operator)

            self.assertIsInstance(result, result_cls)


if __name__ == "__main__":
    unittest.main()
