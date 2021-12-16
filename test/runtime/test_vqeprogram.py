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

"""Test the VQE program."""

from test import QiskitOptimizationTestCase

import unittest
import warnings
from ddt import ddt, data
import numpy as np
from qiskit.algorithms.optimizers import COBYLA
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import I, Z

from qiskit_optimization.runtime import (
    VQEClient,
    VQERuntimeResult,
    VQEProgram,
    VQEProgramResult,
)

from .fake_vqeruntime import FakeVQERuntimeProvider


@ddt
class TestVQEProgram(QiskitOptimizationTestCase):
    """Test the VQE program."""

    def setUp(self):
        super().setUp()
        self.provider = FakeVQERuntimeProvider()

    @data(
        {"name": "SPSA", "maxiter": 100},
        COBYLA(),
    )
    def test_standard_case(self, optimizer):
        """Test a standard use case."""
        circuit = RealAmplitudes(3)
        operator = Z ^ I ^ Z
        initial_point = np.random.RandomState(42).random(circuit.num_parameters)
        backend = QasmSimulatorPy()

        for use_deprecated in [False, True]:
            if use_deprecated:
                vqe_cls = VQEProgram
                result_cls = VQEProgramResult
                warnings.filterwarnings("ignore", category=DeprecationWarning)
            else:
                vqe_cls = VQEClient
                result_cls = VQERuntimeResult

            vqe = vqe_cls(
                ansatz=circuit,
                optimizer=optimizer,
                initial_point=initial_point,
                backend=backend,
                provider=self.provider,
            )

            if use_deprecated:
                warnings.filterwarnings("always", category=DeprecationWarning)

            result = vqe.compute_minimum_eigenvalue(operator)

            self.assertIsInstance(result, result_cls)


if __name__ == "__main__":
    unittest.main()
