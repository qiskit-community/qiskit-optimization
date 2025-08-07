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

"""Test Grover's algorithm."""

import itertools
from test import QiskitAlgorithmsTestCase

import numpy as np
from ddt import data, ddt
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator
from qiskit.quantum_info import Operator, Statevector

from qiskit_optimization.amplitude_amplifiers import AmplificationProblem


@ddt
class TestAmplificationProblem(QiskitAlgorithmsTestCase):
    """Test the amplification problem."""

    def setUp(self):
        super().setUp()
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        self._expected_grover_op = GroverOperator(oracle=oracle)

    @data("oracle_only", "oracle_and_stateprep")
    def test_groverop_getter(self, kind):
        """Test the default construction of the Grover operator."""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)

        if kind == "oracle_only":
            problem = AmplificationProblem(oracle, is_good_state=["11"])
            expected = GroverOperator(oracle)
        else:
            stateprep = QuantumCircuit(2)
            stateprep.ry(0.2, [0, 1])
            problem = AmplificationProblem(
                oracle, state_preparation=stateprep, is_good_state=["11"]
            )
            expected = GroverOperator(oracle, stateprep)

        self.assertEqual(Operator(expected), Operator(problem.grover_operator))

    @data("list_str", "list_int", "statevector", "callable")
    def test_is_good_state(self, kind):
        """Test is_good_state works on different input types."""
        if kind == "list_str":
            is_good_state = ["01", "11"]
        elif kind == "list_int":
            is_good_state = [1]  # means bitstr[1] == '1'
        elif kind == "statevector":
            is_good_state = Statevector(np.array([0, 1, 0, 1]) / np.sqrt(2))
        else:

            def is_good_state(bitstr):
                # same as ``bitstr in ['01', '11']``
                return bitstr[1] == "1"

        possible_states = [
            "".join(list(map(str, item))) for item in itertools.product([0, 1], repeat=2)
        ]

        oracle = QuantumCircuit(2)
        problem = AmplificationProblem(oracle, is_good_state=is_good_state)

        expected = [state in ["01", "11"] for state in possible_states]
        actual = [problem.is_good_state(state) for state in possible_states]

        self.assertListEqual(expected, actual)
