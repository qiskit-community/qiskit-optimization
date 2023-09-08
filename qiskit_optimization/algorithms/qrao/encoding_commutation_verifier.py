# This code is part of a Qiskit project.
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

"""The EncodingCommutationVerifier."""

from __future__ import annotations

from qiskit.primitives import BaseEstimator

from qiskit_optimization.exceptions import QiskitOptimizationError

from .quantum_random_access_encoding import QuantumRandomAccessEncoding


class EncodingCommutationVerifier:
    """Class for verifying that the relaxation commutes with the objective function."""

    def __init__(self, encoding: QuantumRandomAccessEncoding, estimator: BaseEstimator):
        """
        Args:
            encoding: The encoding to verify.
            estimator: The estimator to use for the verification.
        """
        self._encoding = encoding
        self._estimator = estimator

    def __len__(self) -> int:
        return 2**self._encoding.num_vars

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i: int) -> tuple[str, float, float]:
        if i < 0 or i >= len(self):
            raise IndexError(f"Index out of range: {i}")

        encoding = self._encoding
        str_dvars = f"{i:0{encoding.num_vars}b}"
        dvars = [int(b) for b in str_dvars]
        encoded_bitstr_qc = encoding.state_preparation_circuit(dvars)

        # Evaluate the original objective function
        problem = encoding.problem
        sense = problem.objective.sense.value
        obj_val = problem.objective.evaluate(dvars) * sense

        # Evaluate the encoded Hamiltonian
        encoded_op = encoding.qubit_op
        offset = encoding.offset

        job = self._estimator.run([encoded_bitstr_qc], [encoded_op])

        try:
            encoded_obj_val = job.result().values[0] + offset
        except Exception as exc:
            raise QiskitOptimizationError(
                "The primitive job to verify commutation failed!"
            ) from exc

        return (str_dvars, obj_val, encoded_obj_val)
