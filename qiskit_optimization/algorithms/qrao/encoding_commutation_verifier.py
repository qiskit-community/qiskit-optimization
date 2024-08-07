# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023, 2024.
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

from qiskit.passmanager import BasePassManager
from qiskit.primitives import BaseEstimatorV1, BaseEstimatorV2

from qiskit_optimization.exceptions import QiskitOptimizationError

from .quantum_random_access_encoding import QuantumRandomAccessEncoding


class EncodingCommutationVerifier:
    """Class for verifying that the relaxation commutes with the objective function."""

    def __init__(
        self,
        encoding: QuantumRandomAccessEncoding,
        estimator: BaseEstimatorV1 | BaseEstimatorV2,
        passmanager: BasePassManager | None = None,
    ):
        """
        Args:
            encoding: The encoding to verify.
            estimator: The estimator to use for the verification.
            passmanager: The pass manager to transpile the circuits
        """
        self._encoding = encoding
        self._estimator = estimator
        self._passmanager = passmanager

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
        if self._passmanager:
            encoded_bitstr_qc = self._passmanager.run(encoded_bitstr_qc)

        # Evaluate the original objective function
        problem = encoding.problem
        sense = problem.objective.sense.value
        obj_val = problem.objective.evaluate(dvars) * sense

        # Evaluate the encoded Hamiltonian
        encoded_op = encoding.qubit_op
        offset = encoding.offset

        if isinstance(self._estimator, BaseEstimatorV1):
            job = self._estimator.run([encoded_bitstr_qc], [encoded_op])

            try:
                encoded_obj_val = job.result().values[0] + offset
            except Exception as exc:
                raise QiskitOptimizationError(
                    "The primitive job to verify commutation failed!"
                ) from exc
        else:  # BaseEstimatorV2
            job = self._estimator.run([(encoded_bitstr_qc, encoded_op)])

            try:
                result = job.result()
                encoded_obj_val = result[0].data.evs.item() + offset
            except Exception as exc:
                raise QiskitOptimizationError(
                    "The primitive job to verify commutation failed!"
                ) from exc

        return str_dvars, obj_val, encoded_obj_val
