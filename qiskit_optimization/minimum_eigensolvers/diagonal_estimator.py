# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Expectation value for a diagonal observable using a sampler primitive."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping, MappingView, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseEstimatorV1, BaseSamplerV1, BaseSamplerV2, EstimatorResult
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

from ..utils.primitives import _circuit_key, _init_observable
from .algorithm_job import AlgorithmJob


@dataclass(frozen=True)
class _DiagonalEstimatorResult(EstimatorResult):
    """A result from an expectation of a diagonal observable."""

    # TODO make each measurement a dataclass rather than a dict
    best_measurements: Sequence[Mapping[str, Any]] | None = None


class _DiagonalEstimator(BaseEstimatorV1):
    """An estimator for diagonal observables."""

    # TODO: _DiagonalEstimator should be updated to inherit BaseEstimatorV2

    def __init__(
        self,
        sampler: BaseSamplerV1 | BaseSamplerV2,
        aggregation: float | Callable[[Iterable[tuple[float, float]]], float] | None = None,
        callback: Callable[[Sequence[Mapping[str, Any]]], None] | None = None,
        **options,
    ) -> None:
        r"""Evaluate the expectation of quantum state with respect to a diagonal operator.

        Args:
            sampler: The sampler used to evaluate the circuits.
            aggregation: The aggregation function to aggregate the measurement outcomes. If a float
                this specified the CVaR :math:`\alpha` parameter.
            callback: A callback which is given the best measurements of all circuits in each
                evaluation.
            run_options: Options for the sampler.

        """
        super().__init__(options=options)
        self._circuits: list[QuantumCircuit] = []
        self._parameters: list[MappingView] = []
        self._observables: list[SparsePauliOp] = []

        self.sampler = sampler
        if not callable(aggregation):
            aggregation = _get_cvar_aggregation(aggregation)

        self.aggregation = aggregation
        self.callback = callback
        self._circuit_ids: dict[tuple, QuantumCircuit] = {}
        self._observable_ids: dict[int, BaseOperator] = {}

        if isinstance(sampler, BaseSamplerV1):
            warnings.warn(
                "Using Sampler V1 is deprecated since 0.7.0. Instead use Sampler V2.",
                category=DeprecationWarning,
                stacklevel=2,
            )

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> AlgorithmJob:
        circuit_indices = []
        for circuit in circuits:
            key = _circuit_key(circuit)
            index = self._circuit_ids.get(key)
            if index is not None:
                circuit_indices.append(index)
            else:
                circuit_indices.append(len(self._circuits))
                self._circuit_ids[key] = len(self._circuits)
                self._circuits.append(circuit)
                self._parameters.append(circuit.parameters)
        observable_indices = []
        for observable in observables:
            index = self._observable_ids.get(id(observable))
            if index is not None:
                observable_indices.append(index)
            else:
                observable_indices.append(len(self._observables))
                self._observable_ids[id(observable)] = len(self._observables)
                converted_observable = _init_observable(observable)
                _check_observable_is_diagonal(converted_observable)  # check it's diagonal
                self._observables.append(converted_observable)
        job = AlgorithmJob(
            self._call, circuit_indices, observable_indices, parameter_values, **run_options
        )
        job.submit()
        return job

    def _call(
        self,
        circuits: Sequence[int],
        observables: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> _DiagonalEstimatorResult:
        if isinstance(self.sampler, BaseSamplerV1):
            job = self.sampler.run(
                [self._circuits[i] for i in circuits],
                parameter_values,
                **run_options,
            )
            sampler_result = job.result()
            metadata = sampler_result.metadata
            samples = sampler_result.quasi_dists
        else:  # BaseSamplerV2
            job = self.sampler.run(
                [(self._circuits[i], val) for i, val in zip(circuits, parameter_values)],
                **run_options,
            )
            sampler_pub_result = job.result()
            metadata = []
            samples = []
            for i, result in zip(circuits, sampler_pub_result):
                creg = self._circuits[i].cregs[0].name
                counts = getattr(result.data, creg).get_int_counts()
                shots = sum(counts.values())
                samples.append({key: val / shots for key, val in counts.items()})
                metadata.append(result.metadata)

        # a list of dictionaries containing: {state: (measurement probability, value)}
        evaluations: list[dict[int, tuple[float, float]]] = [
            {
                state: (probability, _evaluate_sparsepauli(state, self._observables[i]))
                for state, probability in sampled.items()
            }
            for i, sampled in zip(observables, samples)
        ]

        results = np.array([self.aggregation(evaluated.values()) for evaluated in evaluations])

        # get the best measurements
        best_measurements = []
        num_qubits = self._circuits[0].num_qubits
        for evaluated in evaluations:
            best_result = min(evaluated.items(), key=lambda x: x[1][1])
            best_measurements.append(
                {
                    "state": best_result[0],
                    "bitstring": bin(best_result[0])[2:].zfill(num_qubits),
                    "value": best_result[1][1],
                    "probability": best_result[1][0],
                }
            )

        if self.callback is not None:
            self.callback(best_measurements)

        return _DiagonalEstimatorResult(
            values=results, metadata=metadata, best_measurements=best_measurements
        )


def _get_cvar_aggregation(alpha: float | None) -> Callable[[Iterable[tuple[float, float]]], float]:
    """Get the aggregation function for CVaR with confidence level ``alpha``."""
    if alpha is None:
        alpha = 1
    elif not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1] but was {alpha}")

    # if alpha is close to 1 we can avoid the sorting
    if np.isclose(alpha, 1):

        def aggregate(measurements: Iterable[tuple[float, float]]) -> float:
            return sum(probability * value for probability, value in measurements)

    else:

        def aggregate(measurements: Iterable[tuple[float, float]]) -> float:
            # sort by values
            sorted_measurements = sorted(measurements, key=lambda x: x[1])

            accumulated_percent = 0.0  # once alpha is reached, stop
            cvar = 0.0
            for probability, value in sorted_measurements:
                cvar += value * min(probability, alpha - accumulated_percent)
                accumulated_percent += probability
                if accumulated_percent >= alpha:
                    break

            return cvar / alpha

    return aggregate


_PARITY = np.array([-1 if bin(i).count("1") % 2 else 1 for i in range(256)], dtype=np.complex128)


def _evaluate_sparsepauli(state: int, observable: SparsePauliOp) -> float:
    packed_uint8 = np.packbits(observable.paulis.z, axis=1, bitorder="little")
    state_bytes = np.frombuffer(state.to_bytes(packed_uint8.shape[1], "little"), dtype=np.uint8)
    reduced = np.bitwise_xor.reduce(packed_uint8 & state_bytes, axis=1)
    return np.sum(observable.coeffs * _PARITY[reduced])


def _check_observable_is_diagonal(observable: SparsePauliOp) -> None:
    is_diagonal = not np.any(observable.paulis.x)
    if not is_diagonal:
        raise ValueError("The observable must be diagonal.")
