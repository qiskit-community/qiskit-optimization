# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Magic bases rounding"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numbers
import time
import warnings

import numpy as np

from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.opflow import PrimitiveOp
from qiskit.utils import QuantumInstance

from .encoding import z_to_31p_qrac_basis_circuit, z_to_21p_qrac_basis_circuit
from .rounding_common import (
    RoundingSolutionSample,
    RoundingScheme,
    RoundingContext,
    RoundingResult,
)


_invalid_backend_names = [
    "aer_simulator_unitary",
    "aer_simulator_superop",
    "unitary_simulator",
    "pulse_simulator",
]


def _backend_name(backend: Backend) -> str:
    """Return the backend name in a way that is agnostic to Backend version"""
    # See qiskit.utils.backend_utils in qiskit-terra for similar examples
    if backend.version <= 1:
        return backend.name()
    return backend.name


def _is_original_statevector_simulator(backend: Backend) -> bool:
    """Return True if the original statevector simulator"""
    return _backend_name(backend) == "statevector_simulator"


class MagicRoundingResult(RoundingResult):
    """Result of magic rounding"""

    def __init__(
        self,
        samples: List[RoundingSolutionSample],
        *,
        bases=None,
        basis_shots=None,
        basis_counts=None,
        time_taken=None,
    ):
        self._bases = bases
        self._basis_shots = basis_shots
        self._basis_counts = basis_counts
        super().__init__(samples, time_taken=time_taken)

    @property
    def bases(self):
        return self._bases

    @property
    def basis_shots(self):
        return self._basis_shots

    @property
    def basis_counts(self):
        return self._basis_counts


class MagicRounding(RoundingScheme):
    """ "Magic rounding" method

    This method is described in https://arxiv.org/abs/2111.03167v2.

    """

    _DECODING = {
        3: (  # Eq. (8)
            {"0": [0, 0, 0], "1": [1, 1, 1]},  # I mu+ I, I mu- I
            {"0": [0, 1, 1], "1": [1, 0, 0]},  # X mu+ X, X mu- X
            {"0": [1, 0, 1], "1": [0, 1, 0]},  # Y mu+ Y, Y mu- Y
            {"0": [1, 1, 0], "1": [0, 0, 1]},  # Z mu+ Z, Z mu- Z
        ),
        2: (  # Sec. VII
            {"0": [0, 0], "1": [1, 1]},  # I xi+ I, I xi- I
            {"0": [0, 1], "1": [1, 0]},  # X xi+ X, X xi- X
        ),
        1: ({"0": [0], "1": [1]},),
    }

    # Pauli op string to label index in ops
    _OP_INDICES = {1: {"Z": 0}, 2: {"X": 0, "Z": 1}, 3: {"X": 0, "Y": 1, "Z": 2}}

    def __init__(
        self,
        quantum_instance: QuantumInstance,
        *,
        basis_sampling: str = "uniform",
        seed: Optional[int] = None,
    ):
        """
        Args:

            quantum_instance: Provides the ``Backend`` for quantum execution
                and the ``shots`` count (i.e., the number of samples to collect
                from the magic bases).

            basis_sampling: Method to use for sampling the magic bases.  Must
                be either ``"uniform"`` (default) or ``"weighted"``.
                ``"uniform"`` samples all magic bases uniformly, and is the
                method described in https://arxiv.org/abs/2111.03167v2.
                ``"weighted"`` attempts to choose bases strategically using the
                Pauli expectation values from the minimum eigensolver.
                However, the approximation bounds given in
                https://arxiv.org/abs/2111.03167v2 apply only to ``"uniform"``
                sampling.

            seed: Seed for random number generator, which is used to sample the
                magic bases.

        """
        if basis_sampling not in ("uniform", "weighted"):
            raise ValueError(
                f"'{basis_sampling}' is not an implemented sampling method. "
                "Please choose either 'uniform' or 'weighted'."
            )
        self.quantum_instance = quantum_instance
        self.rng = np.random.RandomState(seed)
        self._basis_sampling = basis_sampling
        super().__init__()

    @property
    def shots(self) -> int:
        """Shots count as configured by the given ``quantum_instance``."""
        return self.quantum_instance.run_config.shots

    @property
    def basis_sampling(self):
        """Basis sampling method (either ``"uniform"`` or ``"weighted"``)."""
        return self._basis_sampling

    @property
    def quantum_instance(self) -> QuantumInstance:
        """Provides the ``Backend`` and the ``shots`` (samples) count."""
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance: QuantumInstance) -> None:
        backend_name = _backend_name(quantum_instance.backend)
        if backend_name in _invalid_backend_names:
            raise ValueError(f"{backend_name} is not supported.")
        if _is_original_statevector_simulator(quantum_instance.backend):
            warnings.warn(
                'Use of "statevector_simulator" is discouraged because it effectively '
                "brute-forces all possible solutions.  We suggest using the newer "
                '"aer_simulator_statevector" instead.'
            )
        self._quantum_instance = quantum_instance

    def _unpack_measurement_outcome(
        self,
        bits: str,
        basis: List[int],
        var2op: Dict[int, Tuple[int, PrimitiveOp]],
        vars_per_qubit: int,
    ) -> List[int]:
        output_bits = []
        # iterate in order over decision variables
        # (assumes variables are numbered consecutively beginning with 0)
        for var in range(len(var2op)):  # pylint: disable=consider-using-enumerate
            q, op = var2op[var]
            # get the decoding outcome index for the variable
            # corresponding to this Pauli op.
            op_index = self._OP_INDICES[vars_per_qubit][str(op)]
            # get the bits associated to this magic basis'
            # measurement outcomes
            bit_outcomes = self._DECODING[vars_per_qubit][basis[q]]
            # select which measurement outcome we observed
            # this gives up to 3 bits of information
            magic_bits = bit_outcomes[bits[q]]
            # Assign our variable's value depending on
            # which pauli our variable was associated to
            variable_value = magic_bits[op_index]
            output_bits.append(variable_value)
        return output_bits

    @staticmethod
    def _make_circuits(
        circ: QuantumCircuit, bases: List[List[int]], measure: bool, vars_per_qubit: int
    ) -> List[QuantumCircuit]:
        circuits = []
        for basis in bases:
            if vars_per_qubit == 3:
                qc = circ.compose(
                    z_to_31p_qrac_basis_circuit(basis).inverse(), inplace=False
                )
            elif vars_per_qubit == 2:
                qc = circ.compose(
                    z_to_21p_qrac_basis_circuit(basis).inverse(), inplace=False
                )
            elif vars_per_qubit == 1:
                qc = circ.copy()
            if measure:
                qc.measure_all()
            circuits.append(qc)
        return circuits

    def _evaluate_magic_bases(self, circuit, bases, basis_shots, vars_per_qubit):
        """
        Given a circuit you wish to measure, a list of magic bases to measure,
        and a list of the shots to use for each magic basis configuration.

        Measure the provided circuit in the magic bases given and return the counts
        dictionaries associated with each basis measurement.

        len(bases) == len(basis_shots) == len(basis_counts)
        """
        measure = not _is_original_statevector_simulator(self.quantum_instance.backend)
        circuits = self._make_circuits(circuit, bases, measure, vars_per_qubit)

        # Execute each of the rotated circuits and collect the results

        # Batch the circuits into jobs where each group has the same number of
        # shots, so that you can wait for the queue as few times as possible if
        # using hardware.
        circuit_indices_by_shots: Dict[int, List[int]] = defaultdict(list)
        assert len(circuits) == len(basis_shots)
        for i, shots in enumerate(basis_shots):
            circuit_indices_by_shots[shots].append(i)

        basis_counts: List[Optional[Dict[str, int]]] = [None] * len(circuits)
        overall_shots = self.quantum_instance.run_config.shots
        try:
            for shots, indices in sorted(
                circuit_indices_by_shots.items(), reverse=True
            ):
                self.quantum_instance.set_config(shots=shots)
                result = self.quantum_instance.execute([circuits[i] for i in indices])
                counts_list = result.get_counts()
                if not isinstance(counts_list, List):
                    # This is the only case where this should happen, and that
                    # it does at all (namely, when a single-element circuit
                    # list is provided) is a weird API quirk of Qiskit.
                    # https://github.com/Qiskit/qiskit-terra/issues/8103
                    assert len(indices) == 1
                    counts_list = [counts_list]
                assert len(indices) == len(counts_list)
                for i, counts in zip(indices, counts_list):
                    basis_counts[i] = counts
        finally:
            # We've temporarily modified quantum_instance; now we restore it to
            # its initial state.
            self.quantum_instance.set_config(shots=overall_shots)
        assert None not in basis_counts

        # Process the outcomes and extract expectation of decision vars

        # The "statevector_simulator", unlike all the others, returns
        # probabilities instead of integer counts.  So if probabilities are
        # detected, we rescale them.
        if any(
            any(not isinstance(x, numbers.Integral) for x in counts.values())
            for counts in basis_counts
        ):
            basis_counts = [
                {key: val * basis_shots[i] for key, val in counts.items()}
                for i, counts in enumerate(basis_counts)
            ]

        return basis_counts

    def _compute_dv_counts(self, basis_counts, bases, var2op, vars_per_qubit):
        """
        Given a list of bases, basis_shots, and basis_counts, convert
        each observed bitstrings to its corresponding decision variable
        configuration. Return the counts of each decision variable configuration.
        """
        dv_counts = {}
        for i, counts in enumerate(basis_counts):
            base = bases[i]
            # For each measurement outcome...
            for bitstr, count in counts.items():

                # For each bit in the observed bitstring...
                soln = self._unpack_measurement_outcome(
                    bitstr, base, var2op, vars_per_qubit
                )
                soln = "".join([str(int(bit)) for bit in soln])
                if soln in dv_counts:
                    dv_counts[soln] += count
                else:
                    dv_counts[soln] = count
        return dv_counts

    def _sample_bases_uniform(self, q2vars, vars_per_qubit):
        bases = [
            self.rng.choice(2 ** (vars_per_qubit - 1), size=len(q2vars)).tolist()
            for _ in range(self.shots)
        ]
        bases, basis_shots = np.unique(bases, axis=0, return_counts=True)
        return bases, basis_shots

    def _sample_bases_weighted(self, q2vars, trace_values, vars_per_qubit):
        """Perform weighted sampling from the expectation values.

        The goal is to make smarter choices about which bases to measure in
        using the trace values.
        """
        # First, we make sure all Pauli expectation values have absolute value
        # at most 1.  Otherwise, some of the probabilities computed below might
        # be negative.
        tv = np.clip(trace_values, -1, 1)
        # basis_probs will have num_qubits number of elements.
        # Each element will be a list of length 4 specifying the
        # probability of picking the corresponding magic basis on that qubit.
        basis_probs = []
        for dvars in q2vars:
            if vars_per_qubit == 3:
                x = 0.5 * (1 - tv[dvars[0]])
                y = 0.5 * (1 - tv[dvars[1]]) if (len(dvars) > 1) else 0
                z = 0.5 * (1 - tv[dvars[2]]) if (len(dvars) > 2) else 0
                # ppp:   mu±   = .5(I ± 1/sqrt(3)( X + Y + Z))
                # pmm: X mu± X = .5(I ± 1/sqrt(3)( X - Y - Z))
                # mpm: Y mu± Y = .5(I ± 1/sqrt(3)(-X + Y - Z))
                # mmp: Z mu± Z = .5(I ± 1/sqrt(3)(-X - Y + Z))
                # fmt: off
                ppp_mmm =   x   *   y   *   z   + (1-x) * (1-y) * (1-z)
                pmm_mpp =   x   * (1-y) * (1-z) + (1-x) *   y   *   z
                mpm_pmp = (1-x) *   y   * (1-z) +   x   * (1-y) *   z
                ppm_mmp =   x   *   y   * (1-z) + (1-x) * (1-y) *   z
                # fmt: on
                basis_probs.append([ppp_mmm, pmm_mpp, mpm_pmp, ppm_mmp])
            elif vars_per_qubit == 2:
                x = 0.5 * (1 - tv[dvars[0]])
                z = 0.5 * (1 - tv[dvars[1]]) if (len(dvars) > 1) else 0
                # pp:   xi±   = .5(I ± 1/sqrt(2)( X + Z ))
                # pm: X xi± X = .5(I ± 1/sqrt(2)( X - Z ))
                # fmt: off
                pp_mm =   x   *   z   + (1-x) * (1-z)
                pm_mp =   x   * (1-z) + (1-x) *   z
                # fmt: on
                basis_probs.append([pp_mm, pm_mp])
            elif vars_per_qubit == 1:
                basis_probs.append([1.0])
        bases = [
            [
                self.rng.choice(2 ** (vars_per_qubit - 1), p=probs)
                for probs in basis_probs
            ]
            for _ in range(self.shots)
        ]
        bases, basis_shots = np.unique(bases, axis=0, return_counts=True)
        return bases, basis_shots

    def round(self, ctx: RoundingContext) -> MagicRoundingResult:
        """Perform magic rounding"""

        start_time = time.time()
        trace_values = ctx.trace_values
        circuit = ctx.circuit

        if circuit is None:
            raise NotImplementedError(
                "Magic rounding requires a circuit to be available.  Perhaps try "
                "semideterministic rounding instead."
            )

        # We've already checked that it is one of these two in the constructor
        if self.basis_sampling == "uniform":
            bases, basis_shots = self._sample_bases_uniform(
                ctx.q2vars, ctx._vars_per_qubit
            )
        elif self.basis_sampling == "weighted":
            if trace_values is None:
                raise NotImplementedError(
                    "Magic rounding with weighted sampling requires the trace values "
                    "to be available, but they are not."
                )
            bases, basis_shots = self._sample_bases_weighted(
                ctx.q2vars, trace_values, ctx._vars_per_qubit
            )
        else:  # pragma: no cover
            raise NotImplementedError(
                f'No such basis sampling method: "{self.basis_sampling}".'
            )

        assert self.shots == np.sum(basis_shots)
        # For each of the Magic Bases sampled above, measure
        # the appropriate number of times (given by basis_shots)
        # and return the circuit results

        basis_counts = self._evaluate_magic_bases(
            circuit, bases, basis_shots, ctx._vars_per_qubit
        )
        # keys will be configurations of decision variables
        # values will be total number of observations.
        soln_counts = self._compute_dv_counts(
            basis_counts, bases, ctx.var2op, ctx._vars_per_qubit
        )

        soln_samples = [
            RoundingSolutionSample(
                x=np.asarray([int(bit) for bit in soln]),
                probability=count / self.shots,
            )
            for soln, count in soln_counts.items()
        ]

        assert np.isclose(
            sum(soln_counts.values()), self.shots
        ), f"{sum(soln_counts.values())} != {self.shots}"
        assert len(bases) == len(basis_shots) == len(basis_counts)
        stop_time = time.time()

        return MagicRoundingResult(
            samples=soln_samples,
            bases=bases,
            basis_shots=basis_shots,
            basis_counts=basis_counts,
            time_taken=stop_time - start_time,
        )
