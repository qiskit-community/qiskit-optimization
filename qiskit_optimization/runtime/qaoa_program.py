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

"""The Qiskit Optimization QAOA Quantum Program."""


from typing import List, Callable, Optional, Any, Dict, Union
import numpy as np

from qiskit import QuantumCircuit
from qiskit.algorithms import MinimumEigensolverResult
from qiskit.circuit.library import QAOAAnsatz
from qiskit.opflow import OperatorBase
from qiskit.providers import Provider
from qiskit.providers.backend import Backend

from qiskit_optimization.exceptions import QiskitOptimizationError
from .vqe_program import VQEProgram


class QAOAProgram(VQEProgram):
    """The Qiskit Optimization QAOA Quantum Program."""

    def __init__(
        self,
        optimizer: Optional[Dict[str, Any]] = None,
        reps: int = 1,
        initial_state: Optional[QuantumCircuit] = None,
        mixer: Union[QuantumCircuit, OperatorBase] = None,
        initial_point: Optional[np.ndarray] = None,
        provider: Optional[Provider] = None,
        backend: Optional[Backend] = None,
        shots: int = 1024,
        measurement_error_mitigation: bool = False,
        callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
        store_intermediate: bool = False,
    ) -> None:
        """
        Args:
            optimizer: A dictionary specifying a classical optimizer.
                Currently only SPSA and QN-SPSA are supported. Per default, SPSA is used.
                The dictionary must contain a key ``name`` for the name of the optimizer and may
                contain additional keys for the settings.
                E.g. ``{'name': 'SPSA', 'maxiter': 100}``.
            reps: the integer parameter :math:`p` as specified in https://arxiv.org/abs/1411.4028,
                Has a minimum valid value of 1.
            initial_state: An optional initial state to prepend the QAOA circuit with
            mixer: the mixer Hamiltonian to evolve with or a custom quantum circuit. Allows support
                of optimizations in constrained subspaces as per https://arxiv.org/abs/1709.03489
                as well as warm-starting the optimization as introduced
                in http://arxiv.org/abs/2009.10095.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` a random vector is used.
            provider: The provider.
            backend: The backend to run the circuits on.
            shots: The number of shots to be used
            measurement_error_mitigation: Whether or not to use measurement error mitigation.
            callback: a callback that can access the intermediate data during the optimization.
                Four parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the optimizer parameters for the
                ansatz, the evaluated mean and the evaluated standard deviation.
            store_intermediate: Whether or not to store intermediate values of the optimization
                steps. Per default False.
        """
        super().__init__(
            ansatz=None,
            optimizer=optimizer,
            initial_point=initial_point,
            provider=provider,
            backend=backend,
            shots=shots,
            measurement_error_mitigation=measurement_error_mitigation,
            callback=callback,
            store_intermediate=store_intermediate,
        )
        self._initial_state = initial_state
        self._mixer = mixer
        self._reps = reps

    @property
    def ansatz(self) -> Optional[QuantumCircuit]:
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: QuantumCircuit) -> None:
        raise QiskitOptimizationError(
            "Cannot set the ansatz for QAOA, it is directly inferred from "
            "the problem Hamiltonian."
        )

    @property
    def initial_state(self) -> Optional[QuantumCircuit]:
        """
        Returns:
            Returns the initial state.
        """
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state: Optional[QuantumCircuit]) -> None:
        """
        Args:
            initial_state: Initial state to set.
        """
        self._initial_state = initial_state

    @property
    def mixer(self) -> Union[QuantumCircuit, OperatorBase]:
        """
        Returns:
            Returns the mixer.
        """
        return self._mixer

    @mixer.setter
    def mixer(self, mixer: Union[QuantumCircuit, OperatorBase]) -> None:
        """
        Args:
            mixer: Mixer to set.
        """
        self._mixer = mixer

    @property
    def reps(self) -> int:
        """
        Returns:
            Returns the reps.
        """
        return self._reps

    @reps.setter
    def reps(self, reps: int) -> None:
        """
        Args:
            reps: The new number of reps.
        """
        self._reps = reps

    def compute_minimum_eigenvalue(
        self,
        operator: OperatorBase,
        aux_operators: Optional[List[Optional[OperatorBase]]] = None,
    ) -> MinimumEigensolverResult:
        self._ansatz = QAOAAnsatz(
            operator,
            reps=self.reps,
            initial_state=self.initial_state,
            mixer_operator=self.mixer,
        )
        return super().compute_minimum_eigenvalue(operator, aux_operators)
