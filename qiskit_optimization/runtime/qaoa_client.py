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
from qiskit.algorithms.optimizers import Optimizer
from qiskit.opflow import OperatorBase
from qiskit.providers import Provider
from qiskit.providers.backend import Backend

from qiskit_optimization.exceptions import QiskitOptimizationError
from .vqe_client import VQEClient


class QAOAClient(VQEClient):
    """The Qiskit Optimization QAOA Runtime Client."""

    def __init__(
        self,
        optimizer: Optional[Union[Optimizer, Dict[str, Any]]] = None,
        reps: int = 1,
        initial_state: Optional[QuantumCircuit] = None,
        mixer: Union[QuantumCircuit, OperatorBase] = None,
        initial_point: Optional[np.ndarray] = None,
        alpha: float = 1.0,
        provider: Optional[Provider] = None,
        backend: Optional[Backend] = None,
        shots: int = 1024,
        measurement_error_mitigation: bool = False,
        callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
        store_intermediate: bool = False,
        use_swap_strategies: bool = False,
        use_initial_mapping: bool = False,
        use_pulse_efficient: bool = False,
        optimization_level: Optional[int] = None,
    ) -> None:
        """
        Args:
            optimizer: An optimizer or dictionary specifying a classical optimizer.
                If a dictionary, only SPSA and QN-SPSA are supported. The dictionary must contain a
                key ``name`` for the name of the optimizer and may contain additional keys for the
                settings. E.g. ``{'name': 'SPSA', 'maxiter': 100}``.
                Per default, SPSA is used.
            reps: the integer parameter :math:`p` as specified in https://arxiv.org/abs/1411.4028,
                Has a minimum valid value of 1.
            initial_state: An optional initial state to prepend the QAOA circuit with
            mixer: the mixer Hamiltonian to evolve with or a custom quantum circuit. Allows support
                of optimizations in constrained subspaces as per https://arxiv.org/abs/1709.03489
                as well as warm-starting the optimization as introduced
                in http://arxiv.org/abs/2009.10095.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` a random vector is used.
            alpha: The fraction of top measurement samples to be used for the expectation value
                (CVaR expectation). Defaults to 1, i.e. using all samples to construct the
                expectation value.
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
            use_swap_strategies: A boolean on whether or not to use swap strategies when
                transpiling. If this is False then the standard transpiler with the given
                optimization level will run.
            use_initial_mapping: A boolean flag that, if set to True (the default is False), runs
                a heuristic algorithm to permute the Paulis in the cost operator to better fit the
                coupling map and the swap strategy. This is only needed when the optimization
                problem is sparse and when using swap strategies to transpile.
            use_pulse_efficient: A boolean on whether or not to use a pulse-efficient transpilation.
                If this flag is set to False by default.
            optimization_level: The transpiler optimization level to run if the swap strategies are
                not used. This value defaults to 1 in the QAOA runtime.

        Raises:
            QiskitOptimizationError: if optimization_level is not None and use_swap_strategies
                is True.
        """
        if initial_point is None:
            initial_point = np.random.rand(2 * reps)

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
        self.use_swap_strategies = use_swap_strategies
        self.use_initial_mapping = use_initial_mapping
        self.use_pulse_efficient = use_pulse_efficient
        self._alpha = alpha
        self._program_id = "qaoa"

        # Use the setter to check consistency with other settings.
        self._optimization_level = None
        self.optimization_level = optimization_level

    @property
    def optimization_level(self) -> int:
        """Return the optimization level to use if swap strategies are not used."""
        return self._optimization_level

    @optimization_level.setter
    def optimization_level(self, optimization_level: int):
        """Set the optimization level."""
        if optimization_level is not None and self.use_swap_strategies:
            raise QiskitOptimizationError(
                "optimization_level is not needed when use_swap_strategies is True."
            )

        self._optimization_level = optimization_level

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

    def _send_job(self, inputs: Dict[str, Any], options: Dict[str, Any]):
        """Internal helper function that sub-classes can overwrite."""
        return self.provider.runtime.run(
            program_id=self.program_id,
            inputs=inputs,
            options=options,
        )

    def program_inputs(
        self, operator: OperatorBase, aux_operators: Optional[List[Optional[OperatorBase]]] = None
    ) -> Dict[str, Any]:
        """Return the QAOA program inputs"""
        inputs = super().program_inputs(operator, aux_operators)

        # QAOA doesn't set the ansatz
        del inputs["ansatz"]

        inputs.update(
            {
                "reps": self._reps,
                "pulse_efficient": self._use_pulse_efficient,
                "swap_strategies": self._use_swap_strategies,
                "use_initial_mapping": self._use_initial_mapping,
                "alpha": self._alpha,
            }
        )

        return inputs
