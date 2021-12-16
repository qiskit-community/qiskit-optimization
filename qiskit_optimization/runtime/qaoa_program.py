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
import warnings
import numpy as np

from qiskit import QuantumCircuit
from qiskit.providers import Provider
from qiskit.providers.backend import Backend
from qiskit.algorithms import MinimumEigensolverResult
from qiskit.algorithms.optimizers import Optimizer
from qiskit.opflow import OperatorBase

from ..deprecation import warn_deprecated, DeprecatedType

from .qaoa_client import QAOAClient
from .vqe_program import VQEProgramResult


class QAOAProgram(QAOAClient):
    """DEPRECATED. This class has been renamed to ``qiskit_optimization.runtime.QAOAClient``."""

    def __init__(
        self,
        optimizer: Optional[Union[Optimizer, Dict[str, Any]]] = None,
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
        warn_deprecated(
            version="0.3.0",
            old_type=DeprecatedType.CLASS,
            old_name="QAOAProgram",
            new_name="QAOAClient",
            additional_msg="from qiskit_optimization.runtime",
        )

        super().__init__(
            optimizer=optimizer,
            reps=reps,
            initial_state=initial_state,
            mixer=mixer,
            initial_point=initial_point,
            provider=provider,
            backend=backend,
            shots=shots,
            measurement_error_mitigation=measurement_error_mitigation,
            callback=callback,
            store_intermediate=store_intermediate,
        )

    def compute_minimum_eigenvalue(
        self, operator: OperatorBase, aux_operators: Optional[List[Optional[OperatorBase]]] = None
    ) -> MinimumEigensolverResult:
        result = super().compute_minimum_eigenvalue(operator, aux_operators)

        # convert to previous result type
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            vqe_result = VQEProgramResult()

        vqe_result.combine(result)
        return vqe_result
