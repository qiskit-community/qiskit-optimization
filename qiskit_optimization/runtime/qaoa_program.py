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
    """DEPRECATED. This class has been renamed to ``qiskit_optimization.runtime.QAOAClient``.

    This renaming reflects that this class is a client for a program executed in the cloud.
    """

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
        warn_deprecated(
            version="0.3.0",
            old_type=DeprecatedType.CLASS,
            old_name="QAOAProgram",
            new_name="QAOAClient",
            additional_msg="from qiskit_optimization.runtime",
        )

        super().__init__(
            optimizer,
            reps,
            initial_state,
            mixer,
            initial_point,
            provider,
            backend,
            shots,
            measurement_error_mitigation,
            callback,
            store_intermediate,
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
