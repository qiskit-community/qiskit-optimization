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
"""
Utility functions for primitives
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import PauliList, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.symplectic.base_pauli import BasePauli
from qiskit.transpiler import TranspileLayout

from qiskit_optimization import QiskitOptimizationError


def _init_observable(observable: BaseOperator | str) -> SparsePauliOp:
    """Initialize observable by converting the input to a :class:`~qiskit.quantum_info.SparsePauliOp`.

    Args:
        observable: The observable.

    Returns:
        The observable as :class:`~qiskit.quantum_info.SparsePauliOp`.

    Raises:
        QiskitError: when observable type cannot be converted to SparsePauliOp.
    """

    if isinstance(observable, SparsePauliOp):
        return observable
    elif isinstance(observable, BaseOperator) and not isinstance(observable, BasePauli):
        raise QiskitError(f"observable type not supported: {type(observable)}")
    else:
        if isinstance(observable, PauliList):
            raise QiskitError(f"observable type not supported: {type(observable)}")
        return SparsePauliOp(observable)


def _bits_key(bits: tuple, circuit: QuantumCircuit) -> tuple:
    return tuple(
        (
            circuit.find_bit(bit).index,
            tuple((reg[0].size, reg[0].name, reg[1]) for reg in circuit.find_bit(bit).registers),
        )
        for bit in bits
    )


def _format_params(param):
    if isinstance(param, np.ndarray):
        return param.data.tobytes()
    elif isinstance(param, QuantumCircuit):
        return _circuit_key(param)
    elif isinstance(param, Iterable):
        return tuple(param)
    return param


def _circuit_key(circuit: QuantumCircuit, functional: bool = True) -> tuple:
    """Private key function for QuantumCircuit.

    This is the workaround until :meth:`QuantumCircuit.__hash__` will be introduced.
    If key collision is found, please add elements to avoid it.

    Args:
        circuit: Input quantum circuit.
        functional: If True, the returned key only includes functional data (i.e. execution related).

    Returns:
        Composite key for circuit.
    """
    functional_key: tuple = (
        circuit.num_qubits,
        circuit.num_clbits,
        circuit.num_parameters,
        tuple(  # circuit.data
            (
                _bits_key(data.qubits, circuit),  # qubits
                _bits_key(data.clbits, circuit),  # clbits
                data.operation.name,  # operation.name
                tuple(_format_params(param) for param in data.operation.params),  # operation.params
            )
            for data in circuit.data
        ),
        None if circuit._op_start_times is None else tuple(circuit._op_start_times),
    )
    if functional:
        return functional_key
    return (
        circuit.name,
        *functional_key,
    )


def _apply_layout(
    operators: list[SparsePauliOp | int] | dict[str, SparsePauliOp | int], layout: TranspileLayout
) -> list[SparsePauliOp] | dict[str, SparsePauliOp]:
    op_list = operators if isinstance(operators, list) else operators.values()
    if any(op == 0 for op in op_list):
        raise QiskitOptimizationError("Zero operator is not supported since Qiskit 2.1.0")

    if isinstance(operators, list):
        return [op.apply_layout(layout) for op in operators]
    else:
        op_dict: dict[str, SparsePauliOp] = operators
        return {key: op.apply_layout(layout) for key, op in op_dict.items()}
