# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Translator between an Ising Hamiltonian and a quadratic program"""

import math
from typing import Tuple, Union

import numpy as np

from qiskit.opflow import I, ListOp, OperatorBase, PauliOp, PauliSumOp, SummedOp
from qiskit.quantum_info import Pauli
from qiskit_optimization.exceptions import QiskitOptimizationError
from qiskit_optimization.problems.quadratic_program import QuadraticProgram


def to_ising(quad_prog: QuadraticProgram) -> Tuple[OperatorBase, float]:
    """Return the Ising Hamiltonian of this problem.

    Variables are mapped to qubits in the same order, i.e.,
    i-th variable is mapped to i-th qubit.
    See https://github.com/Qiskit/qiskit-terra/issues/1148 for details.

    Returns:
        qubit_op: The qubit operator for the problem
        offset: The constant value in the Ising Hamiltonian.

    Raises:
        QiskitOptimizationError: If an integer variable or a continuous variable exists
            in the problem.
        QiskitOptimizationError: If constraints exist in the problem.
    """
    # if problem has variables that are not binary, raise an error
    if quad_prog.get_num_vars() > quad_prog.get_num_binary_vars():
        raise QiskitOptimizationError(
            "The type of all variables must be binary. "
            "You can use `QuadraticProgramToQubo` converter "
            "to convert integer variables to binary variables. "
            "If the problem contains continuous variables, `to_ising` cannot handle it. "
            "You might be able to solve it with `ADMMOptimizer`."
        )

    # if constraints exist, raise an error
    if quad_prog.linear_constraints or quad_prog.quadratic_constraints:
        raise QiskitOptimizationError(
            "There must be no constraint in the problem. "
            "You can use `QuadraticProgramToQubo` converter "
            "to convert constraints to penalty terms of the objective function."
        )

    # initialize Hamiltonian.
    num_nodes = quad_prog.get_num_vars()
    pauli_list = []
    offset = 0.0
    zero = np.zeros(num_nodes, dtype=bool)

    # set a sign corresponding to a maximized or minimized problem.
    # sign == 1 is for minimized problem. sign == -1 is for maximized problem.
    sense = quad_prog.objective.sense.value

    # convert a constant part of the object function into Hamiltonian.
    offset += quad_prog.objective.constant * sense

    # convert linear parts of the object function into Hamiltonian.
    for idx, coef in quad_prog.objective.linear.to_dict().items():
        z_p = zero.copy()
        weight = coef * sense / 2
        z_p[idx] = True

        pauli_list.append(PauliOp(Pauli((z_p, zero)), -weight))
        offset += weight

    # create Pauli terms
    for (i, j), coeff in quad_prog.objective.quadratic.to_dict().items():
        weight = coeff * sense / 4

        if i == j:
            offset += weight
        else:
            z_p = zero.copy()
            z_p[i] = True
            z_p[j] = True
            pauli_list.append(PauliOp(Pauli((z_p, zero)), weight))

        z_p = zero.copy()
        z_p[i] = True
        pauli_list.append(PauliOp(Pauli((z_p, zero)), -weight))

        z_p = zero.copy()
        z_p[j] = True
        pauli_list.append(PauliOp(Pauli((z_p, zero)), -weight))

        offset += weight

    # Remove paulis whose coefficients are zeros.
    qubit_op = sum(pauli_list)

    # qubit_op could be the integer 0, in this case return an identity operator of
    # appropriate size
    if isinstance(qubit_op, OperatorBase):
        qubit_op = qubit_op.reduce()
    else:
        qubit_op = I ^ num_nodes

    return qubit_op, offset


def from_ising(
    qubit_op: Union[OperatorBase, PauliSumOp],
    offset: float = 0.0,
    linear: bool = False,
) -> QuadraticProgram:
    r"""Create a quadratic program from a qubit operator and a shift value.

    Variables are mapped to qubits in the same order, i.e.,
    i-th variable is mapped to i-th qubit.
    See https://github.com/Qiskit/qiskit-terra/issues/1148 for details.

    Args:
        qubit_op: The qubit operator of the problem.
        offset: The constant term in the Ising Hamiltonian.
        linear: If linear is True, :math:`x^2` is treated as a linear term
            since :math:`x^2 = x` for :math:`x \in \{0,1\}`.
            Otherwise, :math:`x^2` is treat as a quadratic term.
            The default value is False.

    Returns:
        The quadratic program corresponding to the qubit operator.

    Raises:
        QiskitOptimizationError: if there are Pauli Xs or Ys in any Pauli term
        QiskitOptimizationError: if there are more than 2 Pauli Zs in any Pauli term
        QiskitOptimizationError: if any Pauli term has an imaginary coefficient
        NotImplementedError: If the input operator is a ListOp
    """
    if isinstance(qubit_op, PauliSumOp):
        qubit_op = qubit_op.to_pauli_op()

    # No support for ListOp yet, this can be added in future
    # pylint: disable=unidiomatic-typecheck
    if type(qubit_op) == ListOp:
        raise NotImplementedError(
            "Conversion of a ListOp is not supported, convert each "
            "operator in the ListOp separately."
        )

    quad_prog = QuadraticProgram()
    quad_prog.binary_var_list(qubit_op.num_qubits)

    if not isinstance(qubit_op, SummedOp):
        pauli_list = [qubit_op.to_pauli_op()]
    else:
        pauli_list = qubit_op.to_pauli_op()

    # prepare a matrix of coefficients of Pauli terms
    # `pauli_coeffs_diag` is the diagonal part
    # `pauli_coeffs_triu` is the upper triangular part
    pauli_coeffs_diag = [0.0] * qubit_op.num_qubits
    pauli_coeffs_triu = {}

    for pauli_op in pauli_list:
        pauli_op = pauli_op.to_pauli_op()
        pauli = pauli_op.primitive
        coeff = pauli_op.coeff

        if not math.isclose(coeff.imag, 0.0, abs_tol=1e-10):
            raise QiskitOptimizationError(f"Imaginary coefficient exists: {pauli_op}")

        if np.any(pauli.x):
            raise QiskitOptimizationError(f"Pauli X or Y exists in the Pauli term: {pauli}")

        # indices of Pauli Zs in the Pauli term
        z_index = np.where(pauli.z)[0]
        num_z = len(z_index)

        if num_z == 1:
            pauli_coeffs_diag[z_index[0]] = coeff.real
        elif num_z == 2:
            pauli_coeffs_triu[z_index[0], z_index[1]] = coeff.real
        else:
            raise QiskitOptimizationError(
                f"There are more than 2 Pauli Zs in the Pauli term: {pauli}"
            )

    linear_terms = {}
    quadratic_terms = {}

    # For quadratic pauli terms of operator
    # x_i * x_j = (1 - Z_i - Z_j + Z_i * Z_j)/4
    for (i, j), weight in pauli_coeffs_triu.items():
        # Add a quadratic term to the object function of `QuadraticProgram`
        # The coefficient of the quadratic term in `QuadraticProgram` is
        # 4 * weight of the pauli
        quadratic_terms[i, j] = 4 * weight
        pauli_coeffs_diag[i] += weight
        pauli_coeffs_diag[j] += weight
        offset -= weight

    # After processing quadratic pauli terms, only linear paulis are left
    # x_i = (1 - Z_i)/2
    for i, weight in enumerate(pauli_coeffs_diag):
        # Add a linear term to the object function of `QuadraticProgram`
        # The coefficient of the linear term in `QuadraticProgram` is
        # 2 * weight of the pauli
        if linear:
            linear_terms[i] = -2 * weight
        else:
            quadratic_terms[i, i] = -2 * weight
        offset += weight

    quad_prog.minimize(constant=offset, linear=linear_terms, quadratic=quadratic_terms)

    return quad_prog
