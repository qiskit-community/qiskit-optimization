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

from typing import TYPE_CHECKING, Any, Dict, Tuple, Union

from numpy import zeros
from qiskit.opflow import I, ListOp, OperatorBase, PauliOp, PauliSumOp, SummedOp
from qiskit.quantum_info import Pauli

from qiskit_optimization.exceptions import QiskitOptimizationError

if TYPE_CHECKING:
    # type hint for mypy
    # pylint: disable=cyclic-import
    from qiskit_optimization.problems.quadratic_program import QuadraticProgram
else:
    # type hint for sphinx
    QuadraticProgram = Any


def to_ising(quad_prog: "QuadraticProgram") -> Tuple[OperatorBase, float]:
    """Return the Ising Hamiltonian of this problem.

    Variables are mapped to qubits in the same order, i.e.,
    i-th variable is mapped to i-th qubit.
    See https://github.com/Qiskit/qiskit-terra/issues/1148 for details.

    Returns:
        qubit_op: The qubit operator for the problem
        offset: The constant value in the Ising Hamiltonian.

    Raises:
        QiskitOptimizationError: If a variable type is not binary.
        QiskitOptimizationError: If constraints exist in the problem.
    """
    # if problem has variables that are not binary, raise an error
    if quad_prog.get_num_vars() > quad_prog.get_num_binary_vars():
        raise QiskitOptimizationError(
            "The type of variable must be a binary variable. "
            "Use a QuadraticProgramToQubo converter to convert "
            "integer variables to binary variables. "
            "If the problem contains continuous variables, "
            "currently we can not apply VQE/QAOA directly. "
            "you might want to use an ADMM optimizer "
            "for the problem. "
        )

    # if constraints exist, raise an error
    if quad_prog.linear_constraints or quad_prog.quadratic_constraints:
        raise QiskitOptimizationError(
            "An constraint exists. "
            "The method supports only model with no constraints. "
            "Use a QuadraticProgramToQubo converter. "
            "It converts inequality constraints to equality "
            "constraints, and then, it converters equality "
            "constraints to penalty terms of the object function."
        )

    # initialize Hamiltonian.
    num_nodes = quad_prog.get_num_vars()
    pauli_list = []
    offset = 0.0
    zero = zeros(num_nodes, dtype=bool)

    # set a sign corresponding to a maximized or minimized problem.
    # sign == 1 is for minimized problem. sign == -1 is for maximized problem.
    sense = quad_prog.objective.sense.value

    # convert a constant part of the object function into Hamiltonian.
    offset += quad_prog.objective.constant * sense

    # convert linear parts of the object function into Hamiltonian.
    for idx, coef in quad_prog.objective.linear.to_dict().items():
        z_p = zeros(num_nodes, dtype=bool)
        weight = coef * sense / 2
        z_p[idx] = True

        pauli_list.append([-weight, Pauli((z_p, zero))])
        offset += weight

    # convert quadratic parts of the object function into Hamiltonian.
    # first merge coefficients (i, j) and (j, i)
    coeffs = {}  # type: Dict
    for (i, j), coeff in quad_prog.objective.quadratic.to_dict().items():
        if j < i:  # type: ignore
            coeffs[(j, i)] = coeffs.get((j, i), 0.0) + coeff
        else:
            coeffs[(i, j)] = coeffs.get((i, j), 0.0) + coeff

    # create Pauli terms
    for (i, j), coeff in coeffs.items():

        weight = coeff * sense / 4

        if i == j:
            offset += weight
        else:
            z_p = zeros(num_nodes, dtype=bool)
            z_p[i] = True
            z_p[j] = True
            pauli_list.append([weight, Pauli((z_p, zero))])

        z_p = zeros(num_nodes, dtype=bool)
        z_p[i] = True
        pauli_list.append([-weight, Pauli((z_p, zero))])

        z_p = zeros(num_nodes, dtype=bool)
        z_p[j] = True
        pauli_list.append([-weight, Pauli((z_p, zero))])

        offset += weight

    # Remove paulis whose coefficients are zeros.
    qubit_op = sum(PauliOp(pauli, coeff=coeff) for coeff, pauli in pauli_list)

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
) -> "QuadraticProgram":
    r"""Create a quadratic program from a qubit operator and a shift value.

    Variables are mapped to qubits in the same order, i.e.,
    i-th variable is mapped to i-th qubit.
    See https://github.com/Qiskit/qiskit-terra/issues/1148 for details.

    Args:
        qubit_op: The qubit operator of the problem.
        offset: The constant value in the Ising Hamiltonian.
        linear: If linear is True, :math:`x^2` is treated as a linear term
            since :math:`x^2 = x` for :math:`x \in \{0,1\}`.
            Else, :math:`x^2` is treat as a quadratic term.
            The default value is False.

    Returns:
        The quadratic program corresponding to the qubit operator.

    Raises:
        QiskitOptimizationError: If there are Pauli Xs in any Pauli term
        QiskitOptimizationError: If there are more than 2 Pauli Zs in any Pauli term
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

    # pylint: disable=cyclic-import
    from qiskit_optimization.problems.quadratic_program import QuadraticProgram

    quad_prog = QuadraticProgram()

    # add binary variables
    for i in range(qubit_op.num_qubits):
        quad_prog.binary_var(name="x_{0}".format(i))

    # Create a QUBO matrix
    # The Qubo matrix is an upper triangular matrix.
    # Diagonal elements in the QUBO matrix are for linear terms of the qubit operator.
    # The other elements in the QUBO matrix are for quadratic terms of the qubit operator.
    qubo_matrix = zeros((qubit_op.num_qubits, qubit_op.num_qubits))

    if not isinstance(qubit_op, SummedOp):
        pauli_list = [qubit_op.to_pauli_op()]
    else:
        pauli_list = qubit_op.to_pauli_op()

    for pauli_op in pauli_list:
        pauli_op = pauli_op.to_pauli_op()
        pauli = pauli_op.primitive
        coeff = pauli_op.coeff
        # Count the number of Pauli Zs in a Pauli term
        lst_z = pauli.z.tolist()
        z_index = [i for i, z in enumerate(lst_z) if z is True]
        num_z = len(z_index)

        # Add its weight of the Pauli term to the corresponding element of QUBO matrix
        if num_z == 1:
            qubo_matrix[z_index[0], z_index[0]] = coeff.real
        elif num_z == 2:
            qubo_matrix[z_index[0], z_index[1]] = coeff.real
        else:
            raise QiskitOptimizationError(
                "There are more than 2 Pauli Zs in the Pauli term {}".format(pauli.z)
            )

        # If there are Pauli Xs in the Pauli term, raise an error
        lst_x = pauli.x.tolist()
        x_index = [i for i, x in enumerate(lst_x) if x is True]
        if len(x_index) > 0:
            raise QiskitOptimizationError("Pauli Xs exist in the Pauli {}".format(pauli.x))

    # Initialize dicts for linear terms and quadratic terms
    linear_terms = {}
    quadratic_terms = {}

    # For quadratic pauli terms of operator
    # x_i * x_ j = (1 - Z_i - Z_j + Z_i * Z_j)/4
    for i, row in enumerate(qubo_matrix):
        for j, weight in enumerate(row):
            # Focus on the upper triangular matrix
            if j <= i:
                continue
            # Add a quadratic term to the object function of `QuadraticProgram`
            # The coefficient of the quadratic term in `QuadraticProgram` is
            # 4 * weight of the pauli
            coef = weight * 4
            quadratic_terms[i, j] = coef
            # Sub the weight of the quadratic pauli term from the QUBO matrix
            qubo_matrix[i, j] -= weight
            # Sub the weight of the linear pauli term from the QUBO matrix
            qubo_matrix[i, i] += weight
            qubo_matrix[j, j] += weight
            # Sub the weight from offset
            offset -= weight

    # After processing quadratic pauli terms, only linear paulis are left
    # x_i = (1 - Z_i)/2
    for i in range(qubit_op.num_qubits):
        weight = qubo_matrix[i, i]
        # Add a linear term to the object function of `QuadraticProgram`
        # The coefficient of the linear term in `QuadraticProgram` is
        # 2 * weight of the pauli
        coef = weight * 2
        if linear:
            # If the linear option is True, add it into linear_terms
            linear_terms[i] = -coef
        else:
            # Else, add it into quadratic_terms as a diagonal element.
            quadratic_terms[i, i] = -coef
        # Sub the weight of the linear pauli term from the QUBO matrix
        qubo_matrix[i, i] -= weight
        offset += weight

    # Set the objective function
    quad_prog.minimize(constant=offset, linear=linear_terms, quadratic=quadratic_terms)

    return quad_prog
