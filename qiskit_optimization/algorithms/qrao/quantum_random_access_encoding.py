# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Quantum Random Access Encoding module."""
from __future__ import annotations

from collections import defaultdict
from functools import reduce
from typing import cast

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qiskit_optimization.exceptions import QiskitOptimizationError
from qiskit_optimization.problems.quadratic_program import QuadraticProgram


def _z_to_31p_qrac_basis_circuit(bases: list[int], bit_flip: int = 0) -> QuantumCircuit:
    """Return the circuit that implements the rotation to the (3,1,p)-QRAC.

    Args:
        bases: The basis, 0, 1, 2, or 3, for the qubit.
        bit_flip: Whether to flip the state of the qubit. 1 for flip, 0 for no flip.

    Returns:
        The ``QuantumCircuit`` implementing the rotation to the (3,1,p)-QRAC.

    Raises:
        ValueError: If the basis is not 0, 1, 2, or 3
    """
    circ = QuantumCircuit(len(bases))
    BETA = np.arccos(1 / np.sqrt(3))  # pylint: disable=invalid-name

    for i, base in enumerate(reversed(bases)):
        if bit_flip:
            # if bit_flip == 1: then flip the state of the qubit to |1>
            circ.x(i)

        if base == 0:
            circ.r(-BETA, -np.pi / 4, i)
        elif base == 1:
            circ.r(np.pi - BETA, np.pi / 4, i)
        elif base == 2:
            circ.r(np.pi + BETA, np.pi / 4, i)
        elif base == 3:
            circ.r(BETA, -np.pi / 4, i)
        else:
            raise ValueError(f"Unknown basis: {base}. Basis must be 0, 1, 2, or 3.")
    return circ


def _z_to_21p_qrac_basis_circuit(bases: list[int], bit_flip: int = 0) -> QuantumCircuit:
    """Return the circuit that implements the rotation to the (2,1,p)-QRAC.

    Args:
        bases: The basis, 0, 1, for the qubit.
        bit_flip: Whether to flip the state of the qubit. 1 for flip, 0 for no flip.

    Returns:
        The ``QuantumCircuit`` implementing the rotation to the (2,1,p)-QRAC.

    Raises:
        ValueError: if the basis is not 0 or 1
    """
    circ = QuantumCircuit(len(bases))

    for i, base in enumerate(reversed(bases)):
        if bit_flip:
            # if bit_flip == 1: then flip the state of the qubit to |1>
            circ.x(i)

        if base == 0:
            circ.r(-1 * np.pi / 4, -np.pi / 2, i)
        elif base == 1:
            circ.r(-3 * np.pi / 4, -np.pi / 2, i)
        else:
            raise ValueError(f"Unknown basis: {base}. Basis must be 0, 1.")
    return circ


def _qrac_state_prep_1q(bit_list: list[int]) -> QuantumCircuit:
    """
    Return the circuit that prepares the state for a (1,1,p), (2,1,p), or (3,1,p)-QRAC.

    Args:
        bit_list: The bitstring to prepare. If 1 argument is given, then a (1,1,p)-QRAC is generated.
            If 2 arguments are given, then a (2,1,p)-QRAC is generated. If 3 arguments are given,
            then a (3,1,p)-QRAC is generated.

    Returns:
        The ``QuantumCircuit`` implementing the state preparation.

    Raises:
        TypeError: if the number of arguments is not 1, 2, or 3
        ValueError: if any of the arguments are not 0 or 1
    """
    # pylint: disable=C0401
    if len(bit_list) not in (1, 2, 3):
        raise TypeError(f"qrac_state_prep_1q requires 1, 2, or 3 arguments, not {len(bit_list)}.")
    if not all(bit in (0, 1) for bit in bit_list):
        raise ValueError("Each argument to qrac_state_prep_1q must be 0 or 1.")

    if len(bit_list) == 3:
        # Prepare (3,1,p)-qrac
        # In the following lines, the input bits are XORed to match the
        # conventions used in the paper.
        # To understand why this transformation happens,
        # observe that the two states that define each magic basis
        # correspond to the same bitstrings but with a global bit flip.
        # Thus the three bits of information we use to construct these states are:
        # base_index0,base_index1 : two bits to pick one of four magic bases
        # bit_flip: one bit to indicate which magic basis projector we are interested in.

        bit_flip = bit_list[0] ^ bit_list[1] ^ bit_list[2]
        base_index0 = bit_list[1] ^ bit_list[2]
        base_index1 = bit_list[0] ^ bit_list[2]

        # This is a convention chosen to be consistent with https://arxiv.org/abs/2111.03167
        # See SI:4 second paragraph and observe that π+ = |0X0|, π- = |1X1|
        base = [2 * base_index0 + base_index1]
        circ = _z_to_31p_qrac_basis_circuit(base, bit_flip)

    elif len(bit_list) == 2:
        # Prepare (2,1,p)-qrac
        # (00,01) or (10,11)
        bit_flip = bit_list[0]
        # (00,11) or (01,10)
        base_index0 = bit_list[0] ^ bit_list[1]

        # This is a convention chosen to be consistent with https://arxiv.org/abs/2111.03167
        # # See SI:4 second paragraph and observe that π+ = |0X0|, π- = |1X1|
        base = [base_index0]
        circ = _z_to_21p_qrac_basis_circuit(base, bit_flip)

    else:
        bit_flip = bit_list[0]
        circ = QuantumCircuit(1)
        if bit_flip:
            circ.x(0)

    return circ


def _qrac_state_prep_multi_qubit(
    x: list[int],
    q2vars: list[list[int]],
    max_vars_per_qubit: int,
) -> QuantumCircuit:
    """Prepares a multi qubit QRAC state.

    Args:
        x: The state of each decision variable (0 or 1).
        q2vars: A list of lists of integers. Each inner list contains the indices of decision variables
        mapped to a specific qubit.
        max_vars_per_qubit: The maximum number of decision variables that can be mapped to a
        single qubit.

    Returns:
        A QuantumCircuit object representing the prepared state.

    Raises:
        ValueError: If any qubit is associated with more than `max_vars_per_qubit` variables.
        ValueError: If a decision variable in ``q2vars`` is not included in `x`.
        ValueError: If there are unused decision variables in `x` after mapping to qubits.
    """
    # Create a set of all remaining decision variables
    remaining_dvars = set(range(len(x)))
    # Create a list to store the binary mappings of each qubit to its corresponding decision variables
    variable_mappings: list[list[int]] = []
    # Check that each qubit is associated with at most max_vars_per_qubit variables
    for qi_vars in q2vars:
        if len(qi_vars) > max_vars_per_qubit:
            raise ValueError(
                "Each qubit is expected to be associated with at most "
                f"`max_vars_per_qubit` ({max_vars_per_qubit}) variables, "
                f"not {len(qi_vars)} variables."
            )
        # Create a list to store the binary mapping of the current qubit
        qi_bits: list[int] = []

        # Map each decision variable associated with the current qubit to a binary value and add it
        # to the qubit bits
        for dvar in qi_vars:
            try:
                qi_bits.append(x[dvar])
            except IndexError:
                raise ValueError(f"Decision variable not included in dvars: {dvar}") from None
            try:
                remaining_dvars.remove(dvar)
            except KeyError:
                raise ValueError(
                    f"Unused decision variable(s) in dvars: {remaining_dvars}"
                ) from None

        # Pad with zeros if necessary
        while len(qi_bits) < max_vars_per_qubit:
            qi_bits.append(0)

        variable_mappings.append(qi_bits)

    # Raise an error if not all decision variables are used
    if remaining_dvars:
        raise ValueError(f"Not all dvars were included in q2vars: {remaining_dvars}")

    # Prepare the individual qrac circuit and combine them into a multi qubit circuit
    qracs = [_qrac_state_prep_1q(qi_bits) for qi_bits in variable_mappings]
    qrac_circ = reduce(lambda x, y: x.tensor(y), qracs)
    return qrac_circ


class QuantumRandomAccessEncoding:
    """This class specifies a Quantum Random Access Code that can be used to encode
    the binary variables of a QUBO (quadratic unconstrained binary optimization
    problem).

    """

    # This defines the convention of the Pauli operators (and their ordering)
    # for each encoding.
    _OPERATORS = (
        (SparsePauliOp("Z"),),  # (1,1,1) QRAC
        (SparsePauliOp("X"), SparsePauliOp("Z")),  # (2,1,p) QRAC, p ≈ 0.85
        (SparsePauliOp("X"), SparsePauliOp("Y"), SparsePauliOp("Z")),  # (3,1,p) QRAC, p ≈ 0.79
    )

    def __init__(self, max_vars_per_qubit: int = 3):
        """
        Args:
            max_vars_per_qubit: The maximum number of decision variables per qubit.
                Integer values 1, 2 and 3 are supported (default to 3).
        """
        if max_vars_per_qubit not in (1, 2, 3):
            raise ValueError("max_vars_per_qubit must be 1, 2, or 3")
        self._ops = self._OPERATORS[max_vars_per_qubit - 1]

        self._qubit_op: SparsePauliOp | None = None
        self._offset: float | None = None
        self._problem: QuadraticProgram | None = None
        self._var2op: dict[int, tuple[int, SparsePauliOp]] = {}
        self._q2vars: list[list[int]] = []
        self._frozen = False

    @property
    def num_qubits(self) -> int:
        """Number of qubits"""
        return len(self._q2vars)

    @property
    def num_vars(self) -> int:
        """Number of decision variables"""
        return len(self._var2op)

    @property
    def max_vars_per_qubit(self) -> int:
        """Maximum number of variables per qubit"""
        return len(self._ops)

    @property
    def var2op(self) -> dict[int, tuple[int, SparsePauliOp]]:
        """Maps each decision variable to ``(qubit_index, operator)``"""
        return self._var2op

    @property
    def q2vars(self) -> list[list[int]]:
        """Each element contains the list of decision variable indices encoded on that qubit"""
        return self._q2vars

    @property
    def compression_ratio(self) -> float:
        """Compression ratio. Number of decision variables divided by number of qubits"""
        return self.num_vars / self.num_qubits

    @property
    def minimum_recovery_probability(self) -> float:
        """Minimum recovery probability, as set by ``max_vars_per_qubit``"""
        n = self.max_vars_per_qubit
        return (1 + 1 / np.sqrt(n)) / 2

    @property
    def qubit_op(self) -> SparsePauliOp:
        """Relaxed Hamiltonian operator.

        Raises:
            RuntimeError: If the objective function has not been set yet. Use the ``encode`` method
                to construct the Hamiltonian, or make sure that the objective function has been set.
        """
        if self._qubit_op is None:
            raise RuntimeError(
                "Cannot return the relaxed Hamiltonian operator: no objective function has been "
                "provided yet. Use the ``encode`` method to construct the Hamiltonian, or make "
                "sure that the objective function has been set."
            )
        return self._qubit_op

    @property
    def offset(self) -> float:
        """Relaxed Hamiltonian offset

        Raises:
            RuntimeError: If the offset has not been set yet. Use the ``encode`` method to construct
                the Hamiltonian, or make sure that the objective function has been set.
        """
        if self._offset is None:
            raise RuntimeError(
                "Cannot return the relaxed Hamiltonian offset: The offset attribute cannot be "
                "accessed until the ``encode`` method has been called to generate the qubit "
                "Hamiltonian. Please call ``encode`` first."
            )
        return self._offset

    @property
    def problem(self) -> QuadraticProgram:
        """The ``QuadraticProgram``  encoding a QUBO optimization problem

        Raises:
            RuntimeError: If the ``QuadraticProgram`` has not been set yet. Use the ``encode``
                method to set the problem.
        """
        if self._problem is None:
            raise RuntimeError(
                "This object has not been associated with a ``QuadraticProgram``. "
                "Please use the ``encode`` method to set the problem."
            )
        return self._problem

    def freeze(self):
        """Freeze the object to prevent further modification.

        Once an instance of this class is frozen, ``encode`` can no longer be called.
        """
        if not self._frozen:
            self._qubit_op = self._qubit_op.simplify(atol=0)
        self._frozen = True

    @property
    def frozen(self) -> bool:
        """Whether the object is frozen or not."""
        return self._frozen

    def _add_variables(self, variables: list[int]) -> None:
        """Add variables to the Encoding object.

        Args:
            variables: A list of variable indices to be added.

        Raises:
            ValueError: If added variables are not unique.
            ValueError: If added variables collide with existing ones.

        """
        # NOTE: If this is called multiple times, it *always* adds an
        # additional qubit (see final line), even if aggregating them into a
        # single call would have resulted in fewer qubits.

        # Check if variables is empty
        if not variables:
            return

        # Check if variables are unique
        if len(variables) != len(set(variables)):
            raise ValueError("Added variables must be unique")

        # Check if variables collide with existing ones
        for v in variables:
            if v in self._var2op:
                raise ValueError("Added variables cannot collide with existing ones")

        # Calculate the number of new qubits required for the added variables.
        n = len(self._ops)
        old_num_qubits = len(self._q2vars)
        num_new_qubits = int(np.ceil(len(variables) / n))
        # Add the new qubits to _q2vars.
        for _ in range(num_new_qubits):
            self._q2vars.append([])
        # Associate each added variable with a qubit and operator.
        for i, v in enumerate(variables):
            qubit, op = divmod(i, n)
            qubit_index = old_num_qubits + qubit
            self._var2op[v] = (qubit_index, self._ops[op])
            self._q2vars[qubit_index].append(v)

    def _add_term(self, w: float, *variables: int) -> None:
        """Add a term to the Hamiltonian.

        Args:
            weight: the coefficient for the term
            *variables: the list of variables for the term
        """
        # Eq. (31) in https://arxiv.org/abs/2111.03167 assumes a weight-2
        # Pauli operator.  To generalize, we replace the `d` in that equation
        # with `d_prime`, defined as follows:
        d_prime = np.sqrt(self.max_vars_per_qubit) ** len(variables)
        op = self._term2op(*variables) * (w * d_prime)

        if w == 0.0:
            return
        if self._qubit_op is None:
            self._qubit_op = op
        else:
            self._qubit_op += op

    def _term2op(self, *variables: int) -> SparsePauliOp:
        """Construct a ``SparsePauliOp`` that is a tensor product of encoded decision variables.

        Args:
            *variables: The indices of the decision variables to encode.

        Returns:
            The encoded ``SparsePauliOp`` representing the product of the provided variables.

        Raises:
            QiskitOptimizationError: If any of the decision variables to be encoded collide in qubit
            space.

        """
        ops = [SparsePauliOp("I")] * self.num_qubits
        done = set()
        for x in variables:
            pos, op = self._var2op[x]
            if pos in done:
                raise QiskitOptimizationError(f"Collision of variables: {variables}")
            ops[pos] = op
            done.add(pos)
        pauli_op = reduce(lambda x, y: x.tensor(y), ops)
        return pauli_op

    @staticmethod
    def _generate_ising_coefficients(
        problem: QuadraticProgram,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Generate coefficients of Hamiltonian from a given problem."""
        num_vars = problem.get_num_vars()

        # set a sign corresponding to a maximized or minimized problem:
        # 1 is for minimized problem, -1 is for maximized problem.
        sense = problem.objective.sense.value

        # convert a constant part of the objective function into Hamiltonian.
        offset = problem.objective.constant * sense

        # convert linear parts of the objective function into Hamiltonian.
        linear = np.zeros(num_vars)
        for idx, coef in problem.objective.linear.to_dict().items():
            idx = cast(int, idx)
            weight = coef * sense / 2
            linear[idx] -= weight
            offset += weight

        # convert quadratic parts of the objective function into Hamiltonian.
        quad = np.zeros((num_vars, num_vars))
        for (i, j), coef in problem.objective.quadratic.to_dict().items():
            i = cast(int, i)
            j = cast(int, j)
            weight = coef * sense / 4
            if i == j:
                linear[i] -= 2 * weight
                offset += 2 * weight
            else:
                quad[i, j] += weight
                linear[i] -= weight
                linear[j] -= weight
                offset += weight

        return offset, linear, quad

    @staticmethod
    def _find_variable_partition(quad: np.ndarray) -> dict[int, list[int]]:
        """Find the variable partition of the quad based on the node coloring of the graph

        Args:
            coefficients of the quadratic part of the Hamiltonian.

        Returns:
            A dictionary of the variable partition of the quad based on the node coloring.
        """
        # pylint: disable=E1101
        color2node: dict[int, list[int]] = defaultdict(list)
        num_nodes = quad.shape[0]
        graph = nx.Graph()
        graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from(list(zip(*np.where(quad != 0))))
        node2color = nx.greedy_color(graph)
        for node, color in sorted(node2color.items()):
            color2node[color].append(node)
        return color2node

    def encode(self, problem: QuadraticProgram) -> None:
        """
        Encodes a given ``QuadraticProgram`` as a (n,1,p) Quantum Random Access Code (QRAC)
        relaxed Hamiltonian. It accomplishes this by mapping each binary decision variable to one
        qubit of the QRAC. The encoding is designed to ensure that the problem's objective function
        commutes with the QRAC encoding.

        After the function is called, it sets the following attributes:
            - qubit_op: The qubit operator that encodes the input ``QuadraticProgram``.
            - offset: The constant value in the encoded Hamiltonian.
            - problem: The original ``QuadraticProgram`` used for encoding.

        Inputs:
            problem: A ``QuadraticProgram`` encoding a QUBO optimization problem

        Raises:
            QiskitOptimizationError: If this method is called more than once on the same object.
            QiskitOptimizationError: If the problem contains non-binary variables.
            QiskitOptimizationError: If the problem contains constraints.
        """
        # Ensure the Encoding object is not already used
        if self._frozen:
            raise QiskitOptimizationError(
                "Cannot reuse an Encoding object that has already been used. "
                "Please create a new Encoding object and call encode() on it."
            )

        # Check for non-binary variables
        if problem.get_num_vars() > problem.get_num_binary_vars():
            raise QiskitOptimizationError(
                "All variables must be binary. "
                "Please convert integer variables to binary variables using the"
                "``QuadraticProgramToQubo`` converter. "
                "Continuous variables are not supported by the QRAO algorithm."
            )

        # Check for constraints
        if problem.linear_constraints or problem.quadratic_constraints:
            raise QiskitOptimizationError(
                "The problem cannot contain constraints. "
                "Please convert constraints to penalty terms of the objective function using the "
                "``QuadraticProgramToQubo`` converter."
            )

        num_vars = problem.get_num_vars()

        # Generate the coefficients of the Hamiltonian
        offset, linear, quad = self._generate_ising_coefficients(problem)

        # Find the partition of the variables into groups
        variable_partition = self._find_variable_partition(quad)

        # Add variables and generate the Hamiltonian
        for _, v in sorted(variable_partition.items()):
            self._add_variables(sorted(v))
        for i in range(num_vars):
            w = linear[i]
            if w != 0:
                self._add_term(w, i)
        for i in range(num_vars):
            for j in range(num_vars):
                w = quad[i, j]
                if w != 0:
                    self._add_term(w, i, j)

        self._offset = offset
        self._problem = problem

        self.freeze()

    def state_preparation_circuit(self, x: list[int]) -> QuantumCircuit:
        """
        Generate a circuit that prepares the state corresponding to the given binary string.

        Args:
            x: A list of binary values to be encoded into the state.

        Returns:
            A QuantumCircuit that prepares the state corresponding to the given binary string.
        """
        return _qrac_state_prep_multi_qubit(x, self.q2vars, self.max_vars_per_qubit)
