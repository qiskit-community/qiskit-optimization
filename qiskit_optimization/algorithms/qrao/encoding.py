# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quantum Random Access Encoding module.

Contains code dealing with QRACs (quantum random access codes) and preparation
of such states.

.. autosummary::
   :toctree: ../stubs/

   z_to_31p_qrac_basis_circuit
   z_to_21p_qrac_basis_circuit
   qrac_state_prep_1q
   qrac_state_prep_multiqubit
   QuantumRandomAccessEncoding

"""

from typing import Tuple, List, Dict, Optional, Union
from collections import defaultdict
from functools import reduce
from itertools import chain

import numpy as np
import rustworkx as rx

from qiskit import QuantumCircuit
from qiskit.opflow import (
    I,
    X,
    Y,
    Z,
    PauliSumOp,
    PrimitiveOp,
    CircuitOp,
    Zero,
    One,
    StateFn,
    CircuitStateFn,
)
from qiskit.quantum_info import SparsePauliOp

from qiskit_optimization.problems.quadratic_program import QuadraticProgram


def _ceildiv(n: int, d: int) -> int:
    """Perform ceiling division in integer arithmetic

    >>> _ceildiv(0, 3)
    0
    >>> _ceildiv(1, 3)
    1
    >>> _ceildiv(3, 3)
    1
    >>> _ceildiv(4, 3)
    2
    """
    return (n - 1) // d + 1


def z_to_31p_qrac_basis_circuit(basis: List[int]) -> QuantumCircuit:
    """Return the basis rotation corresponding to the (3,1,p)-QRAC

    Args:

        basis: 0, 1, 2, or 3 for each qubit

    Returns:
        The ``QuantumCircuit`` implementing the rotation.
    """
    circ = QuantumCircuit(len(basis))
    BETA = np.arccos(1 / np.sqrt(3))
    for i, base in enumerate(reversed(basis)):
        if base == 0:
            circ.r(-BETA, -np.pi / 4, i)
        elif base == 1:
            circ.r(np.pi - BETA, np.pi / 4, i)
        elif base == 2:
            circ.r(np.pi + BETA, np.pi / 4, i)
        elif base == 3:
            circ.r(BETA, -np.pi / 4, i)
        else:
            raise ValueError(f"Unknown base: {base}")
    return circ


def z_to_21p_qrac_basis_circuit(basis: List[int]) -> QuantumCircuit:
    """Return the basis rotation corresponding to the (2,1,p)-QRAC

    Args:

        basis: 0 or 1 for each qubit

    Returns:
        The ``QuantumCircuit`` implementing the rotation.
    """
    circ = QuantumCircuit(len(basis))
    for i, base in enumerate(reversed(basis)):
        if base == 0:
            circ.r(-1 * np.pi / 4, -np.pi / 2, i)
        elif base == 1:
            circ.r(-3 * np.pi / 4, -np.pi / 2, i)
        else:
            raise ValueError(f"Unknown base: {base}")
    return circ


def qrac_state_prep_1q(*m: int) -> CircuitStateFn:
    """Prepare a single qubit QRAC state

      This function accepts 1, 2, or 3 arguments, in which case it generates a
      1-QRAC, 2-QRAC, or 3-QRAC, respectively.

    Args:

        m: The data to be encoded. Each argument must be 0 or 1.

    Returns:

        The circuit state function.

    """
    if len(m) not in (1, 2, 3):
        raise TypeError(
            f"qrac_state_prep_1q requires 1, 2, or 3 arguments, not {len(m)}."
        )
    if not all(mi in (0, 1) for mi in m):
        raise ValueError("Each argument to qrac_state_prep_1q must be 0 or 1.")

    if len(m) == 3:
        # Prepare (3,1,p)-qrac

        # In the following lines, the input bits are XOR'd to match the
        # conventions used in the paper.

        # To understand why this transformation happens,
        # observe that the two states that define each magic basis
        # correspond to the same bitstrings but with a global bitflip.

        # Thus the three bits of information we use to construct these states are:
        # c0,c1 : two bits to pick one of four magic bases
        # c2: one bit to indicate which magic basis projector we are interested in.

        c0 = m[0] ^ m[1] ^ m[2]
        c1 = m[1] ^ m[2]
        c2 = m[0] ^ m[2]

        base = [2 * c1 + c2]
        cob = z_to_31p_qrac_basis_circuit(base)
        # This is a convention chosen to be consistent with https://arxiv.org/pdf/2111.03167v2.pdf
        # See SI:4 second paragraph and observe that π+ = |0X0|, π- = |1X1|
        sf = One if (c0) else Zero
        # Apply the z_to_magic_basis circuit to either |0> or |1>
        logical = CircuitOp(cob) @ sf
    elif len(m) == 2:
        # Prepare (2,1,p)-qrac
        # (00,01) or (10,11)
        c0 = m[0]
        # (00,11) or (01,10)
        c1 = m[0] ^ m[1]

        base = [c1]
        cob = z_to_21p_qrac_basis_circuit(base)
        # This is a convention chosen to be consistent with https://arxiv.org/pdf/2111.03167v2.pdf
        # See SI:4 second paragraph and observe that π+ = |0X0|, π- = |1X1|
        sf = One if (c0) else Zero
        # Apply the z_to_magic_basis circuit to either |0> or |1>
        logical = CircuitOp(cob) @ sf
    else:
        assert len(m) == 1
        c0 = m[0]
        sf = One if (c0) else Zero

        logical = sf

    return logical.to_circuit_op()


def qrac_state_prep_multiqubit(
    dvars: Union[Dict[int, int], List[int]],
    q2vars: List[List[int]],
    max_vars_per_qubit: int,
) -> CircuitStateFn:
    """
    Prepare a multiqubit QRAC state.

    Args:
        dvars: state of each decision variable (0 or 1)
    """
    remaining_dvars = set(dvars if isinstance(dvars, dict) else range(len(dvars)))
    ordered_bits = []
    for qi_vars in q2vars:
        if len(qi_vars) > max_vars_per_qubit:
            raise ValueError(
                "Each qubit is expected to be associated with at most "
                f"`max_vars_per_qubit` ({max_vars_per_qubit}) variables, "
                f"not {len(qi_vars)} variables."
            )
        if not qi_vars:
            # This probably actually doesn't cause any issues, but why support
            # it (and test this edge case) if we don't have to?
            raise ValueError(
                "There is a qubit without any decision variables assigned to it."
            )
        qi_bits: List[int] = []
        for dv in qi_vars:
            try:
                qi_bits.append(dvars[dv])
            except (KeyError, IndexError):
                raise ValueError(
                    f"Decision variable not included in dvars: {dv}"
                ) from None
            try:
                remaining_dvars.remove(dv)
            except KeyError:
                raise ValueError(
                    f"Unused decision variable(s) in dvars: {remaining_dvars}"
                ) from None
        # Pad with zeros if there are fewer than `max_vars_per_qubit`.
        # NOTE: This results in everything being encoded as an n-QRAC,
        # even if there are fewer than n decision variables encoded in the qubit.
        # In the future, we plan to make the encoding "adaptive" so that the
        # optimal encoding is used on each qubit, based on the number of
        # decision variables assigned to that specific qubit.
        # However, we cannot do this until magic state rounding supports 2-QRACs.
        while len(qi_bits) < max_vars_per_qubit:
            qi_bits.append(0)

        ordered_bits.append(qi_bits)

    if remaining_dvars:
        raise ValueError(f"Not all dvars were included in q2vars: {remaining_dvars}")

    qracs = [qrac_state_prep_1q(*qi_bits) for qi_bits in ordered_bits]
    logical = reduce(lambda x, y: x ^ y, qracs)
    return logical


def q2vars_from_var2op(var2op: Dict[int, Tuple[int, PrimitiveOp]]) -> List[List[int]]:
    """Calculate q2vars given var2op"""
    num_qubits = max(qubit_index for qubit_index, _ in var2op.values()) + 1
    q2vars: List[List[int]] = [[] for i in range(num_qubits)]
    for var, (q, _) in var2op.items():
        q2vars[q].append(var)
    return q2vars


class QuantumRandomAccessEncoding:
    """This class specifies a Quantum Random Access Code that can be used to encode
    the binary variables of a QUBO (quadratic unconstrained binary optimization
    problem).

    Args:
        max_vars_per_qubit: maximum possible compression ratio.
            Supported values are 1, 2, or 3.

    """

    # This defines the convention of the Pauli operators (and their ordering)
    # for each encoding.
    OPERATORS = (
        (Z,),  # (1,1,1) QRAC
        (X, Z),  # (2,1,p) QRAC, p ≈ 0.85
        (X, Y, Z),  # (3,1,p) QRAC, p ≈ 0.79
    )

    def __init__(self, max_vars_per_qubit: int = 3):
        if max_vars_per_qubit not in (1, 2, 3):
            raise ValueError("max_vars_per_qubit must be 1, 2, or 3")
        self._ops = self.OPERATORS[max_vars_per_qubit - 1]

        self._qubit_op: Optional[PauliSumOp] = None
        self._offset: Optional[float] = None
        self._problem: Optional[QuadraticProgram] = None
        self._var2op: Dict[int, Tuple[int, PrimitiveOp]] = {}
        self._q2vars: List[List[int]] = []
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
        """Maximum number of variables per qubit

        This is set in the constructor and controls the maximum compression ratio
        """

        return len(self._ops)

    @property
    def var2op(self) -> Dict[int, Tuple[int, PrimitiveOp]]:
        """Maps each decision variable to ``(qubit_index, operator)``"""
        return self._var2op

    @property
    def q2vars(self) -> List[List[int]]:
        """Each element contains the list of decision variable indice(s) encoded on that qubit"""
        return self._q2vars

    @property
    def compression_ratio(self) -> float:
        """Compression ratio

        Number of decision variables divided by number of qubits
        """
        return self.num_vars / self.num_qubits

    @property
    def minimum_recovery_probability(self) -> float:
        """Minimum recovery probability, as set by ``max_vars_per_qubit``"""
        n = self.max_vars_per_qubit
        return (1 + 1 / np.sqrt(n)) / 2

    @property
    def qubit_op(self) -> PauliSumOp:
        """Relaxed Hamiltonian operator"""
        if self._qubit_op is None:
            raise AttributeError(
                "No objective function has been provided from which a "
                "qubit Hamiltonian can be constructed. Please use the "
                "encode method if you wish to manually compile "
                "this field."
            )
        return self._qubit_op

    @property
    def offset(self) -> float:
        """Relaxed Hamiltonian offset"""
        if self._offset is None:
            raise AttributeError(
                "No objective function has been provided from which a "
                "qubit Hamiltonian can be constructed. Please use the "
                "encode method if you wish to manually compile "
                "this field."
            )
        return self._offset

    @property
    def problem(self) -> QuadraticProgram:
        """The ``QuadraticProgram`` used as basis for the encoding"""
        if self._problem is None:
            raise AttributeError(
                "No quadratic program has been associated with this object. "
                "Please use the encode method if you wish to do so."
            )
        return self._problem

    def _add_variables(self, variables: List[int]) -> None:
        self.ensure_thawed()
        # NOTE: If this is called multiple times, it *always* adds an
        # additional qubit (see final line), even if aggregating them into a
        # single call would have resulted in fewer qubits.
        if self._qubit_op is not None:
            raise RuntimeError(
                "_add_variables() cannot be called once terms have been added "
                "to the operator, as the number of qubits must thereafter "
                "remain fixed."
            )
        if not variables:
            return
        if len(variables) != len(set(variables)):
            raise ValueError("Added variables must be unique")
        for v in variables:
            if v in self._var2op:
                raise ValueError("Added variables cannot collide with existing ones")
        # Modify the object now that error checking is complete.
        n = len(self._ops)
        old_num_qubits = len(self._q2vars)
        num_new_qubits = _ceildiv(len(variables), n)
        # Populate self._var2op and self._q2vars
        for _ in range(num_new_qubits):
            self._q2vars.append([])
        for i, v in enumerate(variables):
            qubit, op = divmod(i, n)
            qubit_index = old_num_qubits + qubit
            assert v not in self._var2op  # was checked above
            self._var2op[v] = (qubit_index, self._ops[op])
            self._q2vars[qubit_index].append(v)

    def _add_term(self, w: float, *variables: int) -> None:
        self.ensure_thawed()
        # Eq. (31) in https://arxiv.org/abs/2111.03167v2 assumes a weight-2
        # Pauli operator.  To generalize, we replace the `d` in that equation
        # with `d_prime`, defined as follows:
        d_prime = np.sqrt(self.max_vars_per_qubit) ** len(variables)
        op = self.term2op(*variables).mul(w * d_prime)
        # We perform the following short-circuit *after* calling term2op so at
        # least we have confirmed that the user provided a valid variables list.
        if w == 0.0:
            return
        if self._qubit_op is None:
            self._qubit_op = op
        else:
            self._qubit_op += op

    def term2op(self, *variables: int) -> PauliSumOp:
        """Construct a ``PauliSumOp`` that is a product of encoded decision ``variable``\\(s).

        The decision variables provided must all be encoded on different qubits.
        """
        ops = [I] * self.num_qubits
        done = set()
        for x in variables:
            pos, op = self._var2op[x]
            if pos in done:
                raise RuntimeError(f"Collision of variables: {variables}")
            ops[pos] = op
            done.add(pos)
        pauli_op = reduce(lambda x, y: x ^ y, ops)
        # Convert from PauliOp to PauliSumOp
        return PauliSumOp(SparsePauliOp(pauli_op.primitive, coeffs=[pauli_op.coeff]))

    @staticmethod
    def _generate_ising_terms(
        problem: QuadraticProgram,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        num_vars = problem.get_num_vars()

        # set a sign corresponding to a maximized or minimized problem:
        # 1 is for minimized problem, -1 is for maximized problem.
        sense = problem.objective.sense.value

        # convert a constant part of the objective function into Hamiltonian.
        offset = problem.objective.constant * sense

        # convert linear parts of the objective function into Hamiltonian.
        linear = np.zeros(num_vars)
        for idx, coef in problem.objective.linear.to_dict().items():
            assert isinstance(idx, int)  # hint for mypy
            weight = coef * sense / 2
            linear[idx] -= weight
            offset += weight

        # convert quadratic parts of the objective function into Hamiltonian.
        quad = np.zeros((num_vars, num_vars))
        for (i, j), coef in problem.objective.quadratic.to_dict().items():
            assert isinstance(i, int)  # hint for mypy
            assert isinstance(j, int)  # hint for mypy
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
    def _find_variable_partition(quad: np.ndarray) -> Dict[int, List[int]]:
        num_nodes = quad.shape[0]
        assert quad.shape == (num_nodes, num_nodes)
        graph = rx.PyGraph()
        graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from_no_data(list(zip(*np.where(quad != 0))))
        node2color = rx.graph_greedy_color(graph)
        color2node: Dict[int, List[int]] = defaultdict(list)
        for node, color in sorted(node2color.items()):
            color2node[color].append(node)
        return color2node

    def encode(self, problem: QuadraticProgram) -> None:
        """Encode the (n,1,p) QRAC relaxed Hamiltonian of this problem.

            We associate to each binary decision variable one bit of a
            (n,1,p) Quantum Random Access Code. This is done in such a way that the
            given problem's objective function commutes with the encoding.

        After being called, the object will have the following attributes:
            qubit_op: The qubit operator encoding the input QuadraticProgram.
            offset: The constant value in the encoded Hamiltonian.
            problem: The ``problem`` used for encoding.

        Inputs:
            problem: A QuadraticProgram object encoding a QUBO optimization problem

        Raises:
            RuntimeError: if the ``problem`` isn't a QUBO or if the current
                object has been used already

        """
        # Ensure fresh object
        if self.num_qubits > 0:
            raise RuntimeError(
                "Must call encode() on an Encoding that has not been used already"
            )

        # if problem has variables that are not binary, raise an error
        if problem.get_num_vars() > problem.get_num_binary_vars():
            raise RuntimeError(
                "The type of all variables must be binary. "
                "You can use `QuadraticProgramToQubo` converter "
                "to convert integer variables to binary variables. "
                "If the problem contains continuous variables, `qrao` "
                "cannot handle it."
            )

        # if constraints exist, raise an error
        if problem.linear_constraints or problem.quadratic_constraints:
            raise RuntimeError(
                "There must be no constraint in the problem. "
                "You can use `QuadraticProgramToQubo` converter to convert "
                "constraints to penalty terms of the objective function."
            )

        num_vars = problem.get_num_vars()

        # Generate the decision variable terms in terms of Ising variables (+1 or -1)
        offset, linear, quad = self._generate_ising_terms(problem)

        # Find variable partition (a graph coloring is sufficient)
        variable_partition = self._find_variable_partition(quad)

        # The other methods of the current class allow for the variables to
        # have arbitrary integer indices [i.e., they need not correspond to
        # range(num_vars)], and the tests corresponding to this file ensure
        # that this works.  However, the current method is a high-level one
        # that takes a QuadraticProgram, which always has its variables
        # numbered sequentially.  Furthermore, other portions of the QRAO code
        # base [most notably the assignment of variable_ops in solve_relaxed()
        # and the corresponding result objects] assume that the variables are
        # numbered from 0 to (num_vars - 1).  So we enforce that assumption
        # here, both as a way of documenting it and to make sure
        # _find_variable_partition() returns a sensible result (in case the
        # user overrides it).
        assert sorted(chain.from_iterable(variable_partition.values())) == list(
            range(num_vars)
        )

        # generate a Hamiltonian
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

        # This is technically optional and can wait until the optimizer is
        # constructed, but there's really no reason not to freeze
        # immediately.
        self.freeze()

    def freeze(self):
        """Freeze the object to prevent further modification.

        Once an instance of this class is frozen, ``_add_variables`` and ``_add_term``
        can no longer be called.

        This operation is idempotent.  There is no way to undo it, as it exists
        to allow another object to rely on this one not changing its state
        going forward without having to make a copy as a distinct object.
        """
        if self._frozen is False:
            self._qubit_op = self._qubit_op.reduce()
        self._frozen = True

    @property
    def frozen(self) -> bool:
        """``True`` if the object can no longer be modified, ``False`` otherwise."""
        return self._frozen

    def ensure_thawed(self) -> None:
        """Raise a ``RuntimeError`` if the object is frozen and thus cannot be modified."""
        if self._frozen:
            raise RuntimeError("Cannot modify an encoding that has been frozen")

    def state_prep(self, dvars: Union[Dict[int, int], List[int]]) -> CircuitStateFn:
        """Prepare a multiqubit QRAC state."""
        return qrac_state_prep_multiqubit(dvars, self.q2vars, self.max_vars_per_qubit)


class EncodingCommutationVerifier:
    """Class for verifying that the relaxation commutes with the objective function

    See also the "check encoding problem commutation" how-to notebook.
    """

    def __init__(self, encoding: QuantumRandomAccessEncoding):
        self._encoding = encoding

    def __len__(self) -> int:
        return 2**self._encoding.num_vars

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i: int) -> Tuple[str, float, float]:
        if i not in range(len(self)):
            raise IndexError(f"Index out of range: {i}")

        encoding = self._encoding
        str_dvars = ("{0:0" + str(encoding.num_vars) + "b}").format(i)
        dvars = [int(b) for b in str_dvars]
        encoded_bitstr = encoding.state_prep(dvars)

        # Offset accounts for the value of the encoded Hamiltonian's
        # identity coefficient. This term need not be evaluated directly as
        # Tr[I•rho] is always 1.
        offset = encoding.offset

        # Evaluate Un-encoded Problem
        # ========================
        # `sense` accounts for sign flips depending on whether
        # we are minimizing or maximizing the objective function
        problem = encoding.problem
        sense = problem.objective.sense.value
        obj_val = problem.objective.evaluate(dvars) * sense

        # Evaluate Encoded Problem
        # ========================
        encoded_problem = encoding.qubit_op  # H
        encoded_obj_val = (
            np.real((~StateFn(encoded_problem) @ encoded_bitstr).eval()) + offset
        )

        return (str_dvars, obj_val, encoded_obj_val)
