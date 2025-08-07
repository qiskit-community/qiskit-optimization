# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Grover's search algorithm."""
from __future__ import annotations

import itertools
from collections.abc import Generator, Iterator

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit

from .amplification_problem import AmplificationProblem


class Grover:
    r"""Grover's Search algorithm.

    .. note::

        If you want to learn more about the theory behind Grover's Search algorithm, check
        out the `Qiskit Textbook <https://qiskit.org/textbook/ch-algorithms/grover.html>`_.
        or the `Qiskit Tutorials
        <https://qiskit.org/documentation/tutorials/algorithms/07_grover_examples.html>`_
        for more concrete how-to examples.

    Grover's Search [1, 2] is a well known quantum algorithm that can be used for
    searching through unstructured collections of records for particular targets
    with quadratic speedup compared to classical algorithms.

    Given a set :math:`X` of :math:`N` elements :math:`X=\{x_1,x_2,\ldots,x_N\}`
    and a boolean function :math:`f : X \rightarrow \{0,1\}`, the goal of an
    unstructured-search problem is to find an element :math:`x^* \in X` such
    that :math:`f(x^*)=1`.

    The search is called *unstructured* because there are no guarantees as to how
    the database is ordered.  On a sorted database, for instance, one could perform
    binary search to find an element in :math:`\mathbb{O}(\log N)` worst-case time.
    Instead, in an unstructured-search problem, there is no prior knowledge about
    the contents of the database. With classical circuits, there is no alternative
    but to perform a linear number of queries to find the target element.
    Conversely, Grover's Search algorithm allows to solve the unstructured-search
    problem on a quantum computer in :math:`\mathcal{O}(\sqrt{N})` queries.

    To carry out this search a so-called oracle is required, that flags a good element/state.
    The action of the oracle :math:`\mathcal{S}_f` is

    .. math::

        \mathcal{S}_f |x\rangle = (-1)^{f(x)} |x\rangle,

    i.e. it flips the phase of the state :math:`|x\rangle` if :math:`x` is a hit.
    The details of how :math:`S_f` works are unimportant to the algorithm; Grover's
    search algorithm treats the oracle as a black box.

    This class supports oracles in form of a :class:`~qiskit.circuit.QuantumCircuit`.

    With the given oracle, Grover's Search constructs the Grover operator to amplify the
    amplitudes of the good states:

    .. math::

        \mathcal{Q} = H^{\otimes n} \mathcal{S}_0 H^{\otimes n} \mathcal{S}_f
                    = D \mathcal{S}_f,

    where :math:`\mathcal{S}_0` flips the phase of the all-zero state and acts as identity
    on all other states. Sometimes the first three operands are summarized as diffusion operator,
    which implements a reflection over the equal superposition state.

    If the number of solutions is known, we can calculate how often :math:`\mathcal{Q}` should be
    applied to find a solution with very high probability, see the method
    `optimal_num_iterations`. If the number of solutions is unknown, the algorithm tries different
    powers of Grover's operator, see the `iterations` argument, and after each iteration checks
    if a good state has been measured using `good_state`.

    The generalization of Grover's Search, Quantum Amplitude Amplification [3], uses a modified
    version of :math:`\mathcal{Q}` where the diffusion operator does not reflect about the
    equal superposition state, but another state specified via an operator :math:`\mathcal{A}`:

    .. math::

        \mathcal{Q} = \mathcal{A} \mathcal{S}_0 \mathcal{A}^\dagger \mathcal{S}_f.

    For more information, see the :class:`~qiskit.circuit.library.GroverOperator` in the
    circuit library.

    References:
        [1]: L. K. Grover (1996), A fast quantum mechanical algorithm for database search,
            `arXiv:quant-ph/9605043 <https://arxiv.org/abs/quant-ph/9605043>`_.
        [2]: I. Chuang & M. Nielsen, Quantum Computation and Quantum Information,
            Cambridge: Cambridge University Press, 2000. Chapter 6.1.2.
        [3]: Brassard, G., Hoyer, P., Mosca, M., & Tapp, A. (2000).
            Quantum Amplitude Amplification and Estimation.
            `arXiv:quant-ph/0005055 <http://arxiv.org/abs/quant-ph/0005055>`_.
    """

    def __init__(
        self,
        iterations: list[int] | Iterator[int] | int | None = None,
        growth_rate: float | None = None,
        sample_from_iterations: bool = False,
    ) -> None:
        r"""
        Args:
            iterations: Specify the number of iterations/power of Grover's operator to be checked.
                * If an int, only one circuit is run with that power of the Grover operator.
                If the number of solutions is known, this option should be used with the optimal
                power. The optimal power can be computed with ``Grover.optimal_num_iterations``.
                * If a list, all the powers in the list are run in the specified order.
                * If an iterator, the powers yielded by the iterator are checked, until a maximum
                number of iterations or maximum power is reached.
                * If ``None``, the :obj:`AmplificationProblem` provided must have an ``is_good_state``,
                and circuits are run until that good state is reached.
            growth_rate: If specified, the iterator is set to increasing powers of ``growth_rate``,
                i.e. to ``int(growth_rate ** 1), int(growth_rate ** 2), ...`` until a maximum
                number of iterations is reached.
            sample_from_iterations: If True, instead of taking the values in ``iterations`` as
                powers of the Grover operator, a random integer sample between 0 and smaller value
                than the iteration is used as a power, see [1], Section 4.
            sampler: A Sampler to use for sampling the results of the circuits.

        Raises:
            ValueError: If ``growth_rate`` is a float but not larger than 1.
            ValueError: If both ``iterations`` and ``growth_rate`` is set.

        References:
            [1]: Boyer et al., Tight bounds on quantum searching
                 `<https://arxiv.org/abs/quant-ph/9605034>`_
        """
        # set default value
        if growth_rate is None and iterations is None:
            growth_rate = 1.2

        if growth_rate is not None and iterations is not None:
            raise ValueError("Pass either a value for iterations or growth_rate, not both.")

        if growth_rate is not None:
            # yield iterations ** 1, iterations ** 2, etc. and casts to int
            self._iterations: Generator[int, None, None] | list[int] = (
                int(growth_rate**x) for x in itertools.count(1)
            )
        elif isinstance(iterations, int):
            self._iterations = [iterations]
        else:
            self._iterations = iterations  # type: ignore[assignment]

        self._sample_from_iterations = sample_from_iterations
        self._iterations_arg = iterations

    @staticmethod
    def optimal_num_iterations(num_solutions: int, num_qubits: int) -> int:
        """Return the optimal number of iterations, if the number of solutions is known.

        Args:
            num_solutions: The number of solutions.
            num_qubits: The number of qubits used to encode the states.

        Returns:
            The optimal number of iterations for Grover's algorithm to succeed.
        """
        amplitude = np.sqrt(num_solutions / 2**num_qubits)
        return round(np.arccos(amplitude) / (2 * np.arcsin(amplitude)))

    def construct_circuit(
        self, problem: AmplificationProblem, power: int | None = None, measurement: bool = False
    ) -> QuantumCircuit:
        """Construct the circuit for Grover's algorithm with ``power`` Grover operators.

        Args:
            problem: The amplification problem for the algorithm.
            power: The number of times the Grover operator is repeated. If None, this argument
                is set to the first item in ``iterations``.
            measurement: Boolean flag to indicate if measurement should be included in the circuit.

        Returns:
            QuantumCircuit: the QuantumCircuit object for the constructed circuit

        Raises:
            ValueError: If no power is passed and the iterations are not an integer.
        """
        if power is None:
            if len(self._iterations) > 1:  # type: ignore[arg-type]
                raise ValueError("Please pass ``power`` if the iterations are not an integer.")
            power = self._iterations[0]  # type: ignore[index]

        qc = QuantumCircuit(problem.oracle.num_qubits, name="Grover circuit")
        qc.compose(problem.state_preparation, inplace=True)
        if power > 0:
            qc.compose(problem.grover_operator.power(power), inplace=True)

        if measurement:
            measurement_cr = ClassicalRegister(len(problem.objective_qubits))
            qc.add_register(measurement_cr)
            qc.measure(problem.objective_qubits, measurement_cr)

        return qc
