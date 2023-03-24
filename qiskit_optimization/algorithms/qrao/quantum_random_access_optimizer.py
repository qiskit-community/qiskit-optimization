# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quantum Random Access Optimizer."""

from typing import Union, List, Tuple, Optional
import time

import numpy as np

from qiskit import QuantumCircuit
from qiskit.algorithms.minimum_eigensolvers import (
    MinimumEigensolver,
    MinimumEigensolverResult,
    NumPyMinimumEigensolver,
)
from qiskit.algorithms.minimum_eigen_solvers import (
    MinimumEigensolver as LegacyMinimumEigensolver,
    MinimumEigensolverResult as LegacyMinimumEigensolverResult,
    NumPyMinimumEigensolver as LegacyNumPyMinimumEigensolver,
)

from qiskit_optimization.algorithms import (
    OptimizationAlgorithm,
    OptimizationResult,
    OptimizationResultStatus,
    SolutionSample,
)
from qiskit_optimization.problems import QuadraticProgram, Variable

from .encoding import QuantumRandomAccessEncoding
from .rounding_common import RoundingScheme, RoundingContext, RoundingResult
from .semideterministic_rounding import SemideterministicRounding


def _get_aux_operators_evaluated(relaxed_results):
    try:
        # Must be using the new "minimum_eigensolvers"
        # https://github.com/Qiskit/qiskit-terra/blob/main/releasenotes/notes/0.22/add-eigensolvers-with-primitives-8b3a9f55f5fd285f.yaml
        return relaxed_results.aux_operators_evaluated
    except AttributeError:
        # Must be using the old (deprecated) "minimum_eigen_solvers"
        return relaxed_results.aux_operator_eigenvalues


class QuantumRandomAccessOptimizationResult(OptimizationResult):
    """Result of Quantum Random Access Optimization procedure."""

    def __init__(
        self,
        *,
        x: Optional[Union[List[float], np.ndarray]],
        fval: Optional[float],
        variables: List[Variable],
        status: OptimizationResultStatus,
        samples: Optional[List[SolutionSample]],
        relaxed_results: Union[
            MinimumEigensolverResult, LegacyMinimumEigensolverResult
        ],
        rounding_results: RoundingResult,
        relaxed_results_offset: float,
        sense: int,
    ) -> None:
        """
        Args:
            x: the optimal value found by ``MinimumEigensolver``.
            fval: the optimal function value.
            variables: the list of variables of the optimization problem.
            status: the termination status of the optimization algorithm.
            min_eigen_solver_result: the result obtained from the underlying algorithm.
            samples: the x values of the QUBO, the objective function value of the QUBO,
                and the probability, and the status of sampling.
        """
        super().__init__(
            x=x,
            fval=fval,
            variables=variables,
            status=status,
            raw_results=None,
            samples=samples,
        )
        self._relaxed_results = relaxed_results
        self._rounding_results = rounding_results
        self._relaxed_results_offset = relaxed_results_offset
        assert sense in (-1, 1)
        self._sense = sense

    @property
    def relaxed_results(
        self,
    ) -> Union[MinimumEigensolverResult, LegacyMinimumEigensolverResult]:
        """Variationally obtained ground state of the relaxed Hamiltonian"""
        return self._relaxed_results

    @property
    def rounding_results(self) -> RoundingResult:
        """Rounding results"""
        return self._rounding_results

    @property
    def trace_values(self):
        """List of expectation values, one corresponding to each decision variable"""
        trace_values = [
            v[0] for v in _get_aux_operators_evaluated(self._relaxed_results)
        ]
        return trace_values

    @property
    def relaxed_fval(self) -> float:
        """Relaxed function value, in the conventions of the original ``QuadraticProgram``

        Restoring convertions may be necessary, for instance, if the provided
        ``QuadraticProgram`` represents a maximization problem, as it will be
        converted to a minimization problem when phrased as a Hamiltonian.
        """
        return self._sense * (
            self._relaxed_results_offset + self.relaxed_results.eigenvalue.real
        )

    def __repr__(self) -> str:
        lines = (
            "QRAO Result",
            "-----------",
            f"relaxed function value: {self.relaxed_fval}",
            super().__repr__(),
        )
        return "\n".join(lines)


class QuantumRandomAccessOptimizer(OptimizationAlgorithm):
    """Quantum Random Access Optimizer."""

    def __init__(
        self,
        min_eigen_solver: Union[MinimumEigensolver, LegacyMinimumEigensolver],
        encoding: QuantumRandomAccessEncoding,
        rounding_scheme: Optional[RoundingScheme] = None,
    ):
        """
        Args:

            min_eigen_solver: The minimum eigensolver to use for solving the
                relaxed problem (typically an instance of ``VQE`` or ``QAOA``).

            encoding: The ``QuantumRandomAccessEncoding``, which must have
                already been ``encode()``ed with a ``QuadraticProgram``.

            rounding_scheme: The rounding scheme.  If ``None`` is provided,
                ``SemideterministicRounding()`` will be used.

        """
        self.min_eigen_solver = min_eigen_solver
        self.encoding = encoding
        if rounding_scheme is None:
            rounding_scheme = SemideterministicRounding()
        self.rounding_scheme = rounding_scheme

    @property
    def min_eigen_solver(self) -> Union[MinimumEigensolver, LegacyMinimumEigensolver]:
        """The minimum eigensolver."""
        return self._min_eigen_solver

    @min_eigen_solver.setter
    def min_eigen_solver(
        self, min_eigen_solver: Union[MinimumEigensolver, LegacyMinimumEigensolver]
    ) -> None:
        """Set the minimum eigensolver."""
        if not min_eigen_solver.supports_aux_operators():
            raise TypeError(
                f"The provided MinimumEigensolver ({type(min_eigen_solver)}) "
                "does not support auxiliary operators."
            )
        self._min_eigen_solver = min_eigen_solver

    @property
    def encoding(self) -> QuantumRandomAccessEncoding:
        """The encoding."""
        return self._encoding

    @encoding.setter
    def encoding(self, encoding: QuantumRandomAccessEncoding) -> None:
        """Set the encoding"""
        if encoding.num_qubits == 0:
            raise ValueError(
                "The passed encoder has no variables associated with it; you probably "
                "need to call `encode()` to encode it with a `QuadraticProgram`."
            )
        # Instead of copying, we "freeze" the encoding to ensure it is not
        # modified going forward.
        encoding.freeze()
        self._encoding = encoding

    def get_compatibility_msg(self, problem: QuadraticProgram) -> str:
        if problem != self.encoding.problem:
            return (
                "The problem passed does not match the problem used "
                "to construct the QuantumRandomAccessEncoding."
            )
        return ""

    def solve_relaxed(
        self,
    ) -> Tuple[
        Union[MinimumEigensolverResult, LegacyMinimumEigensolverResult], RoundingContext
    ]:
        """Solve the relaxed Hamiltonian given the ``encoding`` provided to the constructor."""
        # Get the ordered list of operators that correspond to each decision
        # variable.  This line assumes the variables are numbered consecutively
        # starting with 0.  Note that under this assumption, the following
        # range is equivalent to `sorted(self.encoding.var2op.keys())`.  See
        # encoding.py for more commentary on this assumption, which always
        # holds when starting from a `QuadraticProgram`.
        variable_ops = [self.encoding.term2op(i) for i in range(self.encoding.num_vars)]

        # solve relaxed problem
        start_time_relaxed = time.time()
        relaxed_results = self.min_eigen_solver.compute_minimum_eigenvalue(
            self.encoding.qubit_op, aux_operators=variable_ops
        )
        stop_time_relaxed = time.time()
        relaxed_results.time_taken = stop_time_relaxed - start_time_relaxed

        trace_values = [v[0] for v in _get_aux_operators_evaluated(relaxed_results)]

        # Collect inputs for rounding
        # double check later that there's no funny business with the
        # parameter ordering.

        # If the relaxed solution can be expressed as an explicit circuit
        # then always express it that way - even if a statevector simulator
        # was used and the actual wavefunction could be used. The only exception
        # is the numpy eigensolver. If you wish to round the an explicit statevector,
        # you must do so by manually rounding and passing in a QuantumCircuit
        # initialized to the desired state.
        if hasattr(self.min_eigen_solver, "ansatz"):
            circuit = self.min_eigen_solver.ansatz.bind_parameters(
                relaxed_results.optimal_point
            )
        elif isinstance(
            self.min_eigen_solver,
            (NumPyMinimumEigensolver, LegacyNumPyMinimumEigensolver),
        ):
            statevector = relaxed_results.eigenstate
            if isinstance(self.min_eigen_solver, LegacyNumPyMinimumEigensolver):
                # statevector is a StateFn in this case, so we must convert it
                # to a Statevector
                statevector = statevector.primitive
            circuit = QuantumCircuit(self.encoding.num_qubits)
            circuit.initialize(statevector)
        else:
            circuit = None

        rounding_context = RoundingContext(
            encoding=self.encoding,
            trace_values=trace_values,
            circuit=circuit,
        )

        return relaxed_results, rounding_context

    def solve(self, problem: Optional[QuadraticProgram] = None) -> OptimizationResult:
        if problem is None:
            problem = self.encoding.problem
        else:
            if problem != self.encoding.problem:
                raise ValueError(
                    "The problem given must exactly match the problem "
                    "used to generate the encoded operator. Alternatively, "
                    "the argument to `solve` can be left blank."
                )

        # Solve relaxed problem
        # ============================
        (relaxed_results, rounding_context) = self.solve_relaxed()

        # Round relaxed solution
        # ============================
        rounding_results = self.rounding_scheme.round(rounding_context)

        # Process rounding results
        # ============================
        # The rounding classes don't have enough information to evaluate the
        # objective function, so they return a RoundingSolutionSample, which
        # contains only part of the information in the SolutionSample.  Here we
        # fill in the rest.
        samples: List[SolutionSample] = []
        for sample in rounding_results.samples:
            samples.append(
                SolutionSample(
                    x=sample.x,
                    fval=problem.objective.evaluate(sample.x),
                    probability=sample.probability,
                    status=self._get_feasibility_status(problem, sample.x),
                )
            )

        # TODO: rewrite this logic once the converters are integrated.
        # we need to be very careful about ensuring that the problem
        # sense is taken into account in the relaxed solution and the rounding
        # this is likely only a temporary patch while we are sticking to a
        # maximization problem.
        fsense = {"MINIMIZE": min, "MAXIMIZE": max}[problem.objective.sense.name]
        best_sample = fsense(samples, key=lambda x: x.fval)

        return QuantumRandomAccessOptimizationResult(
            samples=samples,
            x=best_sample.x,
            fval=best_sample.fval,
            variables=problem.variables,
            status=OptimizationResultStatus.SUCCESS,
            relaxed_results=relaxed_results,
            rounding_results=rounding_results,
            relaxed_results_offset=self.encoding.offset,
            sense=problem.objective.sense.value,
        )
