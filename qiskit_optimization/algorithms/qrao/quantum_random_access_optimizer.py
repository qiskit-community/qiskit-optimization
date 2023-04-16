# This code is part of Qiskit.
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

"""Quantum Random Access Optimizer class."""

import time
from typing import List, Optional, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.algorithms.minimum_eigensolvers import (MinimumEigensolver,
                                                    MinimumEigensolverResult,
                                                    NumPyMinimumEigensolver)

from qiskit_optimization.algorithms import (OptimizationResult,
                                            OptimizationResultStatus,
                                            SolutionSample)
from qiskit_optimization.problems import Variable

from .quantum_random_access_encoding import QuantumRandomAccessEncoding
from .rounding_common import RoundingContext, RoundingResult, RoundingScheme
from .semideterministic_rounding import SemideterministicRounding


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
        relaxed_fval: float,
        relaxed_results: MinimumEigensolverResult,
        rounding_results: RoundingResult,
    ) -> None:
        """
        Args:
            x: The optimal value found by ``MinimumEigensolver``.
            fval: The optimal function value.
            variables: The list of variables of the optimization problem.
            status: The termination status of the optimization algorithm.
            samples: The list of ``SolutionSample`` obtained from the optimization algorithm.
            relaxed_results: The result obtained from the underlying minimum eigensolver.
            rounding_results: The rounding results.
        """
        super().__init__(
            x=x,
            fval=fval,
            variables=variables,
            status=status,
            raw_results=None,
            samples=samples,
        )
        self._relaxed_fval = relaxed_fval
        self._relaxed_results = relaxed_results
        self._rounding_results = rounding_results

    @property
    def relaxed_results(
        self,
    ) -> MinimumEigensolverResult:
        """The result obtained from the underlying minimum eigensolver."""
        return self._relaxed_results

    @property
    def rounding_results(self) -> RoundingResult:
        """The rounding results."""
        return self._rounding_results

    @property
    def trace_values(self):
        """List of expectation values, one corresponding to each decision variable"""
        trace_values = [v[0] for v in self._relaxed_results.aux_operators_evaluated]
        return trace_values

    @property
    def relaxed_fval(self) -> float:
        """Relaxed function value, in the conventions of the original ``QuadraticProgram``."""
        return self._relaxed_fval


class QuantumRandomAccessOptimizer:
    """Quantum Random Access Optimizer class."""

    def __init__(
        self,
        min_eigen_solver: MinimumEigensolver,
        rounding_scheme: Optional[RoundingScheme] = None,
    ):
        """
        Args:
            min_eigen_solver: The minimum eigensolver to use for solving the relaxed problem.
            rounding_scheme: The rounding scheme.  If ``None`` is provided,
                ``SemideterministicRounding()`` will be used.

        """
        self.min_eigen_solver = min_eigen_solver
        if rounding_scheme is None:
            rounding_scheme = SemideterministicRounding()
        self.rounding_scheme = rounding_scheme

    @property
    def min_eigen_solver(self) -> MinimumEigensolver:
        """Return the minimum eigensolver."""
        return self._min_eigen_solver

    @min_eigen_solver.setter
    def min_eigen_solver(self, min_eigen_solver: MinimumEigensolver) -> None:
        """Set the minimum eigensolver."""
        if not min_eigen_solver.supports_aux_operators():
            raise TypeError(
                f"The provided MinimumEigensolver ({type(min_eigen_solver)}) "
                "does not support auxiliary operators."
            )
        self._min_eigen_solver = min_eigen_solver

    def solve_relaxed(
        self,
        encoding: QuantumRandomAccessEncoding,
    ) -> Tuple[MinimumEigensolverResult, RoundingContext]:
        """Solve the relaxed Hamiltonian given by the encoding.

        Args:
            encoding: The ``QuantumRandomAccessEncoding``, which must have already been ``encode()``ed
                with a ``QuadraticProgram``.

        Returns:
            The result of the minimum eigensolver, and the rounding context.

        Raises:
            ValueError: If the encoding has not been encoded with a ``QuadraticProgram``.
        """
        if not encoding.frozen:
            raise ValueError(
                "The encoding must call ``encode()`` with a ``QuadraticProgram`` before being passed"
                "to the QuantumRandomAccessOptimizer."
            )

        # Get the list of operators that correspond to each decision variable.
        variable_ops = [encoding._term2op(i) for i in range(encoding.num_vars)]

        # Solve the relaxed problem
        start_time_relaxed = time.time()
        relaxed_results = self.min_eigen_solver.compute_minimum_eigenvalue(
            encoding.qubit_op, aux_operators=variable_ops
        )
        relaxed_results.time_taken = time.time() - start_time_relaxed

        # Get auxiliary trace values for rounding.
        expectation_values = [v[0] for v in relaxed_results.aux_operators_evaluated]

        # Get the circuit corresponding to the relaxed solution.
        if hasattr(self.min_eigen_solver, "ansatz"):
            circuit = self.min_eigen_solver.ansatz.bind_parameters(relaxed_results.optimal_point)
        elif isinstance(self.min_eigen_solver, NumPyMinimumEigensolver):
            statevector = relaxed_results.eigenstate
            circuit = QuantumCircuit(encoding.num_qubits)
            circuit.initialize(statevector)
        else:
            circuit = None

        rounding_context = RoundingContext(
            encoding=encoding,
            expectation_values=expectation_values,
            circuit=circuit,
        )

        return relaxed_results, rounding_context

    def solve(self, encoding: QuantumRandomAccessEncoding) -> QuantumRandomAccessOptimizationResult:
        """Solve the relaxed Hamiltonian given by the encoding and round the solution by the given
            rounding scheme.

        Args:
            encoding: The ``QuantumRandomAccessEncoding``, which must have already been encoded
                with a ``QuadraticProgram``.
        Returns:
            The result of the quantum random access optimization.

        Raises:
            ValueError: If the encoding has not been encoded with a ``QuadraticProgram``.
        """
        if not encoding.frozen:
            raise ValueError(
                "The encoding must call ``encode()`` with a ``QuadraticProgram`` before being passed"
                "to the QuantumRandomAccessOptimizer."
            )

        # Solve the relaxed problem
        (relaxed_results, rounding_context) = self.solve_relaxed(encoding)

        # Round the solution
        rounding_results = self.rounding_scheme.round(rounding_context)

        # Process rounding results
        samples: List[SolutionSample] = []
        for sample in rounding_results.samples:
            if encoding.problem.is_feasible(sample.x):
                status = OptimizationResultStatus.SUCCESS
            else:
                status = OptimizationResultStatus.INFEASIBLE
            samples.append(
                SolutionSample(
                    x=sample.x,
                    fval=encoding.problem.objective.evaluate(sample.x),
                    probability=sample.probability,
                    status=status,
                )
            )

        # Get the best sample
        fsense = {"MINIMIZE": min, "MAXIMIZE": max}[encoding.problem.objective.sense.name]
        best_sample = fsense(samples, key=lambda x: x.fval)

        relaxed_fval = encoding.problem.objective.sense.value * (
            encoding.offset + relaxed_results.eigenvalue.real
        )

        return QuantumRandomAccessOptimizationResult(
            x=best_sample.x,
            fval=best_sample.fval,
            variables=encoding.problem.variables,
            status=OptimizationResultStatus.SUCCESS,
            samples=samples,
            relaxed_fval=relaxed_fval,
            relaxed_results=relaxed_results,
            rounding_results=rounding_results,
        )
