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
from typing import List, Optional, Tuple, Union, cast

import numpy as np
from qiskit import QuantumCircuit
from qiskit.algorithms.minimum_eigensolvers import (
    MinimumEigensolver,
    MinimumEigensolverResult,
    NumPyMinimumEigensolver,
)

from qiskit_optimization.algorithms import (
    OptimizationAlgorithm,
    OptimizationResult,
    OptimizationResultStatus,
    SolutionSample,
)
from qiskit_optimization.converters import QuadraticProgramConverter, QuadraticProgramToQubo
from qiskit_optimization.problems import QuadraticProgram, Variable

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
        encoding: QuantumRandomAccessEncoding,
        relaxed_fval: float,
        relaxed_result: MinimumEigensolverResult,
        rounding_result: RoundingResult,
    ) -> None:
        """
        Args:
            x: The optimal value found by ``MinimumEigensolver``.
            fval: The optimal function value.
            variables: The list of variables of the optimization problem.
            status: The termination status of the optimization algorithm.
            samples: The list of ``SolutionSample`` obtained from the optimization algorithm.
            encoding: The encoding used for the optimization.
            relaxed_fval: The optimal function value of the relaxed problem.
            relaxed_result: The result obtained from the underlying minimum eigensolver.
            rounding_result: The rounding result.
        """
        super().__init__(
            x=x,
            fval=fval,
            variables=variables,
            status=status,
            raw_results=None,
            samples=samples,
        )
        self._encoding = encoding
        self._relaxed_fval = relaxed_fval
        self._relaxed_result = relaxed_result
        self._rounding_result = rounding_result

    @property
    def encoding(self) -> QuantumRandomAccessEncoding:
        """The encoding used for the optimization."""
        return self._encoding

    @property
    def relaxed_fval(self) -> float:
        """The optimal function value of the relaxed problem."""
        return self._relaxed_fval

    @property
    def relaxed_result(
        self,
    ) -> MinimumEigensolverResult:
        """The result obtained from the underlying minimum eigensolver."""
        return self._relaxed_result

    @property
    def rounding_result(self) -> RoundingResult:
        """The rounding result."""
        return self._rounding_result


class QuantumRandomAccessOptimizer(OptimizationAlgorithm):
    """Quantum Random Access Optimizer class."""

    def __init__(
        self,
        min_eigen_solver: MinimumEigensolver,
        penalty: Optional[float] = None,
        max_vars_per_qubit: int = 3,
        rounding_scheme: Optional[RoundingScheme] = None,
    ):
        """
        Args:
            min_eigen_solver: The minimum eigensolver to use for solving the relaxed problem.
            penalty: The factor that is used to scale the penalty terms corresponding to linear
                equality constraints. If ``None`` is provided, the penalty will be automatically
                determined.
            rounding_scheme: The rounding scheme.  If ``None`` is provided,
                ``SemideterministicRounding()`` will be used.

        """
        self.min_eigen_solver = min_eigen_solver
        self.penalty = penalty
        # Use ``QuadraticProgramToQubo`` to convert the problem to a QUBO.
        self._converters: QuadraticProgramConverter = QuadraticProgramToQubo(penalty=penalty)
        self._max_vars_per_qubit = max_vars_per_qubit
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

    @property
    def max_vars_per_qubit(self) -> int:
        """Return the maximum number of variables per qubit."""
        return self._max_vars_per_qubit

    @max_vars_per_qubit.setter
    def max_vars_per_qubit(self, max_vars_per_qubit: int) -> None:
        """Set the maximum number of variables per qubit."""
        self._max_vars_per_qubit = max_vars_per_qubit

    @property
    def rounding_scheme(self) -> RoundingScheme:
        """Return the rounding scheme."""
        return self._rounding_scheme

    @rounding_scheme.setter
    def rounding_scheme(self, rounding_scheme: RoundingScheme) -> None:
        """Set the rounding scheme."""
        if rounding_scheme is None:
            rounding_scheme = SemideterministicRounding()
        self._rounding_scheme = rounding_scheme

    def get_compatibility_msg(self, problem: QuadraticProgram) -> str:
        """Checks whether a given problem can be solved with this optimizer.

        Checks whether the given problem is compatible, i.e., whether the problem can be converted
        to a QUBO, and otherwise, returns a message explaining the incompatibility.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            A message describing the incompatibility.
        """
        return QuadraticProgramToQubo.get_compatibility_msg(problem)

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
        relaxed_result = self.min_eigen_solver.compute_minimum_eigenvalue(
            encoding.qubit_op, aux_operators=variable_ops
        )
        relaxed_result.time_taken = time.time() - start_time_relaxed

        # Get auxiliary expectation values for rounding.
        if relaxed_result.aux_operators_evaluated is not None:
            expectation_values = [v[0] for v in relaxed_result.aux_operators_evaluated]
        else:
            expectation_values = None

        # Get the circuit corresponding to the relaxed solution.
        if hasattr(self.min_eigen_solver, "ansatz"):
            circuit = self.min_eigen_solver.ansatz.bind_parameters(relaxed_result.optimal_point)
        elif isinstance(self.min_eigen_solver, NumPyMinimumEigensolver):
            statevector = relaxed_result.eigenstate
            circuit = QuantumCircuit(encoding.num_qubits)
            circuit.initialize(statevector)
        else:
            circuit = None

        rounding_context = RoundingContext(
            encoding=encoding,
            expectation_values=expectation_values,
            circuit=circuit,
        )

        return relaxed_result, rounding_context

    def solve(self, problem: QuadraticProgram) -> QuantumRandomAccessOptimizationResult:
        """Solve the relaxed Hamiltonian given by the encoding and round the solution by the given
            rounding scheme.

        Args:
            problem: The ``QuadraticProgram`` to be solved.

        Returns:
            The result of the quantum random access optimization.

        Raises:
            ValueError: If the encoding has not been encoded with a ``QuadraticProgram``.
        """
        # Convert the problem to a QUBO
        self._verify_compatibility(problem)
        qubo = self._convert(problem, self._converters)
        # Encode the QUBO into a quantum random access encoding
        encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=self.max_vars_per_qubit)
        encoding.encode(qubo)

        # Solve the relaxed problem
        (relaxed_result, rounding_context) = self.solve_relaxed(encoding)

        # Round the solution
        rounding_result = self.rounding_scheme.round(rounding_context)

        return self.process_result(problem, encoding, relaxed_result, rounding_result)

    def process_result(
        self,
        problem: QuadraticProgram,
        encoding: QuantumRandomAccessEncoding,
        relaxed_result: MinimumEigensolverResult,
        rounding_result: RoundingResult,
    ) -> QuantumRandomAccessOptimizationResult:
        """Process the relaxed result of the minimum eigensolver and rounding scheme.

        Args:
            problem: The ``QuadraticProgram`` to be solved.
            encoding: The ``QuantumRandomAccessEncoding``, which must have already been ``encode()``ed
                with the corresponding problem.
            relaxed_result: The relaxed result of the minimum eigensolver.
            rounding_result: The result of the rounding scheme.

        Returns:
            The result of the quantum random access optimization.
        """
        samples, best_sol = self._interpret_samples(
            problem=problem, raw_samples=rounding_result.samples
        )

        relaxed_fval = encoding.problem.objective.sense.value * (
            encoding.offset + relaxed_result.eigenvalue.real
        )
        return cast(
            QuantumRandomAccessOptimizationResult,
            self._interpret(
                x=best_sol.x,
                problem=problem,
                result_class=QuantumRandomAccessOptimizationResult,
                samples=samples,
                encoding=encoding,
                relaxed_fval=relaxed_fval,
                relaxed_result=relaxed_result,
                rounding_result=rounding_result,
            ),
        )
