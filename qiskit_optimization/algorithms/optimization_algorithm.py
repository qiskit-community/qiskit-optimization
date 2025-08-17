# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2020, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An abstract class for optimization algorithms in Qiskit optimization module."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from logging import getLogger
from typing import Any, cast

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.result import QuasiDistribution

from ..converters.quadratic_program_to_qubo import QuadraticProgramConverter, QuadraticProgramToQubo
from ..exceptions import QiskitOptimizationError
from ..problems.quadratic_program import QuadraticProgram, Variable

logger = getLogger(__name__)


class OptimizationResultStatus(Enum):
    """Termination status of an optimization algorithm."""

    SUCCESS = 0
    """the optimization algorithm succeeded to find a feasible solution."""

    FAILURE = 1
    """the optimization algorithm ended in a failure."""

    INFEASIBLE = 2
    """the optimization algorithm obtained an infeasible solution."""


@dataclass
class SolutionSample:
    """A sample of an optimization solution."""

    x: np.ndarray
    """The values of the variables"""
    fval: float
    """The objective function value"""
    probability: float
    """The probability of this sample"""
    status: OptimizationResultStatus
    """The status of this sample"""


class OptimizationResult:
    """A base class for optimization results.

    The optimization algorithms return an object of the type ``OptimizationResult``
    with the information about the solution obtained.

    ``OptimizationResult`` allows users to get the value of a variable by specifying an index or
    a name as follows.

    Examples:
        >>> from qiskit_optimization import QuadraticProgram
        >>> from qiskit_optimization.algorithms import CplexOptimizer
        >>> problem = QuadraticProgram()
        >>> _ = problem.binary_var('x1')
        >>> _ = problem.binary_var('x2')
        >>> _ = problem.binary_var('x3')
        >>> problem.minimize(linear={'x1': 1, 'x2': -2, 'x3': 3})
        >>> print([var.name for var in problem.variables])
        ['x1', 'x2', 'x3']
        >>> optimizer = CplexOptimizer()
        >>> result = optimizer.solve(problem)
        >>> print(result.variable_names)
        ['x1', 'x2', 'x3']
        >>> print(result.x)
        [0. 1. 0.]
        >>> print(result[1])
        1.0
        >>> print(result['x1'])
        0.0
        >>> print(result.fval)
        -2.0
        >>> print(result.variables_dict)
        {'x1': 0.0, 'x2': 1.0, 'x3': 0.0}

    Note:
        The order of variables should be equal to that of the problem solved by
        optimization algorithms. Optimization algorithms and converters of ``QuadraticProgram``
        should maintain the order when generating a new ``OptimizationResult`` object.
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        x: list[float] | np.ndarray | None,
        fval: float | None,
        variables: list[Variable],
        status: OptimizationResultStatus,
        raw_results: Any | None = None,
        samples: list[SolutionSample] | None = None,
    ) -> None:
        """
        Args:
            x: the variable values found in the optimization, or possibly None in case of FAILURE.
            fval: the objective function value.
            variables: the list of variables of the optimization problem.
            raw_results: the original results object from the optimization algorithm.
            status: the termination status of the optimization algorithm.
            samples: the solution samples.

        Raises:
            QiskitOptimizationError: if sizes of ``x`` and ``variables`` do not match.
        """
        self._variables = variables
        self._variable_names = [var.name for var in self._variables]
        if x is None:
            # if no state is given, it is set to None
            self._x = None  # pylint: disable=invalid-name
            self._variables_dict = None
        else:
            if len(x) != len(variables):
                raise QiskitOptimizationError(
                    f"Inconsistent size of variable values (x) and variables. x: size {len(x)} {x}, "
                    f"variables: size {len(variables)} {[v.name for v in variables]}"
                )
            self._x = np.asarray(x)
            self._variables_dict = {
                name: val.item() for name, val in zip(self._variable_names, self._x)
            }

        self._fval = fval
        self._raw_results = raw_results
        self._status = status
        if samples:
            sum_prob = np.sum([e.probability for e in samples])
            if not np.isclose(sum_prob, 1.0):
                logger.debug("The sum of probability of samples is not close to 1: %f", sum_prob)
            self._samples = samples
        else:
            self._samples = [
                SolutionSample(x=cast(np.ndarray, x), fval=fval, status=status, probability=1.0)
            ]

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {str(self)}>"

    def __str__(self) -> str:
        variables = ", ".join([f"{var}={x}" for var, x in self._variables_dict.items()])
        return f"fval={self._fval}, {variables}, status={self._status.name}"

    def prettyprint(self) -> str:
        """Returns a pretty printed string of this optimization result.

        Returns:
            A pretty printed string representing the result.
        """
        variables = ", ".join([f"{var}={x}" for var, x in self._variables_dict.items()])
        return (
            f"objective function value: {self._fval}\n"
            f"variable values: {variables}\n"
            f"status: {self._status.name}"
        )

    def __getitem__(self, key: int | str) -> float:
        """Returns the value of the variable whose index or name is equal to ``key``.

        The key can be an integer or a string.
        If the key is an integer, this methods returns the value of the variable
        whose index is equal to ``key``.
        If the key is a string, this methods return the value of the variable
        whose name is equal to ``key``.

        Args:
            key: an integer or a string.

        Returns:
            The value of a variable whose index or name is equal to ``key``.

        Raises:
            IndexError: if ``key`` is an integer and is out of range of the variables.
            KeyError: if ``key`` is a string and none of the variables has ``key`` as name.
            TypeError: if ``key`` is neither an integer nor a string.
        """
        if isinstance(key, int):
            return self._x[key]
        if isinstance(key, str):
            return self._variables_dict[key]
        raise TypeError(f"Integer or string key required, instead {type(key)}({key}) provided.")

    def get_correlations(self) -> np.ndarray:
        """
        Get <Zi x Zj> correlation matrix from the samples.

        Returns:
            A correlation matrix.
        """

        states = [v.x for v in self.samples]
        probs = [v.probability for v in self.samples]

        n = len(states[0])
        correlations = np.zeros((n, n))
        for k, prob in enumerate(probs):
            b = states[k]
            for i in range(n):
                for j in range(i):
                    if b[i] == b[j]:
                        correlations[i, j] += prob
                    else:
                        correlations[i, j] -= prob
        return correlations

    @property
    def x(self) -> np.ndarray | None:
        """Returns the variable values found in the optimization or None in case of FAILURE.

        Returns:
            The variable values found in the optimization.
        """
        return self._x

    @property
    def fval(self) -> float | None:
        """Returns the objective function value.

        Returns:
            The function value corresponding to the objective function value found in the optimization.
        """
        return self._fval

    @property
    def raw_results(self) -> Any:
        """Return the original results object from the optimization algorithm.

        Currently a dump for any leftovers.

        Returns:
            Additional result information of the optimization algorithm.
        """
        return self._raw_results

    @property
    def status(self) -> OptimizationResultStatus:
        """Returns the termination status of the optimization algorithm.

        Returns:
            The termination status of the algorithm.
        """
        return self._status

    @property
    def variables(self) -> list[Variable]:
        """Returns the list of variables of the optimization problem.

        Returns:
            The list of variables.
        """
        return self._variables

    @property
    def variables_dict(self) -> dict[str, float]:
        """Returns the variable values as a dictionary of the variable name and corresponding value.

        Returns:
            The variable values as a dictionary of the variable name and corresponding value.
        """
        return self._variables_dict

    @property
    def variable_names(self) -> list[str]:
        """Returns the list of variable names of the optimization problem.

        Returns:
            The list of variable names of the optimization problem.
        """
        return self._variable_names

    @property
    def samples(self) -> list[SolutionSample]:
        """Returns the list of solution samples

        Returns:
            The list of solution samples.
        """
        return self._samples


class OptimizationAlgorithm(ABC):
    """An abstract class for optimization algorithms in Qiskit optimization module."""

    _MIN_PROBABILITY = 1e-6

    @abstractmethod
    def get_compatibility_msg(self, problem: QuadraticProgram) -> str:
        """Checks whether a given problem can be solved with the optimizer implementing this method.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            Returns the incompatibility message. If the message is empty no issues were found.
        """

    def is_compatible(self, problem: QuadraticProgram) -> bool:
        """Checks whether a given problem can be solved with the optimizer implementing this method.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            Returns True if the problem is compatible, False otherwise.
        """
        return len(self.get_compatibility_msg(problem)) == 0

    @abstractmethod
    def solve(self, problem: QuadraticProgram) -> OptimizationResult:
        """Tries to solves the given problem using the optimizer.

        Runs the optimizer to try to solve the optimization problem.

        Args:
            problem: The problem to be solved.

        Returns:
            The result of the optimizer applied to the problem.

        Raises:
            QiskitOptimizationError: If the problem is incompatible with the optimizer.
        """
        raise NotImplementedError

    def _verify_compatibility(self, problem: QuadraticProgram) -> None:
        """Verifies that the problem is suitable for this optimizer. If the problem is not
        compatible then an exception is raised. This method is for convenience for concrete
        optimizers and is not intended to be used by end user.

        Args:
            problem: Problem to verify.

        Returns:
            None

        Raises:
            QiskitOptimizationError: If the problem is incompatible with the optimizer.

        """
        # check compatibility and raise exception if incompatible
        msg = self.get_compatibility_msg(problem)
        if msg:
            raise QiskitOptimizationError(f"Incompatible problem: {msg}")

    @staticmethod
    def _get_feasibility_status(
        problem: QuadraticProgram, x: list[float] | np.ndarray
    ) -> OptimizationResultStatus:
        """Returns whether the input result is feasible or not for the given problem.

        Args:
            problem: Problem to verify.
            x: the input result list.

        Returns:
            The status of the result.
        """
        is_feasible = problem.is_feasible(x)

        return (
            OptimizationResultStatus.SUCCESS if is_feasible else OptimizationResultStatus.INFEASIBLE
        )

    @staticmethod
    def _prepare_converters(
        converters: QuadraticProgramConverter | list[QuadraticProgramConverter] | None,
        penalty: float | None = None,
    ) -> list[QuadraticProgramConverter]:
        """Prepare a list of converters from the input.

        Args:
            converters: The converters to use for converting a problem into a different form.
                By default, when None is specified, an internally created instance of
                :class:`~qiskit_optimization.converters.QuadraticProgramToQubo` will be used.
            penalty: The penalty factor used in the default
                :class:`~qiskit_optimization.converters.QuadraticProgramToQubo` converter

        Returns:
            The list of converters.

        Raises:
            TypeError: When the converters include those that are not
            :class:`~qiskit_optimization.converters.QuadraticProgramConverter type.
        """
        if converters is None:
            return [QuadraticProgramToQubo(penalty=penalty)]
        elif isinstance(converters, QuadraticProgramConverter):
            return [converters]
        elif isinstance(converters, list) and all(
            isinstance(converter, QuadraticProgramConverter) for converter in converters
        ):
            return converters
        else:
            raise TypeError("`converters` must all be of the QuadraticProgramConverter type")

    @staticmethod
    def _convert(
        problem: QuadraticProgram,
        converters: QuadraticProgramConverter | list[QuadraticProgramConverter],
    ) -> QuadraticProgram:
        """Convert the problem with the converters

        Args:
            problem: The problem to be solved
            converters: The converters to use for converting a problem into a different form.

        Returns:
            The problem converted by the converters.
        """
        problem_ = problem

        if not isinstance(converters, list):
            converters = [converters]

        for converter in converters:
            problem_ = converter.convert(problem_)

        return problem_

    @staticmethod
    def _check_converters(
        converters: QuadraticProgramConverter | list[QuadraticProgramConverter] | None,
    ) -> list[QuadraticProgramConverter]:
        if converters is None:
            converters = []
        if not isinstance(converters, list):
            converters = [converters]
        if not all(isinstance(conv, QuadraticProgramConverter) for conv in converters):
            raise TypeError(f"Invalid object of converters: {converters}")
        return converters

    @classmethod
    def _interpret(
        cls,
        x: np.ndarray,
        problem: QuadraticProgram,
        converters: QuadraticProgramConverter | list[QuadraticProgramConverter] | None = None,
        result_class: type[OptimizationResult] = OptimizationResult,
        **kwargs,
    ) -> OptimizationResult:
        """Convert back the result of the converted problem to the result of the original problem.

        Args:
            x: The result of the converted problem.
            converters: The converters to use for converting back the result of the problem
                to the result of the original problem.
            problem: The original problem for which `x` is interpreted.
            result_class: The class of the result object.
            kwargs: parameters of the constructor of result_class

        Returns:
            The result of the original problem.

        Raises:
            QiskitOptimizationError: if result_class is not a sub-class of OptimizationResult.
            TypeError: if converters are not QuadraticProgramConverter or a list of
                QuadraticProgramConverter.
        """
        if not issubclass(result_class, OptimizationResult):
            raise QiskitOptimizationError(
                f"Invalid result class, not derived from OptimizationResult: {result_class}"
            )
        converters = cls._check_converters(converters)

        for converter in converters[::-1]:
            x = converter.interpret(x)
        return result_class(
            x=x,
            fval=problem.objective.evaluate(x),
            variables=problem.variables,
            status=cls._get_feasibility_status(problem, x),
            **kwargs,
        )

    @classmethod
    def _interpret_samples(
        cls,
        problem: QuadraticProgram,
        raw_samples: list[SolutionSample],
        converters: QuadraticProgramConverter | list[QuadraticProgramConverter] | None = None,
    ) -> tuple[list[SolutionSample], SolutionSample]:
        """Interpret and sort all samples and return the raw sample corresponding to the best one"""
        converters = cls._check_converters(converters)

        prob: dict[tuple, float] = {}
        array = {}
        index = {}
        for i, sample in enumerate(raw_samples):
            x = sample.x
            for converter in converters[::-1]:
                x = converter.interpret(x)
            key = tuple(x)
            prob[key] = prob.get(key, 0.0) + sample.probability
            array[key] = x
            index[key] = i

        samples = []
        for key, x in array.items():
            probability = prob[key]
            fval = problem.objective.evaluate(x)
            status = cls._get_feasibility_status(problem, x)
            samples.append(SolutionSample(x, fval, probability, status))

        sorted_samples = sorted(
            samples,
            key=lambda v: (v.status.value, problem.objective.sense.value * v.fval),
        )
        best_raw = raw_samples[index[tuple(sorted_samples[0].x)]]
        return sorted_samples, best_raw

    @staticmethod
    def _eigenvector_to_solutions(
        eigenvector: QuasiDistribution | Statevector | dict | np.ndarray,
        qubo: QuadraticProgram,
        min_probability: float = _MIN_PROBABILITY,
    ) -> list[SolutionSample]:
        """Convert the eigenvector to the bitstrings and corresponding eigenvalues.

        Args:
            eigenvector: The eigenvector from which the solution states are extracted.
            qubo: The QUBO to evaluate at the bitstring.
            min_probability: Only consider states where the amplitude exceeds this threshold.

        Returns:
            For each computational basis state contained in the eigenvector, return the basis
            state as bitstring along with the QUBO evaluated at that bitstring and the
            probability of sampling this bitstring from the eigenvector.

        Raises:
            TypeError: If the type of eigenvector is not supported.
        """

        def generate_solution(bitstr, qubo, probability):
            x = np.fromiter(list(bitstr[::-1]), dtype=int)
            fval = qubo.objective.evaluate(x)
            return SolutionSample(
                x=x,
                fval=fval,
                probability=probability,
                status=OptimizationResultStatus.SUCCESS,
            )

        solutions = []
        if isinstance(eigenvector, QuasiDistribution):
            probabilities = eigenvector.binary_probabilities()
            # iterate over all samples
            for bitstr, sampling_probability in probabilities.items():
                # add the bitstring, if the sampling probability exceeds the threshold
                if sampling_probability >= min_probability:
                    solutions.append(generate_solution(bitstr, qubo, sampling_probability))

        elif isinstance(eigenvector, Statevector):
            probabilities = eigenvector.probabilities()
            num_qubits = eigenvector.num_qubits
            # iterate over all states and their sampling probabilities
            for i, sampling_probability in enumerate(probabilities):
                # add the i-th state if the sampling probability exceeds the threshold
                if sampling_probability >= min_probability:
                    bitstr = f"{i:b}".rjust(num_qubits, "0")
                    solutions.append(generate_solution(bitstr, qubo, sampling_probability))

        elif isinstance(eigenvector, dict):
            # When eigenvector is a dict, square the values since the values are normalized.
            # See https://github.com/Qiskit/qiskit-terra/pull/5496 for more details.
            probabilities = {bitstr: val**2 for (bitstr, val) in eigenvector.items()}
            # iterate over all samples
            for bitstr, sampling_probability in probabilities.items():
                # add the bitstring, if the sampling probability exceeds the threshold
                if sampling_probability >= min_probability:
                    solutions.append(generate_solution(bitstr, qubo, sampling_probability))

        elif isinstance(eigenvector, np.ndarray):
            num_qubits = int(np.log2(eigenvector.size))
            probabilities = np.abs(eigenvector * eigenvector.conj())
            # iterate over all states and their sampling probabilities
            for i, sampling_probability in enumerate(probabilities):
                # add the i-th state if the sampling probability exceeds the threshold
                if sampling_probability >= min_probability:
                    bitstr = f"{i:b}".rjust(num_qubits, "0")
                    solutions.append(generate_solution(bitstr, qubo, sampling_probability))

        else:
            raise TypeError(
                f"Eigenvector should be QuasiDistribution, Statevector, dict or numpy.ndarray. "
                f"But, it was {type(eigenvector)}."
            )
        return solutions
