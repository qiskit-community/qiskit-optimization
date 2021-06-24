# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An abstract class for optimization algorithms in Qiskit's optimization module."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Union, Any, Optional, Dict, Type, Tuple, cast
from warnings import warn

import numpy as np

from qiskit.opflow import StateFn, DictStateFn
from ..exceptions import QiskitOptimizationError
from ..converters.quadratic_program_to_qubo import QuadraticProgramToQubo, QuadraticProgramConverter
from ..problems.quadratic_program import QuadraticProgram, Variable


class OptimizationResultStatus(Enum):
    """Termination status of an optimization algorithm."""

    SUCCESS = 0
    """the optimization algorithm succeeded to find an optimal solution."""

    FAILURE = 1
    """the optimization algorithm ended in a failure."""

    INFEASIBLE = 2
    """the optimization algorithm obtained an infeasible solution."""


@dataclass
class SolutionSample:
    """A sample of an optimization solution

    Attributes:
        x: the values of variables
        fval: the objective function value
        probability: the probability of this sample
        status: the status of this sample
    """

    x: np.ndarray
    fval: float
    probability: float
    status: OptimizationResultStatus


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

    def __init__(
        self,
        x: Optional[Union[List[float], np.ndarray]],
        fval: Optional[float],
        variables: List[Variable],
        status: OptimizationResultStatus,
        raw_results: Optional[Any] = None,
        samples: Optional[List[SolutionSample]] = None,
    ) -> None:
        """
        Args:
            x: the optimal value found in the optimization, or possibly None in case of FAILURE.
            fval: the optimal function value.
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
                    f"Inconsistent size of optimal value and variables. x: size {len(x)} {x}, "
                    f"variables: size {len(variables)} {[v.name for v in variables]}"
                )
            self._x = np.asarray(x)
            self._variables_dict = dict(zip(self._variable_names, self._x))

        self._fval = fval
        self._raw_results = raw_results
        self._status = status
        if samples:
            sum_prob = np.sum([e.probability for e in samples])
            if not np.isclose(sum_prob, 1.0):
                warn("The sum of probability of samples is not close to 1: f{sum_prob}")
            self._samples = samples
        else:
            self._samples = [
                SolutionSample(x=cast(np.ndarray, x), fval=fval, status=status, probability=1.0)
            ]

    def __repr__(self) -> str:
        return (
            f"optimal function value: {self._fval}\n"
            f"optimal value: {self._x}\n"
            f"status: {self._status.name}"
        )

    def __getitem__(self, key: Union[int, str]) -> float:
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
    def x(self) -> Optional[np.ndarray]:
        """Returns the optimal value found in the optimization or None in case of FAILURE.

        Returns:
            The optimal value found in the optimization.
        """
        return self._x

    @property
    def fval(self) -> Optional[float]:
        """Returns the optimal function value.

        Returns:
            The function value corresponding to the optimal value found in the optimization.
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
    def variables(self) -> List[Variable]:
        """Returns the list of variables of the optimization problem.

        Returns:
            The list of variables.
        """
        return self._variables

    @property
    def variables_dict(self) -> Dict[str, float]:
        """Returns the optimal value as a dictionary of the variable name and corresponding value.

        Returns:
            The optimal value as a dictionary of the variable name and corresponding value.
        """
        return self._variables_dict

    @property
    def variable_names(self) -> List[str]:
        """Returns the list of variable names of the optimization problem.

        Returns:
            The list of variable names of the optimization problem.
        """
        return self._variable_names

    @property
    def samples(self) -> List[SolutionSample]:
        """Returns the list of solution samples

        Returns:
            The list of solution samples.
        """
        return self._samples


class OptimizationAlgorithm(ABC):
    """An abstract class for optimization algorithms in Qiskit's optimization module."""

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
    def solve(self, problem: QuadraticProgram) -> "OptimizationResult":
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
        problem: QuadraticProgram, x: Union[List[float], np.ndarray]
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
        converters: Optional[Union[QuadraticProgramConverter, List[QuadraticProgramConverter]]],
        penalty: Optional[float] = None,
    ) -> List[QuadraticProgramConverter]:
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
        converters: Union[QuadraticProgramConverter, List[QuadraticProgramConverter]],
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
        converters: Optional[Union[QuadraticProgramConverter, List[QuadraticProgramConverter]]]
    ) -> List[QuadraticProgramConverter]:
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
        converters: Optional[
            Union[QuadraticProgramConverter, List[QuadraticProgramConverter]]
        ] = None,
        result_class: Type[OptimizationResult] = OptimizationResult,
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
        raw_samples: List[SolutionSample],
        converters: List[QuadraticProgramConverter],
    ) -> Tuple[List[SolutionSample], SolutionSample]:
        """Interpret and sort all samples and return the raw sample corresponding to the best one"""
        converters = cls._check_converters(converters)

        prob: Dict[Tuple, float] = {}
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
        eigenvector: Union[dict, np.ndarray, StateFn],
        qubo: QuadraticProgram,
        min_probability: float = 1e-6,
    ) -> List[SolutionSample]:
        """Convert the eigenvector to the bitstrings and corresponding eigenvalues.

        Args:
            eigenvector: The eigenvector from which the solution states are extracted.
            qubo: The QUBO to evaluate at the bitstring.
            min_probability: Only consider states where the amplitude exceeds this threshold.

        Returns:
            For each computational basis state contained in the eigenvector, return the basis
            state as bitstring along with the QUBO evaluated at that bitstring and the
            probability of sampling this bitstring from the eigenvector.

        Examples:
            >>> op = MatrixOp(numpy.array([[1, 1], [1, -1]]) / numpy.sqrt(2))
            >>> eigenvectors = {'0': 12, '1': 1}
            >>> print(eigenvector_to_solutions(eigenvectors, op))
            [('0', 0.7071067811865475, 0.9230769230769231),
            ('1', -0.7071067811865475, 0.07692307692307693)]

            >>> op = MatrixOp(numpy.array([[1, 1], [1, -1]]) / numpy.sqrt(2))
            >>> eigenvectors = numpy.array([1, 1] / numpy.sqrt(2), dtype=complex)
            >>> print(eigenvector_to_solutions(eigenvectors, op))
            [('0', 0.7071067811865475, 0.4999999999999999),
            ('1', -0.7071067811865475, 0.4999999999999999)]

        Raises:
            TypeError: If the type of eigenvector is not supported.
        """
        if isinstance(eigenvector, DictStateFn):
            eigenvector = eigenvector.primitive
        elif isinstance(eigenvector, StateFn):
            eigenvector = eigenvector.to_matrix()

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
        if isinstance(eigenvector, dict):
            # When eigenvector is a dict, square the values since the values are normalized.
            # See https://github.com/Qiskit/qiskit-terra/pull/5496 for more details.
            probabilities = {bitstr: val ** 2 for (bitstr, val) in eigenvector.items()}
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
                    bitstr = "{:b}".format(i).rjust(num_qubits, "0")
                    solutions.append(generate_solution(bitstr, qubo, sampling_probability))

        else:
            raise TypeError("Unsupported format of eigenvector. Provide a dict or numpy.ndarray.")

        return solutions
