# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Implementation of the Goemans-Williamson algorithm as an optimizer.
Requires CVXPY to run.
"""
import logging
from typing import Optional, List, Tuple, Union, cast

import numpy as np
from qiskit.exceptions import MissingOptionalLibraryError

from .optimization_algorithm import (
    OptimizationResult,
    OptimizationResultStatus,
    OptimizationAlgorithm,
    SolutionSample,
)
from ..converters.flip_problem_sense import MinimizeToMaximize
from ..problems.quadratic_program import QuadraticProgram
from ..problems.variable import Variable

try:
    import cvxpy as cvx
    from cvxpy import DCPError, DGPError, SolverError

    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False


logger = logging.getLogger(__name__)


class GoemansWilliamsonOptimizationResult(OptimizationResult):
    """
    Contains results of the Goemans-Williamson algorithm. The properties ``x`` and ``fval`` contain
    values of just one solution. Explore ``samples`` for all possible solutions.
    """

    def __init__(
        self,
        x: Optional[Union[List[float], np.ndarray]],
        fval: float,
        variables: List[Variable],
        status: OptimizationResultStatus,
        samples: Optional[List[SolutionSample]],
        sdp_solution: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            x: the optimal value found in the optimization.
            fval: the optimal function value.
            variables: the list of variables of the optimization problem.
            status: the termination status of the optimization algorithm.
            samples: the solution samples.
            sdp_solution: an SDP solution of the problem.
        """
        super().__init__(x, fval, variables, status, samples=samples)
        self._sdp_solution = sdp_solution

    @property
    def sdp_solution(self) -> Optional[np.ndarray]:
        """
        Returns:
            Returns an SDP solution of the problem.
        """
        return self._sdp_solution


class GoemansWilliamsonOptimizer(OptimizationAlgorithm):
    """
    Goemans-Williamson algorithm to approximate the max-cut of a problem.
    The quadratic program for max-cut is given by:

    max sum_{i,j<i} w[i,j]*x[i]*(1-x[j])

    Therefore the quadratic term encodes the negative of the adjacency matrix of
    the graph.
    """

    def __init__(
        self,
        num_cuts: int,
        sort_cuts: bool = True,
        unique_cuts: bool = True,
        seed: int = 0,
    ):
        """
        Args:
            num_cuts: Number of cuts to generate.
            sort_cuts: True if sort cuts by their values.
            unique_cuts: The solve method returns only unique cuts, thus there may be less cuts
                than ``num_cuts``.
            seed: A seed value for the random number generator.

        Raises:
            MissingOptionalLibraryError: CVXPY is not installed.
        """
        if not _HAS_CVXPY:
            raise MissingOptionalLibraryError(
                libname="CVXPY",
                name="GoemansWilliamsonOptimizer",
                pip_install="pip install 'qiskit-optimization[cvxpy]'",
            )
        super().__init__()

        self._num_cuts = num_cuts
        self._sort_cuts = sort_cuts
        self._unique_cuts = unique_cuts
        np.random.seed(seed)

    def get_compatibility_msg(self, problem: QuadraticProgram) -> str:
        """Checks whether a given problem can be solved with the optimizer implementing this method.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            Returns the incompatibility message. If the message is empty no issues were found.
        """
        message = ""
        if problem.get_num_binary_vars() != problem.get_num_vars():
            message = (
                f"Only binary variables are supported, while the total number of variables "
                f"{problem.get_num_vars()} and there are {problem.get_num_binary_vars()} "
                f"binary variables across them"
            )
        return message

    def solve(self, problem: QuadraticProgram) -> OptimizationResult:
        """
        Returns a list of cuts generated according to the Goemans-Williamson algorithm.

        Args:
            problem: The quadratic problem that encodes the max-cut problem.

        Returns:
            cuts: A list of generated cuts.
        """
        self._verify_compatibility(problem)

        min2max = MinimizeToMaximize()
        problem = min2max.convert(problem)

        adj_matrix = self._extract_adjacency_matrix(problem)

        try:
            chi = self._solve_max_cut_sdp(adj_matrix)
        except (DCPError, DGPError, SolverError):
            logger.error("Can't solve SDP problem")
            return GoemansWilliamsonOptimizationResult(
                x=[],
                fval=0,
                variables=problem.variables,
                status=OptimizationResultStatus.FAILURE,
                samples=[],
            )

        cuts = self._generate_random_cuts(chi, len(adj_matrix))

        numeric_solutions = [
            (cuts[i, :], self.max_cut_value(cuts[i, :], adj_matrix)) for i in range(self._num_cuts)
        ]

        if self._sort_cuts:
            numeric_solutions.sort(key=lambda x: -x[1])

        if self._unique_cuts:
            numeric_solutions = self._get_unique_cuts(numeric_solutions)

        numeric_solutions = numeric_solutions[: self._num_cuts]
        samples = [
            SolutionSample(
                x=solution[0],
                fval=solution[1],
                probability=1.0 / len(numeric_solutions),
                status=OptimizationResultStatus.SUCCESS,
            )
            for solution in numeric_solutions
        ]

        return cast(
            GoemansWilliamsonOptimizationResult,
            self._interpret(
                x=samples[0].x,
                problem=problem,
                converters=[min2max],
                result_class=GoemansWilliamsonOptimizationResult,
                samples=samples,
            ),
        )

    def _get_unique_cuts(
        self, solutions: List[Tuple[np.ndarray, float]]
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Returns:
            Unique Goemans-Williamson cuts.
        """

        # Remove symmetry in the cuts to chose the unique ones.
        # Cuts 010 and 101 are symmetric(same cut), so we convert all cuts
        # starting from 1 to start from 0. In the next loop repetitive cuts will be removed.
        for idx, cut in enumerate(solutions):
            if cut[0][0] == 1:
                solutions[idx] = (
                    np.array([0 if _ == 1 else 1 for _ in cut[0]]),
                    cut[1],
                )

        seen_cuts = set()
        unique_cuts = []
        for cut in solutions:
            cut_str = "".join([str(_) for _ in cut[0]])
            if cut_str in seen_cuts:
                continue

            seen_cuts.add(cut_str)
            unique_cuts.append(cut)

        return unique_cuts

    @staticmethod
    def _extract_adjacency_matrix(problem: QuadraticProgram) -> np.ndarray:
        """
        Extracts the adjacency matrix from the given quadratic program.

        Args:
            problem: A QuadraticProgram describing the max-cut optimization problem.

        Returns:
            adjacency matrix of the graph.
        """
        adj_matrix = -problem.objective.quadratic.coefficients.toarray()
        adj_matrix = (adj_matrix + adj_matrix.T) / 2

        return adj_matrix

    def _solve_max_cut_sdp(self, adj_matrix: np.ndarray) -> np.ndarray:
        """
        Calculates the maximum weight cut by generating |V| vectors with a vector program,
        then generating a random plane that cuts the vertices. This is the Goemans-Williamson
        algorithm that gives a .878-approximation.

        Returns:
            chi: a list of length |V| where the i-th element is +1 or -1, representing which
                set the it-h vertex is in. Returns None if an error occurs.
        """
        num_vertices = len(adj_matrix)
        constraints, expr = [], 0

        # variables
        x = cvx.Variable((num_vertices, num_vertices), PSD=True)

        # constraints
        for i in range(num_vertices):
            constraints.append(x[i, i] == 1)

        # objective function
        expr = cvx.sum(cvx.multiply(adj_matrix, (np.ones((num_vertices, num_vertices)) - x)))

        # solve
        problem = cvx.Problem(cvx.Maximize(expr), constraints)
        problem.solve()

        return x.value

    def _generate_random_cuts(self, chi: np.ndarray, num_vertices: int) -> np.ndarray:
        """
        Random hyperplane partitions vertices.

        Args:
            chi: a list of length |V| where the i-th element is +1 or -1, representing
                which set the i-th vertex is in.
            num_vertices: the number of vertices in the graph

        Returns:
            An array of random cuts.
        """
        eigenvalues = np.linalg.eigh(chi)[0]
        if min(eigenvalues) < 0:
            chi = chi + (1.001 * abs(min(eigenvalues)) * np.identity(num_vertices))
        elif min(eigenvalues) == 0:
            chi = chi + 0.00001 * np.identity(num_vertices)
        x = np.linalg.cholesky(chi).T

        r = np.random.normal(size=(self._num_cuts, num_vertices))

        return (np.dot(r, x) > 0) + 0

    @staticmethod
    def max_cut_value(x: np.ndarray, adj_matrix: np.ndarray):
        """Compute the value of a cut from an adjacency matrix and a list of binary values.

        Args:
            x: a list of binary value in numpy array.
            adj_matrix: adjacency matrix.

        Returns:
            float: value of the cut.
        """
        cut_matrix = np.outer(x, (1 - x))
        return np.sum(adj_matrix * cut_matrix)
