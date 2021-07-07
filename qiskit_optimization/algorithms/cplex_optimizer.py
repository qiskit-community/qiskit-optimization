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

"""The CPLEX optimizer wrapped to be used within Qiskit's optimization module."""

from typing import Any, Dict, Optional
from warnings import warn

from qiskit.exceptions import MissingOptionalLibraryError

from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from qiskit_optimization.translators import to_docplex_mp
from .optimization_algorithm import (
    OptimizationAlgorithm,
    OptimizationResult,
    OptimizationResultStatus,
)


try:
    from cplex import Cplex  # pylint: disable=unused-import

    _HAS_CPLEX = True
except ImportError:
    _HAS_CPLEX = False


class CplexOptimizer(OptimizationAlgorithm):
    """The CPLEX optimizer wrapped as an Qiskit :class:`OptimizationAlgorithm`.

    This class provides a wrapper for ``cplex.Cplex`` (https://pypi.org/project/cplex/)
    to be used within the optimization module.

    Examples:
        >>> from qiskit_optimization.problems import QuadraticProgram
        >>> from qiskit_optimization.algorithms import CplexOptimizer
        >>> problem = QuadraticProgram()
        >>> # specify problem here, if cplex is installed
        >>> optimizer = CplexOptimizer() if CplexOptimizer.is_cplex_installed() else None
        >>> if optimizer: result = optimizer.solve(problem)
    """

    def __init__(
        self, disp: bool = False, cplex_parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initializes the CplexOptimizer.

        Args:
            disp: Whether to print CPLEX output or not.
            cplex_parameters: The parameters for CPLEX.
                See https://www.ibm.com/docs/en/icos/20.1.0?topic=cplex-parameters for details.

        Raises:
            MissingOptionalLibraryError: CPLEX is not installed.
        """
        if not _HAS_CPLEX:
            raise MissingOptionalLibraryError(
                libname="CPLEX",
                name="CplexOptimizer",
                pip_install="pip install 'qiskit-optimization[cplex]'",
            )

        self._disp = disp
        self._cplex_parameters = cplex_parameters

    @staticmethod
    def is_cplex_installed():
        """Returns True if cplex is installed"""
        return _HAS_CPLEX

    @property
    def disp(self) -> bool:
        """Returns the display setting.

        Returns:
            Whether to print CPLEX information or not.
        """
        return self._disp

    @disp.setter
    def disp(self, disp: bool):
        """Set the display setting.
        Args:
            disp: The display setting.
        """
        self._disp = disp

    @property
    def cplex_parameters(self) -> Optional[Dict[str, Any]]:
        """Returns parameters for CPLEX"""
        return self._cplex_parameters

    @cplex_parameters.setter
    def cplex_parameters(self, parameters: Optional[Dict[str, Any]]):
        """Set parameters for CPLEX
        Args:
            parameters: The parameters for CPLEX
        """
        self._cplex_parameters = parameters

    # pylint:disable=unused-argument
    def get_compatibility_msg(self, problem: QuadraticProgram) -> str:
        """Checks whether a given problem can be solved with this optimizer.

        Returns ``''`` since CPLEX accepts all problems that can be modeled using the
        ``QuadraticProgram``. CPLEX may throw an exception in case the problem is determined
        to be non-convex.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            An empty string.
        """
        return ""

    def solve(self, problem: QuadraticProgram) -> OptimizationResult:
        """Tries to solves the given problem using the optimizer.

        Runs the optimizer to try to solve the optimization problem. If problem is not convex,
        this optimizer may raise an exception due to incompatibility, depending on the settings.

        Args:
            problem: The problem to be solved.

        Returns:
            The result of the optimizer applied to the problem.

        Raises:
            QiskitOptimizationError: If the problem is incompatible with the optimizer.
        """

        mod = to_docplex_mp(problem)
        sol = mod.solve(log_output=self._disp, cplex_parameters=self._cplex_parameters)
        if sol is None:
            # no solution is found
            warn("CPLEX cannot solve the model")
            x = [0.0] * mod.number_of_variables
            return OptimizationResult(
                x=x,
                fval=problem.objective.evaluate(x),
                variables=problem.variables,
                status=OptimizationResultStatus.FAILURE,
                raw_results=None,
            )
        else:
            # a solution is found
            x = sol.get_values(mod.iter_variables())
            return OptimizationResult(
                x=x,
                fval=sol.get_objective_value(),
                variables=problem.variables,
                status=self._get_feasibility_status(problem, x),
                raw_results=sol,
            )
