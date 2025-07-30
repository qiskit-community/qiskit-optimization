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

"""The Gurobi optimizer wrapped to be used within Qiskit optimization module."""

import qiskit_optimization.optionals as _optionals
from ..exceptions import QiskitOptimizationError
from ..problems.quadratic_program import QuadraticProgram
from ..translators.gurobipy import to_gurobipy
from .optimization_algorithm import OptimizationAlgorithm, OptimizationResult


@_optionals.HAS_GUROBIPY.require_in_instance
class GurobiOptimizer(OptimizationAlgorithm):
    """The Gurobi optimizer wrapped as a Qiskit :class:`OptimizationAlgorithm`.

    This class provides a wrapper for ``gurobipy``
    to be used within the optimization module.

    Examples:
        >>> from qiskit_optimization.problems import QuadraticProgram
        >>> from qiskit_optimization.algorithms import GurobiOptimizer
        >>> problem = QuadraticProgram()
        >>> # specify problem here, if gurobi is installed
        >>> optimizer = GurobiOptimizer() if GurobiOptimizer.is_gurobi_installed() else None
        >>> # Suppress gurobipy print info to stdout
        >>> import sys
        >>> class DevNull:
        ...     def noop(*args, **kwargs): pass
        ...     close = write = flush = writelines = noop
        >>> sys.stdout = DevNull()
        >>> result = optimizer.solve(problem)
    """

    def __init__(self, disp: bool = False) -> None:
        """Initializes the GurobiOptimizer.

        Args:
            disp: Whether to print Gurobi output or not.
        """
        self._disp = disp

    @staticmethod
    def is_gurobi_installed():
        """Returns True if gurobi is installed"""
        return _optionals.HAS_GUROBIPY

    @property
    def disp(self) -> bool:
        """Returns the display setting.

        Returns:
            Whether to print Gurobi information or not.
        """
        return self._disp

    @disp.setter
    def disp(self, disp: bool):
        """Set the display setting.
        Args:
            disp: The display setting.
        """
        self._disp = disp

    # pylint:disable=unused-argument
    def get_compatibility_msg(self, problem: QuadraticProgram) -> str:
        """Checks whether a given problem can be solved with this optimizer.

        Returns ``''`` since Gurobi accepts all problems that can be modeled using the
        ``QuadraticProgram``. Gurobi will also solve non-convex problems.

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
        # pylint: disable=import-error
        import gurobipy as gp

        # convert to Gurobi problem
        model = to_gurobipy(problem)

        # set display setting
        if not self.disp:
            model.Params.OutputFlag = 0

        # enable non-convex
        model.Params.NonConvex = 2

        # solve problem
        try:
            model.optimize()
        except gp.GurobiError as ex:
            raise QiskitOptimizationError(str(ex)) from ex

        # create results
        result = OptimizationResult(
            x=model.X,
            fval=model.ObjVal,
            variables=problem.variables,
            status=self._get_feasibility_status(problem, model.X),
            raw_results=model,
        )

        # return solution
        return result
