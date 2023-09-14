# This code is part of a Qiskit project.
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

"""The SciPy MILP optimizer wrapped to be used within Qiskit optimization module."""

from warnings import warn

import numpy as np

from qiskit_optimization import INFINITY
from qiskit_optimization.algorithms.optimization_algorithm import (
    OptimizationAlgorithm,
    OptimizationResult,
    OptimizationResultStatus,
)
from qiskit_optimization.problems.quadratic_program import (
    ConstraintSense,
    QuadraticProgram,
    VarType,
)


def _conv_inf(val):
    # Note: qiskit-optimization treats INFINITY as infinity
    # while scipy treats np.inf as infinity
    if val <= -INFINITY:
        return -np.inf
    elif val >= INFINITY:
        return np.inf
    else:
        return val


class ScipyMilpOptimizer(OptimizationAlgorithm):
    """The MILP optimizer from Scipy wrapped as a Qiskit :class:`OptimizationAlgorithm`.

    This class provides a wrapper for ``scipy.milp``
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)
    to be used within the optimization module.
    """

    def __init__(self, disp: bool = False) -> None:
        """Initializes the ScipyMILPOptimizer.

        Args:
            disp: Whether to print MILP output or not.
        """
        self._disp = disp

    @property
    def disp(self) -> bool:
        """Returns the display setting.

        Returns:
            Whether to print scipy.milp information or not.
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

        Checks if the problem has only linear objective function and linear constraints.
        The ``scipy.milp`` supports only linear objective function and linear constraints.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            An empty string (if compatible) or a string describing the incompatibility.
        """
        msg = []
        if problem.objective.quadratic.to_dict():
            msg.append("scipy.milp supports only linear objective function")
        if problem.quadratic_constraints:
            msg.append("scipy.milp supports only linear constraints")
        return "; ".join(msg)

    def _generate_problem(self, problem: QuadraticProgram):
        from scipy.optimize import Bounds, LinearConstraint
        from scipy.sparse import lil_array  # pylint: disable=no-name-in-module

        sense = problem.objective.sense.value
        objective = problem.objective.linear.to_array() * sense

        integrality = []
        for variable in problem.variables:
            if variable.vartype == VarType.CONTINUOUS:
                integrality.append(0)
            else:
                integrality.append(1)

        lower_bounds = [_conv_inf(variable.lowerbound) for variable in problem.variables]
        upper_bounds = [_conv_inf(variable.upperbound) for variable in problem.variables]
        bounds = Bounds(lb=lower_bounds, ub=upper_bounds)

        lower_bounds = []
        upper_bounds = []
        mat = lil_array((problem.get_num_linear_constraints(), problem.get_num_vars()))
        for i, constraint in enumerate(problem.linear_constraints):
            for variable_id, val in constraint.linear.to_dict().items():
                mat[i, variable_id] = val

            rhs_val = _conv_inf(constraint.rhs)
            if constraint.sense == ConstraintSense.GE:
                lower_bounds.append(rhs_val)
                upper_bounds.append(np.inf)
            elif constraint.sense == ConstraintSense.LE:
                lower_bounds.append(-np.inf)
                upper_bounds.append(rhs_val)
            else:
                # ConstraintSense.EQ
                lower_bounds.append(rhs_val)
                upper_bounds.append(rhs_val)

        constraints = LinearConstraint(mat, lower_bounds, upper_bounds)

        return objective, integrality, bounds, constraints

    def solve(self, problem: QuadraticProgram) -> OptimizationResult:
        """Tries to solve the given problem using the optimizer.

        Runs the optimizer to try to solve the optimization problem. If problem is not convex,
        this optimizer may raise an exception due to incompatibility, depending on the settings.

        Args:
            problem: The problem to be solved.

        Returns:
            The result of the optimizer applied to the problem.

        Raises:
            QiskitOptimizationError: If the problem is incompatible with the optimizer.
        """
        from scipy.optimize import milp  # pylint: disable=no-name-in-module

        self._verify_compatibility(problem)

        objective, integrality, bounds, constraints = self._generate_problem(problem)
        raw_result = milp(
            c=objective,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={"disp": self._disp},
        )

        if raw_result.x is None:
            warn("scipy.milp cannot solve the model. See `raw_results` for details.")
            x = [0.0] * problem.get_num_vars()
            status = OptimizationResultStatus.FAILURE
        else:
            x = []
            for i, ele in enumerate(raw_result.x):
                if integrality[i]:
                    x.append(round(ele))
                else:
                    x.append(ele)
            status = self._get_feasibility_status(problem, x)

        return OptimizationResult(
            x=x,
            fval=problem.objective.evaluate(x),
            variables=problem.variables,
            status=status,
            raw_results=raw_result,
        )
