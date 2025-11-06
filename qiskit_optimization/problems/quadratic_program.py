# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quadratic Program."""
from __future__ import annotations

import logging
from enum import Enum

from qiskit_addon_opt_mapper import OptimizationProblem
from qiskit.utils import deprecate_func

import qiskit_optimization.optionals as _optionals

logger = logging.getLogger(__name__)


class QuadraticProgramStatus(Enum):
    """Status of QuadraticProgram"""

    VALID = 0
    INFEASIBLE = 1


class QuadraticProgram:
    """Quadratically Constrained Quadratic Program representation.

    This representation supports inequality and equality constraints,
    as well as continuous, binary, and integer variables.
    """

    Status = QuadraticProgramStatus

    def __init__(self, name: str = "") -> None:
        """
        Args:
            name: The name of the quadratic program.
        """
        self._optimizationproblem = OptimizationProblem(name=name)

    def __getattr__(self, item):
        return getattr(self._optimizationproblem, item)

    @_optionals.HAS_CPLEX.require_in_call
    @deprecate_func(since="0.7.0", additional_msg="Use from_docplex_mp or from_gurobipy instead.")
    def read_from_lp_file(self, filename: str) -> None:
        """Loads the quadratic program from a LP file.

        Args:
            filename: The filename of the file to be loaded.

        Raises:
            FileNotFoundError: If the file does not exist.

        Note:
            This method requires CPLEX to be installed and present in ``PYTHONPATH``.
        """
        return self._optimizationproblem.read_from_lp_file(filename=filename)
