# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Semi-deterministic rounding module"""
from __future__ import annotations

import numpy as np

from qiskit_optimization.algorithms import OptimizationResultStatus, SolutionSample
from qiskit_optimization.exceptions import QiskitOptimizationError

from .rounding_common import RoundingContext, RoundingResult, RoundingScheme


class SemideterministicRounding(RoundingScheme):
    """Semi-deterministic rounding scheme

    This is referred to as "Pauli rounding" in
    https://arxiv.org/abs/2111.03167.
    """

    def __init__(self, *, atol: float = 1e-8, seed: int | None = None):
        """
        Args:
            seed: Seed for random number generator, which is used to resolve
                expectation values near zero to either +1 or -1.
            atol: Absolute tolerance for determining whether an expectation value is zero.
        """
        super().__init__()
        self._rng = np.random.default_rng(seed)
        self._atol = atol

    def round(self, rounding_context: RoundingContext) -> RoundingResult:
        """Perform semi-deterministic rounding

        Args:
            rounding_context: Rounding context containing information about the problem and solution.

        Returns:
            Result containing the rounded solution.

        Raises:
            QiskitOptimizationError: If the expectation values are not available in the context.
        """
        if rounding_context.expectation_values is None:
            raise QiskitOptimizationError(
                "Semi-deterministic rounding requires the expectation values of the ",
                "``RoundingContext`` to be available, but they are not.",
            )

        rounded_vars = np.where(
            np.isclose(rounding_context.expectation_values, 0, atol=self._atol),
            self._rng.integers(2, size=len(rounding_context.expectation_values)),
            np.less_equal(rounding_context.expectation_values, 0).astype(int),
        )

        soln_samples = [
            SolutionSample(
                x=np.asarray(rounded_vars),
                fval=rounding_context.encoding.problem.objective.evaluate(rounded_vars),
                probability=1.0,
                status=(
                    OptimizationResultStatus.SUCCESS
                    if rounding_context.encoding.problem.is_feasible(rounded_vars)
                    else OptimizationResultStatus.INFEASIBLE
                ),
            )
        ]

        result = RoundingResult(
            expectation_values=rounding_context.expectation_values, samples=soln_samples
        )
        return result
