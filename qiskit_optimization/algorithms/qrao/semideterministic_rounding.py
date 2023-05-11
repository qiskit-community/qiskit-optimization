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

"""Semideterministic rounding module"""
from dataclasses import dataclass

from typing import Optional

import numpy as np

from qiskit_optimization.algorithms import OptimizationResultStatus, SolutionSample

from .rounding_common import (
    RoundingScheme,
    RoundingContext,
    RoundingResult,
)


@dataclass
class SemideterministicRoundingResult(RoundingResult):
    """Result of semideterministic rounding"""


class SemideterministicRounding(RoundingScheme):
    """Semideterministic rounding scheme

    This is referred to as "Pauli rounding" in
    https://arxiv.org/abs/2111.03167v2.
    """

    def __init__(self, *, seed: Optional[int] = None):
        """
        Args:
            seed: Seed for random number generator, which is used to resolve
                expectation values near zero to either +1 or -1.
        """
        super().__init__()
        self.rng = np.random.RandomState(seed)

    def round(self, ctx: RoundingContext) -> SemideterministicRoundingResult:
        """Perform semideterministic rounding

        Args:
            ctx: Rounding context containing information about the problem and solution.

        Returns:
            Result containing the rounded solution.

        Raises:
            NotImplementedError: If the expectation values are not available in the context.
        """

        def sign(val) -> int:
            return 0 if (val > 0) else 1

        if ctx.expectation_values is None:
            raise NotImplementedError(
                "Semideterministric rounding requires the expectation values of the ",
                "``RoundingContext`` to be available, but they are not.",
            )
        rounded_vars = np.array(
            [
                sign(e) if not np.isclose(0, e) else self.rng.randint(2)
                for e in ctx.expectation_values
            ]
        )

        soln_samples = [
            SolutionSample(
                x=np.asarray(rounded_vars),
                fval=ctx.encoding.problem.objective.evaluate(rounded_vars),
                probability=1.0,
                status=OptimizationResultStatus.SUCCESS
                if ctx.encoding.problem.is_feasible(rounded_vars)
                else OptimizationResultStatus.INFEASIBLE,
            )
        ]

        result = SemideterministicRoundingResult(
            expectation_values=ctx.expectation_values, samples=soln_samples
        )
        return result
