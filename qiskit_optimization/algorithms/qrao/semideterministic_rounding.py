# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Semideterministic rounding module"""

from typing import Optional

import numpy as np

from .rounding_common import (
    RoundingSolutionSample,
    RoundingScheme,
    RoundingContext,
    RoundingResult,
)


# pylint: disable=too-few-public-methods


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
        """Perform semideterministic rounding"""

        # # trace_values = ctx.expectation_values

        # if trace_values is None:
        #     raise NotImplementedError(
        #         "Semideterministic rounding requires that trace_values be available."
        #     )

        # if len(trace_values) != len(ctx.var2op):
        #     raise ValueError(
        #         f"trace_values has length {len(trace_values)}, "
        #         "but there are {len(ctx.var2op)} decision variables."
        #     )

        def sign(val) -> int:
            return 0 if (val > 0) else 1

        rounded_vars = [
            sign(e) if not np.isclose(0, e) else self.rng.randint(2) for e in ctx.expectation_values
        ]

        soln_samples = [
            RoundingSolutionSample(
                x=np.asarray(rounded_vars),
                probability=1.0,
            )
        ]

        result = SemideterministicRoundingResult(
            samples=soln_samples, expectation_values=ctx.expectation_values
        )
        return result
