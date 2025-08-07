# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test warm start QAOA optimizer with the primitive-based minimum eigensolver. """

import unittest
from test import QiskitOptimizationTestCase

import numpy as np
from ddt import data, ddt
from docplex.mp.model import Model
from qiskit import generate_preset_pass_manager
from qiskit.utils.optionals import HAS_AER
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler, SamplerV2

import qiskit_optimization.optionals as _optionals
from qiskit_optimization.algorithms import SlsqpOptimizer
from qiskit_optimization.algorithms.goemans_williamson_optimizer import GoemansWilliamsonOptimizer
from qiskit_optimization.algorithms.warm_start_qaoa_optimizer import (
    MeanAggregator,
    WarmStartQAOAOptimizer,
)
from qiskit_optimization.applications.max_cut import Maxcut
from qiskit_optimization.minimum_eigensolvers import QAOA
from qiskit_optimization.optimizers import COBYLA
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.utils import algorithm_globals


@ddt
class TestWarmStartQAOAOptimizer(QiskitOptimizationTestCase):
    """Tests for the warm start QAOA optimizer."""

    @unittest.skipUnless(HAS_AER, "qiskit-aer is required to run this test")
    def setUp(self):
        super().setUp()
        self.seed = 17
        algorithm_globals.random_seed = self.seed
        self.sampler = {
            "v1": Sampler(run_options={"seed_simulator": self.seed}),
            "v2": SamplerV2(seed=18),
        }
        self.passmanager = generate_preset_pass_manager(
            optimization_level=1, target=AerSimulator().target, seed_transpiler=self.seed
        )

    @unittest.skipIf(not _optionals.HAS_CVXPY, "CVXPY not available.")
    @data("v1", "v2")
    def test_max_cut(self, version):
        """Basic test on the max cut problem."""
        graph = np.array(
            [
                [0.0, 1.0, 2.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [2.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )

        presolver = GoemansWilliamsonOptimizer(num_cuts=10, seed=self.seed)
        problem = Maxcut(graph).to_quadratic_program()

        qaoa = QAOA(
            sampler=self.sampler[version], optimizer=COBYLA(), reps=1, passmanager=self.passmanager
        )
        aggregator = MeanAggregator()
        optimizer = WarmStartQAOAOptimizer(
            pre_solver=presolver,
            relax_for_pre_solver=False,
            qaoa=qaoa,
            epsilon=0.25,
            num_initial_solutions=10,
            aggregator=aggregator,
        )
        result_warm = optimizer.solve(problem)

        self.assertIsNotNone(result_warm)
        self.assertIsNotNone(result_warm.x)
        np.testing.assert_allclose(result_warm.x, [0, 0, 1, 0])
        self.assertIsNotNone(result_warm.fval)
        np.testing.assert_allclose(result_warm.fval, 4)

    @data("v1", "v2")
    def test_constrained_binary(self, version):
        """Constrained binary optimization problem."""
        model = Model()
        v = model.binary_var(name="v")
        w = model.binary_var(name="w")
        # pylint:disable=invalid-name
        t = model.binary_var(name="t")

        model.minimize(v + w + t)
        model.add_constraint(2 * v + 10 * w + t <= 3, "cons1")
        model.add_constraint(v + w + t >= 2, "cons2")

        problem = from_docplex_mp(model)

        qaoa = QAOA(
            sampler=self.sampler[version], optimizer=COBYLA(), reps=1, passmanager=self.passmanager
        )
        aggregator = MeanAggregator()
        optimizer = WarmStartQAOAOptimizer(
            pre_solver=SlsqpOptimizer(),
            relax_for_pre_solver=True,
            qaoa=qaoa,
            epsilon=0.25,
            aggregator=aggregator,
        )
        result_warm = optimizer.solve(problem)

        self.assertIsNotNone(result_warm)
        self.assertIsNotNone(result_warm.x)
        np.testing.assert_allclose(result_warm.x, [1, 0, 1])
        self.assertIsNotNone(result_warm.fval)
        np.testing.assert_allclose(result_warm.fval, 2)

    @data("v1", "v2")
    def test_simple_qubo(self, version):
        """Test on a simple QUBO problem."""
        model = Model()
        # pylint:disable=invalid-name
        u = model.binary_var(name="u")
        v = model.binary_var(name="v")

        model.minimize((u - v + 2) ** 2)
        problem = from_docplex_mp(model)

        qaoa = QAOA(
            sampler=self.sampler[version], optimizer=COBYLA(), reps=1, passmanager=self.passmanager
        )
        optimizer = WarmStartQAOAOptimizer(
            pre_solver=SlsqpOptimizer(),
            relax_for_pre_solver=True,
            qaoa=qaoa,
            epsilon=0.25,
        )
        result_warm = optimizer.solve(problem)

        self.assertIsNotNone(result_warm)
        self.assertIsNotNone(result_warm.x)
        np.testing.assert_allclose(result_warm.x, [0, 1])
        self.assertIsNotNone(result_warm.fval)
        np.testing.assert_allclose(result_warm.fval, 1)


if __name__ == "__main__":
    unittest.main()
