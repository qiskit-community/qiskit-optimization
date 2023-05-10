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

"""Tests for MagicRounding"""
import unittest
from test.optimization_test_case import QiskitOptimizationTestCase

import numpy as np
import ddt

from qiskit_optimization.algorithms.qrao import (
    QuantumRandomAccessEncoding,
    SemideterministicRounding,
    SemideterministicRoundingResult,
    RoundingContext,
    MagicRounding,
    MagicRoundingResult,
)
from qiskit_optimization.algorithms.qrao.rounding_common import RoundingSolutionSample

from qiskit_optimization.problems import QuadraticProgram

from qiskit.primitives import Sampler


import numpy as np
from qiskit.algorithms.minimum_eigensolvers import (
    NumPyMinimumEigensolver,
    NumPyMinimumEigensolverResult,
    VQE,
    VQEResult,
)
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import QAOAAnsatz, RealAmplitudes
from qiskit.primitives import Estimator
from qiskit.utils import algorithm_globals

from qiskit_optimization.algorithms.optimization_algorithm import OptimizationResultStatus
from qiskit_optimization.algorithms.qrao import (
    QuantumRandomAccessEncoding,
    QuantumRandomAccessOptimizationResult,
    QuantumRandomAccessOptimizer,
    RoundingContext,
    SemideterministicRoundingResult,
)
from qiskit_optimization.algorithms.qrao.rounding_common import RoundingSolutionSample
from qiskit_optimization.problems import QuadraticProgram


class TestMagicRounding(QiskitOptimizationTestCase):
    """MagicRounding tests."""

    def setUp(self):
        """Set up for all tests."""
        super().setUp()
        self.problem = QuadraticProgram()
        self.problem.binary_var("x")
        self.problem.binary_var("y")
        self.problem.binary_var("z")
        self.problem.minimize(linear={"x": 1, "y": 2, "z": 3})
        self.sampler = Sampler(options={"shots": 10000, "seed": 42})

    def test_magic_rounding_constructor(self):
        """Test constructor"""
        # test default
        magic_rounding = MagicRounding(self.sampler)
        self.assertEqual(magic_rounding.sampler, self.sampler)
        self.assertEqual(magic_rounding.basis_sampling, "uniform")
        # test weighted basis sampling
        magic_rounding = MagicRounding(self.sampler, basis_sampling="weighted")
        self.assertEqual(magic_rounding.sampler, self.sampler)
        self.assertEqual(magic_rounding.basis_sampling, "weighted")
        # test uniform basis sampling
        magic_rounding = MagicRounding(self.sampler, basis_sampling="uniform")
        self.assertEqual(magic_rounding.sampler, self.sampler)
        self.assertEqual(magic_rounding.basis_sampling, "uniform")
        # test invalid basis sampling
        with self.assertRaises(ValueError):
            MagicRounding(self.sampler, basis_sampling="invalid")

    def test_magic_rounding_round_uniform(self):
        """Test round method with uniform basis sampling"""
        encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=3)
        encoding.encode(self.problem)
        np_solver = NumPyMinimumEigensolver()
        qrao = QuantumRandomAccessOptimizer(min_eigen_solver=np_solver)
        _, rounding_context = qrao.solve_relaxed(encoding=encoding)
        magic_rounding = MagicRounding(self.sampler, seed=42)
        rounding_result = magic_rounding.round(rounding_context)
        self.assertIsInstance(rounding_result, MagicRoundingResult)
        np.testing.assert_allclose(rounding_result.bases, [[0], [1], [2], [3]])
        np.testing.assert_allclose(rounding_result.basis_shots, [2536, 2486, 2477, 2501])
        expected_basis_counts = [
            {"0": 2436.0, "1": 100.0},
            {"0": 461.0, "1": 2025.0},
            {"0": 831.0, "1": 1646.0},
            {"0": 1259.0, "1": 1242.0},
        ]
        for i, basis_counts in enumerate(rounding_result.basis_counts):
            self.assertEqual(basis_counts, expected_basis_counts[i])
        samples = rounding_result.samples
        samples.sort(key=lambda sample: np.array2string(sample.x))
        expected_samples = [
            RoundingSolutionSample(x=np.array([0, 0, 0]), probability=0.2436),
            RoundingSolutionSample(x=np.array([0, 0, 1]), probability=0.1242),
            RoundingSolutionSample(x=np.array([0, 1, 0]), probability=0.1646),
            RoundingSolutionSample(x=np.array([0, 1, 1]), probability=0.0461),
            RoundingSolutionSample(x=np.array([1, 0, 0]), probability=0.2025),
            RoundingSolutionSample(x=np.array([1, 0, 1]), probability=0.0831),
            RoundingSolutionSample(x=np.array([1, 1, 0]), probability=0.1259),
            RoundingSolutionSample(x=np.array([1, 1, 1]), probability=0.01),
        ]
        for i, sample in enumerate(samples):
            np.testing.assert_allclose(sample.x, expected_samples[i].x)
            self.assertAlmostEqual(sample.probability, expected_samples[i].probability)
        np.testing.assert_allclose(
            rounding_result.expectation_values,
            [0.2672612419124245, 0.5345224838248487, 0.8017837257372733],
        )

    def test_magic_rounding_round_weighted(self):
        """Test round method with weighted basis sampling"""
        encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=3)
        encoding.encode(self.problem)
        np_solver = NumPyMinimumEigensolver()
        qrao = QuantumRandomAccessOptimizer(min_eigen_solver=np_solver)
        _, rounding_context = qrao.solve_relaxed(encoding=encoding)
        magic_rounding = MagicRounding(self.sampler, basis_sampling="weighted", seed=42)
        rounding_result = magic_rounding.round(rounding_context)
        self.assertIsInstance(rounding_result, MagicRoundingResult)
        np.testing.assert_allclose(rounding_result.bases, [[0], [1], [2], [3]])
        np.testing.assert_allclose(rounding_result.basis_shots, [4542, 2703, 1559, 1196])
        expected_basis_counts = [
            {"0": 4393.0, "1": 149.0},
            {"0": 501.0, "1": 2202.0},
            {"0": 523.0, "1": 1036.0},
            {"0": 583.0, "1": 613.0},
        ]
        for i, basis_counts in enumerate(rounding_result.basis_counts):
            self.assertEqual(basis_counts, expected_basis_counts[i])
        samples = rounding_result.samples
        samples.sort(key=lambda sample: np.array2string(sample.x))
        expected_samples = [
            RoundingSolutionSample(x=np.array([0, 0, 0]), probability=0.4393),
            RoundingSolutionSample(x=np.array([0, 0, 1]), probability=0.0613),
            RoundingSolutionSample(x=np.array([0, 1, 0]), probability=0.1036),
            RoundingSolutionSample(x=np.array([0, 1, 1]), probability=0.0501),
            RoundingSolutionSample(x=np.array([1, 0, 0]), probability=0.2202),
            RoundingSolutionSample(x=np.array([1, 0, 1]), probability=0.0523),
            RoundingSolutionSample(x=np.array([1, 1, 0]), probability=0.0583),
            RoundingSolutionSample(x=np.array([1, 1, 1]), probability=0.0149),
        ]
        for i, sample in enumerate(samples):
            np.testing.assert_allclose(sample.x, expected_samples[i].x)
            self.assertAlmostEqual(sample.probability, expected_samples[i].probability)
        np.testing.assert_allclose(
            rounding_result.expectation_values,
            [0.2672612419124245, 0.5345224838248487, 0.8017837257372733],
        )


if __name__ == "__main__":
    unittest.main()
