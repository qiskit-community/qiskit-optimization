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
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit.primitives import Sampler

from qiskit_optimization.algorithms.qrao import (
    MagicRounding,
    MagicRoundingResult,
    QuantumRandomAccessEncoding,
    QuantumRandomAccessOptimizer,
)
from qiskit_optimization.algorithms import OptimizationResultStatus, SolutionSample
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
        np.testing.assert_allclose(rounding_result.basis_shots, [2534, 2527, 2486, 2453])
        expected_basis_counts = [
            {"0": 2434.0, "1": 100.0},
            {"0": 469.0, "1": 2058.0},
            {"0": 833.0, "1": 1653.0},
            {"0": 1234.0, "1": 1219.0},
        ]
        for i, basis_counts in enumerate(rounding_result.basis_counts):
            self.assertEqual(basis_counts, expected_basis_counts[i])
        samples = rounding_result.samples
        samples.sort(key=lambda sample: np.array2string(sample.x))
        expected_samples = [
            make_solution_sample(x=np.array([0, 0, 0]), probability=0.2434, problem=self.problem),
            make_solution_sample(x=np.array([0, 0, 1]), probability=0.1219, problem=self.problem),
            make_solution_sample(x=np.array([0, 1, 0]), probability=0.1653, problem=self.problem),
            make_solution_sample(x=np.array([0, 1, 1]), probability=0.0469, problem=self.problem),
            make_solution_sample(x=np.array([1, 0, 0]), probability=0.2058, problem=self.problem),
            make_solution_sample(x=np.array([1, 0, 1]), probability=0.0833, problem=self.problem),
            make_solution_sample(x=np.array([1, 1, 0]), probability=0.1234, problem=self.problem),
            make_solution_sample(x=np.array([1, 1, 1]), probability=0.01, problem=self.problem),
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
        np.testing.assert_allclose(rounding_result.basis_shots, [4499, 2700, 1574, 1227])
        expected_basis_counts = [
            {"0": 4352.0, "1": 147.0},
            {"0": 500.0, "1": 2200.0},
            {"0": 528.0, "1": 1046.0},
            {"0": 597.0, "1": 630.0},
        ]
        for i, basis_counts in enumerate(rounding_result.basis_counts):
            self.assertEqual(basis_counts, expected_basis_counts[i])
        samples = rounding_result.samples
        samples.sort(key=lambda sample: np.array2string(sample.x))
        expected_samples = [
            make_solution_sample(x=np.array([0, 0, 0]), probability=0.4352, problem=self.problem),
            make_solution_sample(x=np.array([0, 0, 1]), probability=0.063, problem=self.problem),
            make_solution_sample(x=np.array([0, 1, 0]), probability=0.1046, problem=self.problem),
            make_solution_sample(x=np.array([0, 1, 1]), probability=0.05, problem=self.problem),
            make_solution_sample(x=np.array([1, 0, 0]), probability=0.22, problem=self.problem),
            make_solution_sample(x=np.array([1, 0, 1]), probability=0.0528, problem=self.problem),
            make_solution_sample(x=np.array([1, 1, 0]), probability=0.0597, problem=self.problem),
            make_solution_sample(x=np.array([1, 1, 1]), probability=0.0147, problem=self.problem),
        ]
        for i, sample in enumerate(samples):
            np.testing.assert_allclose(sample.x, expected_samples[i].x)
            self.assertAlmostEqual(sample.probability, expected_samples[i].probability)
        np.testing.assert_allclose(
            rounding_result.expectation_values,
            [0.2672612419124245, 0.5345224838248487, 0.8017837257372733],
        )


def make_solution_sample(
    x: np.ndarray, probability: float, problem: QuadraticProgram
) -> SolutionSample:
    """Make a solution sample."""
    return SolutionSample(
        x=x,
        fval=problem.objective.evaluate(x),
        probability=probability,
        status=OptimizationResultStatus.SUCCESS
        if problem.is_feasible(x)
        else OptimizationResultStatus.INFEASIBLE,
    )


if __name__ == "__main__":
    unittest.main()
