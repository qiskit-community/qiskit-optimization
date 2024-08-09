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

"""Tests for MagicRounding"""
import unittest
from test.optimization_test_case import QiskitOptimizationTestCase

import networkx as nx
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit_algorithms import NumPyMinimumEigensolver

from qiskit_optimization.algorithms import OptimizationResultStatus, SolutionSample
from qiskit_optimization.algorithms.qrao import (
    MagicRounding,
    QuantumRandomAccessEncoding,
    QuantumRandomAccessOptimizer,
    RoundingContext,
    RoundingResult,
)
from qiskit_optimization.applications import Maxcut
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

        # trivial maxcut problem (solution: {1}, {0, 2} with cut val 2)
        graph = nx.Graph()
        graph.add_nodes_from([0, 1, 2])
        graph.add_edges_from([(0, 1), (1, 2)])
        maxcut = Maxcut(graph)
        self.maxcut_problem = maxcut.to_quadratic_program()
        self.maxcut_optimal_value = 2.0
        self.maxcut_optimal_solution = [0, 1, 0]

    def test_magic_rounding_constructor(self):
        """Test constructor"""
        sampler = Sampler(options={"shots": 10000, "seed": 42})
        # test default
        magic_rounding = MagicRounding(sampler)
        self.assertEqual(magic_rounding.sampler, sampler)
        self.assertEqual(magic_rounding.basis_sampling, "uniform")
        # test weighted basis sampling
        magic_rounding = MagicRounding(sampler, basis_sampling="weighted")
        self.assertEqual(magic_rounding.sampler, sampler)
        self.assertEqual(magic_rounding.basis_sampling, "weighted")
        # test uniform basis sampling
        magic_rounding = MagicRounding(sampler, basis_sampling="uniform")
        self.assertEqual(magic_rounding.sampler, sampler)
        self.assertEqual(magic_rounding.basis_sampling, "uniform")
        # test invalid basis sampling
        with self.assertRaises(ValueError):
            MagicRounding(sampler, basis_sampling="invalid")

    def test_magic_rounding_round_uniform_1_1_qrac(self):
        """Test round method with uniform basis sampling for max_vars_per_qubit=1"""
        sampler = Sampler(options={"shots": 10000, "seed": 42})
        encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=1)
        encoding.encode(self.problem)
        np_solver = NumPyMinimumEigensolver()
        qrao = QuantumRandomAccessOptimizer(min_eigen_solver=np_solver)
        _, rounding_context = qrao.solve_relaxed(encoding=encoding)
        magic_rounding = MagicRounding(sampler, seed=42)
        rounding_result = magic_rounding.round(rounding_context)
        self.assertIsInstance(rounding_result, RoundingResult)
        np.testing.assert_allclose(rounding_result.bases, [[0, 0, 0]])
        np.testing.assert_allclose(rounding_result.basis_shots, [10000])
        expected_basis_counts = [{"000": 10000}]
        for i, basis_counts in enumerate(rounding_result.basis_counts):
            for key, value in basis_counts.items():
                self.assertAlmostEqual(value, expected_basis_counts[i][key], delta=50)
        samples = rounding_result.samples
        samples.sort(key=lambda sample: np.array2string(sample.x))
        expected_samples = [
            make_solution_sample(x=np.array([0, 0, 0]), probability=1, problem=self.problem),
        ]
        for i, sample in enumerate(samples):
            np.testing.assert_allclose(sample.x, expected_samples[i].x)
            self.assertAlmostEqual(sample.probability, expected_samples[i].probability, delta=0.05)
        np.testing.assert_allclose(
            rounding_result.expectation_values,
            [1, 1, 1],
        )

    def test_magic_rounding_round_weighted_1_1_qrac(self):
        """Test round method with uniform basis sampling for max_vars_per_qubit=1"""
        sampler = Sampler(options={"shots": 10000, "seed": 42})
        encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=1)
        encoding.encode(self.problem)
        np_solver = NumPyMinimumEigensolver()
        qrao = QuantumRandomAccessOptimizer(min_eigen_solver=np_solver)
        _, rounding_context = qrao.solve_relaxed(encoding=encoding)
        magic_rounding = MagicRounding(sampler, basis_sampling="weighted", seed=42)
        rounding_result = magic_rounding.round(rounding_context)
        self.assertIsInstance(rounding_result, RoundingResult)
        np.testing.assert_allclose(rounding_result.bases, [[0, 0, 0]])
        np.testing.assert_allclose(rounding_result.basis_shots, [10000])
        expected_basis_counts = [{"000": 10000}]
        for i, basis_counts in enumerate(rounding_result.basis_counts):
            for key, value in basis_counts.items():
                self.assertAlmostEqual(value, expected_basis_counts[i][key], delta=50)
        samples = rounding_result.samples
        samples.sort(key=lambda sample: np.array2string(sample.x))
        expected_samples = [
            make_solution_sample(x=np.array([0, 0, 0]), probability=1, problem=self.problem),
        ]
        for i, sample in enumerate(samples):
            np.testing.assert_allclose(sample.x, expected_samples[i].x)
            self.assertAlmostEqual(sample.probability, expected_samples[i].probability, delta=0.05)
        np.testing.assert_allclose(
            rounding_result.expectation_values,
            [1, 1, 1],
        )

    def test_magic_rounding_round_uniform_2_1_qrac(self):
        """Test round method with uniform basis sampling for max_vars_per_qubit=2"""
        sampler = Sampler(options={"shots": 10000, "seed": 42})
        encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=2)
        encoding.encode(self.problem)
        np_solver = NumPyMinimumEigensolver()
        qrao = QuantumRandomAccessOptimizer(min_eigen_solver=np_solver)
        _, rounding_context = qrao.solve_relaxed(encoding=encoding)
        magic_rounding = MagicRounding(sampler, seed=42)
        rounding_result = magic_rounding.round(rounding_context)
        self.assertIsInstance(rounding_result, RoundingResult)
        np.testing.assert_allclose(rounding_result.bases, [[0, 0], [0, 1], [1, 0], [1, 1]])
        np.testing.assert_allclose(rounding_result.basis_shots, [2575, 2482, 2440, 2503])
        expected_basis_counts = [
            {"00": 2154.0, "01": 367, "10": 44.0, "11": 10.0},
            {"00": 2076.0, "01": 357.0, "10": 45.0, "11": 4.0},
            {"00": 689.0, "01": 137.0, "10": 1401.0, "11": 213.0},
            {"00": 708.0, "01": 110.0, "10": 1446.0, "11": 239.0},
        ]
        for i, basis_counts in enumerate(rounding_result.basis_counts):
            for key, value in basis_counts.items():
                self.assertAlmostEqual(value, expected_basis_counts[i][key], delta=50)
        samples = rounding_result.samples
        samples.sort(key=lambda sample: np.array2string(sample.x))
        expected_samples = [
            make_solution_sample(x=np.array([0, 0, 0]), probability=0.423, problem=self.problem),
            make_solution_sample(x=np.array([0, 0, 1]), probability=0.0724, problem=self.problem),
            make_solution_sample(x=np.array([0, 1, 0]), probability=0.1397, problem=self.problem),
            make_solution_sample(x=np.array([0, 1, 1]), probability=0.0247, problem=self.problem),
            make_solution_sample(x=np.array([1, 0, 0]), probability=0.2847, problem=self.problem),
            make_solution_sample(x=np.array([1, 0, 1]), probability=0.0452, problem=self.problem),
            make_solution_sample(x=np.array([1, 1, 0]), probability=0.0089, problem=self.problem),
            make_solution_sample(x=np.array([1, 1, 1]), probability=0.0014, problem=self.problem),
        ]
        for i, sample in enumerate(samples):
            np.testing.assert_allclose(sample.x, expected_samples[i].x)
            self.assertAlmostEqual(sample.probability, expected_samples[i].probability, delta=0.05)
        np.testing.assert_allclose(
            rounding_result.expectation_values,
            [0.44721359549995743, 0.8944271909999162, 1],
        )

    def test_magic_rounding_round_weighted_2_1_qrac(self):
        """Test round method with weighted basis sampling for max_vars_per_qubit=2"""
        sampler = Sampler(options={"shots": 10000, "seed": 42})
        encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=2)
        encoding.encode(self.problem)
        np_solver = NumPyMinimumEigensolver()
        qrao = QuantumRandomAccessOptimizer(min_eigen_solver=np_solver)
        _, rounding_context = qrao.solve_relaxed(encoding=encoding)
        magic_rounding = MagicRounding(sampler, basis_sampling="weighted", seed=42)
        rounding_result = magic_rounding.round(rounding_context)
        self.assertIsInstance(rounding_result, RoundingResult)
        np.testing.assert_allclose(rounding_result.bases, [[0, 0], [1, 0]])
        np.testing.assert_allclose(rounding_result.basis_shots, [7058, 2942])
        expected_basis_counts = [
            {"00": 5858.0, "01": 1036.0, "10": 137.0, "11": 27.0},
            {"00": 832.0, "01": 131.0, "10": 1698.0, "11": 281.0},
        ]
        for i, basis_counts in enumerate(rounding_result.basis_counts):
            for key, value in basis_counts.items():
                self.assertAlmostEqual(value, expected_basis_counts[i][key], delta=50)
        samples = rounding_result.samples
        samples.sort(key=lambda sample: np.array2string(sample.x))
        expected_samples = [
            make_solution_sample(x=np.array([0, 0, 0]), probability=0.5858, problem=self.problem),
            make_solution_sample(x=np.array([0, 0, 1]), probability=0.1036, problem=self.problem),
            make_solution_sample(x=np.array([0, 1, 0]), probability=0.0832, problem=self.problem),
            make_solution_sample(x=np.array([0, 1, 1]), probability=0.0131, problem=self.problem),
            make_solution_sample(x=np.array([1, 0, 0]), probability=0.1698, problem=self.problem),
            make_solution_sample(x=np.array([1, 0, 1]), probability=0.0281, problem=self.problem),
            make_solution_sample(x=np.array([1, 1, 0]), probability=0.0137, problem=self.problem),
            make_solution_sample(x=np.array([1, 1, 1]), probability=0.0027, problem=self.problem),
        ]
        for i, sample in enumerate(samples):
            np.testing.assert_allclose(sample.x, expected_samples[i].x)
            self.assertAlmostEqual(sample.probability, expected_samples[i].probability, delta=0.05)
        np.testing.assert_allclose(
            rounding_result.expectation_values,
            [0.44721359549995743, 0.8944271909999162, 1],
        )

    def test_magic_rounding_round_uniform_3_1_qrac(self):
        """Test round method with uniform basis sampling for max_vars_per_qubit=3"""
        sampler = Sampler(options={"shots": 10000, "seed": 42})
        encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=3)
        encoding.encode(self.problem)
        np_solver = NumPyMinimumEigensolver()
        qrao = QuantumRandomAccessOptimizer(min_eigen_solver=np_solver)
        _, rounding_context = qrao.solve_relaxed(encoding=encoding)
        magic_rounding = MagicRounding(sampler, seed=42)
        rounding_result = magic_rounding.round(rounding_context)
        self.assertIsInstance(rounding_result, RoundingResult)
        np.testing.assert_allclose(rounding_result.bases, [[0], [1], [2], [3]])
        np.testing.assert_allclose(rounding_result.basis_shots, [2534, 2527, 2486, 2453])
        expected_basis_counts = [
            {"0": 2434.0, "1": 100.0},
            {"0": 469.0, "1": 2058.0},
            {"0": 833.0, "1": 1653.0},
            {"0": 1219.0, "1": 1234.0},
        ]
        for i, basis_counts in enumerate(rounding_result.basis_counts):
            for key, value in basis_counts.items():
                self.assertAlmostEqual(value, expected_basis_counts[i][key], delta=50)
        samples = rounding_result.samples
        samples.sort(key=lambda sample: np.array2string(sample.x))
        expected_samples = [
            make_solution_sample(x=np.array([0, 0, 0]), probability=0.2434, problem=self.problem),
            make_solution_sample(x=np.array([0, 0, 1]), probability=0.1234, problem=self.problem),
            make_solution_sample(x=np.array([0, 1, 0]), probability=0.1653, problem=self.problem),
            make_solution_sample(x=np.array([0, 1, 1]), probability=0.0469, problem=self.problem),
            make_solution_sample(x=np.array([1, 0, 0]), probability=0.2058, problem=self.problem),
            make_solution_sample(x=np.array([1, 0, 1]), probability=0.0833, problem=self.problem),
            make_solution_sample(x=np.array([1, 1, 0]), probability=0.1219, problem=self.problem),
            make_solution_sample(x=np.array([1, 1, 1]), probability=0.01, problem=self.problem),
        ]
        for i, sample in enumerate(samples):
            np.testing.assert_allclose(sample.x, expected_samples[i].x)
            self.assertAlmostEqual(sample.probability, expected_samples[i].probability, delta=0.05)
        np.testing.assert_allclose(
            rounding_result.expectation_values,
            [0.2672612419124245, 0.5345224838248487, 0.8017837257372733],
        )

    def test_magic_rounding_round_weighted_3_1_qrac(self):
        """Test round method with weighted basis sampling for max_vars_per_qubit=3"""
        encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=3)
        encoding.encode(self.problem)
        np_solver = NumPyMinimumEigensolver()
        qrao = QuantumRandomAccessOptimizer(min_eigen_solver=np_solver)
        _, rounding_context = qrao.solve_relaxed(encoding=encoding)
        sampler = Sampler(options={"shots": 10000, "seed": 42})
        magic_rounding = MagicRounding(sampler, basis_sampling="weighted", seed=42)
        rounding_result = magic_rounding.round(rounding_context)
        self.assertIsInstance(rounding_result, RoundingResult)
        np.testing.assert_allclose(rounding_result.bases, [[0], [1], [2], [3]])
        np.testing.assert_allclose(rounding_result.basis_shots, [4499, 2700, 1574, 1227])
        expected_basis_counts = [
            {"0": 4352.0, "1": 147.0},
            {"0": 500.0, "1": 2200.0},
            {"0": 528.0, "1": 1046.0},
            {"0": 630.0, "1": 597.0},
        ]
        for i, basis_counts in enumerate(rounding_result.basis_counts):
            for key, value in basis_counts.items():
                self.assertAlmostEqual(value, expected_basis_counts[i][key], delta=50)
        samples = rounding_result.samples
        samples.sort(key=lambda sample: np.array2string(sample.x))
        expected_samples = [
            make_solution_sample(x=np.array([0, 0, 0]), probability=0.4352, problem=self.problem),
            make_solution_sample(x=np.array([0, 0, 1]), probability=0.0597, problem=self.problem),
            make_solution_sample(x=np.array([0, 1, 0]), probability=0.1046, problem=self.problem),
            make_solution_sample(x=np.array([0, 1, 1]), probability=0.05, problem=self.problem),
            make_solution_sample(x=np.array([1, 0, 0]), probability=0.22, problem=self.problem),
            make_solution_sample(x=np.array([1, 0, 1]), probability=0.0528, problem=self.problem),
            make_solution_sample(x=np.array([1, 1, 0]), probability=0.063, problem=self.problem),
            make_solution_sample(x=np.array([1, 1, 1]), probability=0.0147, problem=self.problem),
        ]
        for i, sample in enumerate(samples):
            np.testing.assert_allclose(sample.x, expected_samples[i].x)
            self.assertAlmostEqual(sample.probability, expected_samples[i].probability, delta=0.005)
        np.testing.assert_allclose(
            rounding_result.expectation_values,
            [0.2672612419124245, 0.5345224838248487, 0.8017837257372733],
        )

    def test_mapping_magic_rounding_result(self):
        """Test the mapping of magic rounding result bits to variables"""
        encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=1)
        encoding.encode(self.maxcut_problem)
        circuit = encoding.state_preparation_circuit(self.maxcut_optimal_solution)
        rounding_context = RoundingContext(encoding=encoding, expectation_values=0, circuit=circuit)
        sampler = Sampler(options={"shots": 1})
        magic_rounding = MagicRounding(sampler=sampler)
        rounding_result = magic_rounding.round(rounding_context)
        solution = rounding_result.samples[0]
        self.assertEqual(solution.fval, self.maxcut_optimal_value)
        np.testing.assert_allclose(solution.x, self.maxcut_optimal_solution)

    def test_magic_rounding_exceptions(self):
        """Test exceptions in the MagicRounding class"""
        encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=3)
        encoding.encode(self.problem)

        with self.assertRaises(ValueError):
            # circuit is None
            sampler = Sampler(options={"shots": 10000, "seed": 42})
            magic_rounding = MagicRounding(sampler=sampler)
            rounding_context = RoundingContext(encoding, expectation_values=[1, 1, 1], circuit=None)
            magic_rounding.round(rounding_context)

        with self.assertRaises(ValueError):
            # sampler without shots
            sampler = Sampler()
            magic_rounding = MagicRounding(sampler=sampler)

        with self.assertRaises(ValueError):
            # expectation_values is None for weighted basis sampling
            sampler = Sampler()
            magic_rounding = MagicRounding(sampler=sampler, basis_sampling="weighted")
            rounding_context = RoundingContext(
                encoding, expectation_values=None, circuit=QuantumCircuit(1)
            )
            magic_rounding.round(rounding_context)

        with self.assertRaises(ValueError):
            # vars_per_qubit is invalid
            sampler = Sampler(options={"shots": 10000, "seed": 42})
            magic_rounding = MagicRounding(sampler=sampler)
            magic_rounding._make_circuits(circuit=QuantumCircuit(1), bases=[[0]], vars_per_qubit=4)


def make_solution_sample(
    x: np.ndarray, probability: float, problem: QuadraticProgram
) -> SolutionSample:
    """Make a solution sample."""
    return SolutionSample(
        x=x,
        fval=problem.objective.evaluate(x),
        probability=probability,
        status=(
            OptimizationResultStatus.SUCCESS
            if problem.is_feasible(x)
            else OptimizationResultStatus.INFEASIBLE
        ),
    )


if __name__ == "__main__":
    unittest.main()
