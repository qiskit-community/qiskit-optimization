# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Recursive Min Eigen Optimizer."""

import unittest
from test import QiskitOptimizationTestCase, requires_extra_library

import numpy as np

from qiskit import BasicAer
from qiskit.utils import algorithm_globals

from qiskit.algorithms import NumPyMinimumEigensolver, QAOA

from qiskit_optimization.algorithms import (
    MinimumEigenOptimizer,
    CplexOptimizer,
    RecursiveMinimumEigenOptimizer,
    WarmStartQAOAOptimizer,
    SlsqpOptimizer,
)
from qiskit_optimization.algorithms.recursive_minimum_eigen_optimizer import (
    IntermediateResult,
)
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.converters import (
    IntegerToBinary,
    InequalityToEquality,
    LinearEqualityToPenalty,
    QuadraticProgramToQubo,
)


class TestRecursiveMinEigenOptimizer(QiskitOptimizationTestCase):
    """Recursive Min Eigen Optimizer Tests."""

    @requires_extra_library
    def test_recursive_min_eigen_optimizer(self):
        """Test the recursive minimum eigen optimizer."""
        filename = "op_ip1.lp"
        # get minimum eigen solver
        min_eigen_solver = NumPyMinimumEigensolver()

        # construct minimum eigen optimizer
        min_eigen_optimizer = MinimumEigenOptimizer(min_eigen_solver)
        recursive_min_eigen_optimizer = RecursiveMinimumEigenOptimizer(
            min_eigen_optimizer, min_num_vars=4
        )

        # load optimization problem
        problem = QuadraticProgram()
        lp_file = self.get_resource_path(filename, "algorithms/resources")
        problem.read_from_lp_file(lp_file)

        # solve problem with cplex
        cplex = CplexOptimizer()
        cplex_result = cplex.solve(problem)

        # solve problem
        result = recursive_min_eigen_optimizer.solve(problem)

        # analyze results
        np.testing.assert_array_almost_equal(cplex_result.x, result.x, 4)
        self.assertAlmostEqual(cplex_result.fval, result.fval)

    @requires_extra_library
    def test_recursive_history(self):
        """Tests different options for history."""
        filename = "op_ip1.lp"
        # load optimization problem
        problem = QuadraticProgram()
        lp_file = self.get_resource_path(filename, "algorithms/resources")
        problem.read_from_lp_file(lp_file)

        # get minimum eigen solver
        min_eigen_solver = NumPyMinimumEigensolver()

        # construct minimum eigen optimizer
        min_eigen_optimizer = MinimumEigenOptimizer(min_eigen_solver)

        # no history
        recursive_min_eigen_optimizer = RecursiveMinimumEigenOptimizer(
            min_eigen_optimizer,
            min_num_vars=4,
            history=IntermediateResult.NO_ITERATIONS,
        )
        result = recursive_min_eigen_optimizer.solve(problem)
        self.assertIsNotNone(result.replacements)
        self.assertIsNotNone(result.history)
        self.assertIsNotNone(result.history[0])
        self.assertEqual(len(result.history[0]), 0)
        self.assertIsNone(result.history[1])

        # only last iteration in the history
        recursive_min_eigen_optimizer = RecursiveMinimumEigenOptimizer(
            min_eigen_optimizer,
            min_num_vars=4,
            history=IntermediateResult.LAST_ITERATION,
        )
        result = recursive_min_eigen_optimizer.solve(problem)
        self.assertIsNotNone(result.replacements)
        self.assertIsNotNone(result.history)
        self.assertIsNotNone(result.history[0])
        self.assertEqual(len(result.history[0]), 0)
        self.assertIsNotNone(result.history[1])

        # full history
        recursive_min_eigen_optimizer = RecursiveMinimumEigenOptimizer(
            min_eigen_optimizer,
            min_num_vars=4,
            history=IntermediateResult.ALL_ITERATIONS,
        )
        result = recursive_min_eigen_optimizer.solve(problem)
        self.assertIsNotNone(result.replacements)
        self.assertIsNotNone(result.history)
        self.assertIsNotNone(result.history[0])
        self.assertGreater(len(result.history[0]), 1)
        self.assertIsNotNone(result.history[1])

    @requires_extra_library
    def test_recursive_warm_qaoa(self):
        """Test the recursive optimizer with warm start qaoa."""
        algorithm_globals.random_seed = 12345
        backend = BasicAer.get_backend("statevector_simulator")
        qaoa = QAOA(quantum_instance=backend, reps=1)
        warm_qaoa = WarmStartQAOAOptimizer(
            pre_solver=SlsqpOptimizer(), relax_for_pre_solver=True, qaoa=qaoa
        )

        recursive_min_eigen_optimizer = RecursiveMinimumEigenOptimizer(warm_qaoa, min_num_vars=4)

        # load optimization problem
        problem = QuadraticProgram()
        lp_file = self.get_resource_path("op_ip1.lp", "algorithms/resources")
        problem.read_from_lp_file(lp_file)

        # solve problem with cplex
        cplex = CplexOptimizer()
        cplex_result = cplex.solve(problem)

        # solve problem
        result = recursive_min_eigen_optimizer.solve(problem)

        # analyze results
        np.testing.assert_array_almost_equal(cplex_result.x, result.x, 4)
        self.assertAlmostEqual(cplex_result.fval, result.fval)

    def test_converter_list(self):
        """Test converter list"""
        op = QuadraticProgram()
        op.integer_var(0, 3, "x")
        op.binary_var("y")

        op.maximize(linear={"x": 1, "y": 2})
        op.linear_constraint(linear={"y": 1, "x": 1}, sense="LE", rhs=3, name="xy_leq")

        # construct minimum eigen optimizer
        min_eigen_solver = NumPyMinimumEigensolver()
        min_eigen_optimizer = MinimumEigenOptimizer(min_eigen_solver)
        # a single converter
        qp2qubo = QuadraticProgramToQubo()
        recursive_min_eigen_optimizer = RecursiveMinimumEigenOptimizer(
            min_eigen_optimizer, min_num_vars=2, converters=qp2qubo
        )
        result = recursive_min_eigen_optimizer.solve(op)
        self.assertEqual(result.fval, 4)
        # a list of converters
        ineq2eq = InequalityToEquality()
        int2bin = IntegerToBinary()
        penalize = LinearEqualityToPenalty()
        converters = [ineq2eq, int2bin, penalize]
        recursive_min_eigen_optimizer = RecursiveMinimumEigenOptimizer(
            min_eigen_optimizer, min_num_vars=2, converters=converters
        )
        result = recursive_min_eigen_optimizer.solve(op)
        self.assertEqual(result.fval, 4)
        # invalid converters
        with self.assertRaises(TypeError):
            invalid = [qp2qubo, "invalid converter"]
            RecursiveMinimumEigenOptimizer(min_eigen_optimizer, min_num_vars=2, converters=invalid)


if __name__ == "__main__":
    unittest.main()
