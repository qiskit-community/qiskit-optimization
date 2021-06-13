# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Grover Optimizer."""

import unittest
from test import QiskitOptimizationTestCase

import numpy as np
from ddt import data, ddt
from docplex.mp.model import Model
from qiskit import Aer
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import (
    GroverOptimizer,
    MinimumEigenOptimizer,
    OptimizationResultStatus,
)
from qiskit_optimization.converters import (
    InequalityToEquality,
    IntegerToBinary,
    LinearEqualityToPenalty,
    MaximizeToMinimize,
    QuadraticProgramToQubo,
)
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp


@ddt
class TestGroverOptimizer(QiskitOptimizationTestCase):
    """GroverOptimizer tests."""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 1
        self.sv_simulator = QuantumInstance(
            Aer.get_backend("aer_simulator_statevector"),
            seed_simulator=921,
            seed_transpiler=200,
        )
        self.qasm_simulator = QuantumInstance(
            Aer.get_backend("aer_simulator"), seed_simulator=123, seed_transpiler=123
        )
        self.n_iter = 8

    def validate_results(self, problem, results):
        """Validate the results object returned by GroverOptimizer."""
        # Get expected value.
        solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        comp_result = solver.solve(problem)
        # Validate results.
        np.testing.assert_array_almost_equal(comp_result.x, results.x)
        self.assertEqual(comp_result.fval, results.fval)
        # optimizer internally deals with minimization problem
        self.assertAlmostEqual(
            results.fval, problem.objective.sense.value * results.intermediate_fval
        )

    def test_qubo_gas_int_zero(self):
        """Test for when the answer is zero."""

        # Input.
        model = Model()
        x_0 = model.binary_var(name="x0")
        x_1 = model.binary_var(name="x1")
        model.minimize(0 * x_0 + 0 * x_1)
        op = from_docplex_mp(model)

        # Will not find a negative, should return 0.
        gmf = GroverOptimizer(1, num_iterations=1, quantum_instance=self.sv_simulator)
        results = gmf.solve(op)
        np.testing.assert_array_almost_equal(results.x, [0, 0])
        self.assertEqual(results.fval, 0.0)
        self.assertAlmostEqual(results.fval, results.intermediate_fval)

    def test_qubo_gas_int_simple(self):
        """Test for simple case, with 2 linear coeffs and no quadratic coeffs or constants."""

        # Input.
        model = Model()
        x_0 = model.binary_var(name="x0")
        x_1 = model.binary_var(name="x1")
        model.minimize(-x_0 + 2 * x_1)
        op = from_docplex_mp(model)

        # Get the optimum key and value.
        gmf = GroverOptimizer(4, num_iterations=self.n_iter, quantum_instance=self.sv_simulator)
        results = gmf.solve(op)
        self.validate_results(op, results)

        self.assertIsNotNone(results.operation_counts)
        self.assertEqual(results.n_input_qubits, 2)
        self.assertEqual(results.n_output_qubits, 4)

    def test_qubo_gas_int_simple_maximize(self):
        """Test for simple case, but with maximization."""

        # Input.
        model = Model()
        x_0 = model.binary_var(name="x0")
        x_1 = model.binary_var(name="x1")
        model.maximize(-x_0 + 2 * x_1)
        op = from_docplex_mp(model)

        # Get the optimum key and value.
        gmf = GroverOptimizer(4, num_iterations=self.n_iter, quantum_instance=self.sv_simulator)
        results = gmf.solve(op)
        self.validate_results(op, results)

    @data("sv", "qasm")
    def test_qubo_gas_int_paper_example(self, simulator):
        """
        Test the example from https://arxiv.org/abs/1912.04088 using the state vector simulator
        and the qasm simulator
        """

        # Input.
        model = Model()
        x_0 = model.binary_var(name="x0")
        x_1 = model.binary_var(name="x1")
        x_2 = model.binary_var(name="x2")
        model.minimize(-x_0 + 2 * x_1 - 3 * x_2 - 2 * x_0 * x_2 - 1 * x_1 * x_2)
        op = from_docplex_mp(model)

        # Get the optimum key and value.
        q_instance = self.sv_simulator if simulator == "sv" else self.qasm_simulator
        gmf = GroverOptimizer(6, num_iterations=self.n_iter, quantum_instance=q_instance)
        results = gmf.solve(op)
        self.validate_results(op, results)

    def test_converter_list(self):
        """Test converters list"""
        # Input.

        model = Model()
        x_0 = model.binary_var(name="x0")
        x_1 = model.binary_var(name="x1")
        model.maximize(-x_0 + 2 * x_1)
        op = from_docplex_mp(model)

        # Get the optimum key and value.
        # a single converter.
        qp2qubo = QuadraticProgramToQubo()
        gmf = GroverOptimizer(
            4,
            num_iterations=self.n_iter,
            quantum_instance=self.sv_simulator,
            converters=qp2qubo,
        )
        results = gmf.solve(op)
        self.validate_results(op, results)

        # a list of converters
        ineq2eq = InequalityToEquality()
        int2bin = IntegerToBinary()
        penalize = LinearEqualityToPenalty()
        max2min = MaximizeToMinimize()
        converters = [ineq2eq, int2bin, penalize, max2min]
        gmf = GroverOptimizer(
            4,
            num_iterations=self.n_iter,
            quantum_instance=self.sv_simulator,
            converters=converters,
        )
        results = gmf.solve(op)
        self.validate_results(op, results)
        # invalid converters
        with self.assertRaises(TypeError):
            invalid = [qp2qubo, "invalid converter"]
            GroverOptimizer(
                4,
                num_iterations=self.n_iter,
                quantum_instance=self.sv_simulator,
                converters=invalid,
            )

    @data("sv", "qasm")
    def test_samples_and_raw_samples(self, simulator):
        """Test samples and raw_samples"""
        algorithm_globals.random_seed = 2
        op = QuadraticProgram()
        op.integer_var(0, 3, "x")
        op.binary_var("y")
        op.minimize(linear={"x": 1, "y": 2})
        op.linear_constraint(linear={"x": 1, "y": 1}, sense=">=", rhs=1, name="xy")
        q_instance = self.sv_simulator if simulator == "sv" else self.qasm_simulator
        grover_optimizer = GroverOptimizer(
            8, num_iterations=self.n_iter, quantum_instance=q_instance
        )
        opt_sol = 1
        success = OptimizationResultStatus.SUCCESS
        results = grover_optimizer.solve(op)
        self.assertEqual(len(results.samples), 8)
        self.assertEqual(len(results.raw_samples), 32)
        self.assertAlmostEqual(sum(s.probability for s in results.samples), 1)
        self.assertAlmostEqual(sum(s.probability for s in results.raw_samples), 1)
        self.assertAlmostEqual(min(s.fval for s in results.samples), 0)
        self.assertAlmostEqual(min(s.fval for s in results.samples if s.status == success), opt_sol)
        self.assertAlmostEqual(min(s.fval for s in results.raw_samples), opt_sol)
        for sample in results.raw_samples:
            self.assertEqual(sample.status, success)
        np.testing.assert_array_almost_equal(results.x, results.samples[0].x)
        self.assertAlmostEqual(results.fval, results.samples[0].fval)
        self.assertEqual(results.status, results.samples[0].status)
        self.assertAlmostEqual(results.fval, results.raw_samples[0].fval)
        self.assertEqual(results.status, results.raw_samples[0].status)
        np.testing.assert_array_almost_equal([1, 0, 0, 0, 0], results.raw_samples[0].x)

    @data("sv", "qasm")
    def test_bit_ordering(self, simulator):
        """Test bit ordering"""
        # test minimize
        algorithm_globals.random_seed = 2
        q_instance = self.sv_simulator if simulator == "sv" else self.qasm_simulator
        mdl = Model("docplex model")
        x = mdl.binary_var("x")
        y = mdl.binary_var("y")
        mdl.minimize(x - 2 * y)
        op = from_docplex_mp(mdl)
        opt_sol = -2
        success = OptimizationResultStatus.SUCCESS
        grover_optimizer = GroverOptimizer(
            3, num_iterations=self.n_iter, quantum_instance=q_instance
        )
        results = grover_optimizer.solve(op)
        self.assertEqual(results.fval, opt_sol)
        np.testing.assert_array_almost_equal(results.x, [0, 1])
        self.assertEqual(results.status, success)
        results.raw_samples.sort(key=lambda x: x.probability, reverse=True)
        self.assertAlmostEqual(sum(s.probability for s in results.samples), 1, delta=1e-5)
        self.assertAlmostEqual(sum(s.probability for s in results.raw_samples), 1, delta=1e-5)
        self.assertAlmostEqual(min(s.fval for s in results.samples), -2)
        self.assertAlmostEqual(min(s.fval for s in results.samples if s.status == success), opt_sol)
        self.assertAlmostEqual(min(s.fval for s in results.raw_samples), opt_sol)
        for sample in results.raw_samples:
            self.assertEqual(sample.status, success)


if __name__ == "__main__":
    unittest.main()
