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

""" Test Min Eigen Optimizer """

import unittest
from test.optimization_test_case import (
    QiskitOptimizationTestCase,
    requires_extra_library,
)

import numpy as np
from ddt import data, ddt
from docplex.mp.model import Model

from qiskit import BasicAer
from qiskit.algorithms import QAOA, VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import COBYLA, SPSA
from qiskit.circuit.library import TwoLocal
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_optimization.algorithms import CplexOptimizer, MinimumEigenOptimizer
from qiskit_optimization.algorithms.optimization_algorithm import (
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
class TestMinEigenOptimizer(QiskitOptimizationTestCase):
    """Min Eigen Optimizer Tests."""

    def setUp(self):
        super().setUp()

        # setup minimum eigen solvers
        self.min_eigen_solvers = {}

        # exact eigen solver
        self.min_eigen_solvers["exact"] = NumPyMinimumEigensolver()

        # QAOA
        optimizer = COBYLA()
        self.min_eigen_solvers["qaoa"] = QAOA(optimizer=optimizer)
        # simulators
        self.sv_simulator = QuantumInstance(
            BasicAer.get_backend("statevector_simulator"),
            seed_simulator=123,
            seed_transpiler=123,
        )
        self.qasm_simulator = QuantumInstance(
            BasicAer.get_backend("qasm_simulator"),
            seed_simulator=123,
            seed_transpiler=123,
        )
        # test minimize
        self.op_minimize = QuadraticProgram()
        self.op_minimize.integer_var(0, 3, "x")
        self.op_minimize.binary_var("y")
        self.op_minimize.minimize(linear={"x": 1, "y": 2})
        self.op_minimize.linear_constraint(linear={"x": 1, "y": 1}, sense=">=", rhs=1, name="xy")

        # test maximize
        self.op_maximize = QuadraticProgram()
        self.op_maximize.integer_var(0, 3, "x")
        self.op_maximize.binary_var("y")
        self.op_maximize.maximize(linear={"x": 1, "y": 2})
        self.op_maximize.linear_constraint(linear={"x": 1, "y": 1}, sense="<=", rhs=1, name="xy")

        # test bit ordering
        mdl = Model("docplex model")
        x = mdl.binary_var("x")
        y = mdl.binary_var("y")
        mdl.minimize(x - 2 * y)
        self.op_ordering = from_docplex_mp(mdl)

    @data(
        ("exact", None, "op_ip1.lp"),
        ("qaoa", "statevector_simulator", "op_ip1.lp"),
        ("qaoa", "qasm_simulator", "op_ip1.lp"),
    )
    @requires_extra_library
    def test_min_eigen_optimizer(self, config):
        """Min Eigen Optimizer Test"""
        try:
            # unpack configuration
            min_eigen_solver_name, backend, filename = config

            # get minimum eigen solver
            min_eigen_solver = self.min_eigen_solvers[min_eigen_solver_name]
            if backend:
                min_eigen_solver.quantum_instance = BasicAer.get_backend(backend)

            # construct minimum eigen optimizer
            min_eigen_optimizer = MinimumEigenOptimizer(min_eigen_solver)

            # load optimization problem
            problem = QuadraticProgram()
            lp_file = self.get_resource_path(filename, "algorithms/resources")
            problem.read_from_lp_file(lp_file)

            # solve problem with cplex
            cplex = CplexOptimizer()
            cplex_result = cplex.solve(problem)

            # solve problem
            result = min_eigen_optimizer.solve(problem)
            self.assertIsNotNone(result)

            # analyze results
            self.assertAlmostEqual(cplex_result.fval, result.fval)

            # check that eigensolver result is present
            self.assertIsNotNone(result.min_eigen_solver_result)
        except RuntimeError as ex:
            self.fail(str(ex))

    @data(
        ("op_ip1.lp", -470, 12, OptimizationResultStatus.SUCCESS),
        ("op_ip1.lp", np.inf, None, OptimizationResultStatus.FAILURE),
    )
    @requires_extra_library
    def test_min_eigen_optimizer_with_filter(self, config):
        """Min Eigen Optimizer Test"""
        try:
            # unpack configuration
            filename, lowerbound, fval, status = config

            # get minimum eigen solver
            min_eigen_solver = NumPyMinimumEigensolver()

            # set filter
            # pylint: disable=unused-argument
            def filter_criterion(x, v, aux):
                return v > lowerbound

            min_eigen_solver.filter_criterion = filter_criterion

            # construct minimum eigen optimizer
            min_eigen_optimizer = MinimumEigenOptimizer(min_eigen_solver)

            # load optimization problem
            problem = QuadraticProgram()
            lp_file = self.get_resource_path(filename, "algorithms/resources")
            problem.read_from_lp_file(lp_file)

            # solve problem
            result = min_eigen_optimizer.solve(problem)
            self.assertIsNotNone(result)

            # analyze results
            self.assertAlmostEqual(fval, result.fval)
            self.assertEqual(status, result.status)

            # check that eigensolver result is present
            self.assertIsNotNone(result.min_eigen_solver_result)
        except RuntimeError as ex:
            self.fail(str(ex))

    def test_converter_list(self):
        """Test converter list"""
        op = QuadraticProgram()
        op.integer_var(0, 3, "x")
        op.binary_var("y")

        op.maximize(linear={"x": 1, "y": 2})
        op.linear_constraint(linear={"x": 1, "y": 1}, sense="LE", rhs=3, name="xy_leq")
        min_eigen_solver = NumPyMinimumEigensolver()
        # a single converter
        qp2qubo = QuadraticProgramToQubo()
        min_eigen_optimizer = MinimumEigenOptimizer(min_eigen_solver, converters=qp2qubo)
        result = min_eigen_optimizer.solve(op)
        self.assertEqual(result.fval, 4)
        # a list of converters
        ineq2eq = InequalityToEquality()
        int2bin = IntegerToBinary()
        penalize = LinearEqualityToPenalty()
        max2min = MaximizeToMinimize()
        converters = [ineq2eq, int2bin, penalize, max2min]
        min_eigen_optimizer = MinimumEigenOptimizer(min_eigen_solver, converters=converters)
        result = min_eigen_optimizer.solve(op)
        self.assertEqual(result.fval, 4)
        with self.assertRaises(TypeError):
            invalid = [qp2qubo, "invalid converter"]
            MinimumEigenOptimizer(min_eigen_solver, converters=invalid)

    def test_samples_numpy_eigen_solver(self):
        """Test samples for NumPyMinimumEigensolver"""
        # test minimize
        min_eigen_solver = NumPyMinimumEigensolver()
        min_eigen_optimizer = MinimumEigenOptimizer(min_eigen_solver)
        result = min_eigen_optimizer.solve(self.op_minimize)
        opt_sol = 1
        success = OptimizationResultStatus.SUCCESS
        self.assertEqual(result.fval, opt_sol)
        self.assertEqual(len(result.samples), 1)
        np.testing.assert_array_almost_equal(result.samples[0].x, [1, 0])
        self.assertAlmostEqual(result.samples[0].fval, opt_sol)
        self.assertAlmostEqual(result.samples[0].probability, 1.0)
        self.assertEqual(result.samples[0].status, success)
        self.assertEqual(len(result.raw_samples), 1)
        np.testing.assert_array_almost_equal(result.raw_samples[0].x, [1, 0, 0, 0, 0])
        self.assertAlmostEqual(result.raw_samples[0].fval, opt_sol)
        self.assertAlmostEqual(result.raw_samples[0].probability, 1.0)
        self.assertEqual(result.raw_samples[0].status, success)
        # test maximize
        min_eigen_solver = NumPyMinimumEigensolver()
        min_eigen_optimizer = MinimumEigenOptimizer(min_eigen_solver)
        result = min_eigen_optimizer.solve(self.op_maximize)
        opt_sol = 2
        self.assertEqual(result.fval, opt_sol)
        self.assertEqual(len(result.samples), 1)
        np.testing.assert_array_almost_equal(result.samples[0].x, [0, 1])
        self.assertAlmostEqual(result.samples[0].fval, opt_sol)
        self.assertAlmostEqual(result.samples[0].probability, 1.0)
        self.assertEqual(result.samples[0].status, success)
        self.assertEqual(len(result.raw_samples), 1)
        np.testing.assert_array_almost_equal(result.raw_samples[0].x, [0, 0, 1, 0])
        # optimizer internally deals with minimization problem
        self.assertAlmostEqual(
            self.op_maximize.objective.sense.value * result.raw_samples[0].fval, opt_sol
        )
        self.assertAlmostEqual(result.raw_samples[0].probability, 1.0)
        self.assertEqual(result.raw_samples[0].status, success)

    @data("sv", "qasm")
    def test_samples_qaoa(self, simulator):
        """Test samples for QAOA"""
        # test minimize
        algorithm_globals.random_seed = 4
        quantum_instance = self.sv_simulator if simulator == "sv" else self.qasm_simulator
        qaoa = QAOA(quantum_instance=quantum_instance, reps=2)
        min_eigen_optimizer = MinimumEigenOptimizer(qaoa)
        result = min_eigen_optimizer.solve(self.op_minimize)
        success = OptimizationResultStatus.SUCCESS
        opt_sol = 1
        self.assertEqual(len(result.samples), 8)
        self.assertEqual(len(result.raw_samples), 32)
        self.assertAlmostEqual(sum(s.probability for s in result.samples), 1)
        self.assertAlmostEqual(sum(s.probability for s in result.raw_samples), 1)
        self.assertAlmostEqual(min(s.fval for s in result.samples), 0)
        self.assertAlmostEqual(min(s.fval for s in result.samples if s.status == success), opt_sol)
        self.assertAlmostEqual(min(s.fval for s in result.raw_samples), opt_sol)
        for sample in result.raw_samples:
            self.assertEqual(sample.status, success)
        np.testing.assert_array_almost_equal(result.x, [1, 0])
        self.assertAlmostEqual(result.fval, result.samples[0].fval)
        self.assertEqual(result.status, result.samples[0].status)
        self.assertAlmostEqual(result.samples[0].fval, opt_sol)
        self.assertEqual(result.samples[0].status, success)
        np.testing.assert_array_almost_equal(result.raw_samples[0].x, [1, 0, 0, 0, 0])
        self.assertAlmostEqual(result.raw_samples[0].fval, opt_sol)
        self.assertEqual(result.raw_samples[0].status, success)
        # test maximize
        opt_sol = 2
        qaoa = QAOA(quantum_instance=quantum_instance, reps=2)
        min_eigen_optimizer = MinimumEigenOptimizer(qaoa)
        result = min_eigen_optimizer.solve(self.op_maximize)
        self.assertEqual(len(result.samples), 8)
        self.assertEqual(len(result.raw_samples), 16)
        self.assertAlmostEqual(sum(s.probability for s in result.samples), 1)
        self.assertAlmostEqual(sum(s.probability for s in result.raw_samples), 1)
        self.assertAlmostEqual(max(s.fval for s in result.samples), 5)
        self.assertAlmostEqual(max(s.fval for s in result.samples if s.status == success), opt_sol)
        # optimizer internally deals with minimization problem
        self.assertAlmostEqual(
            max(self.op_maximize.objective.sense.value * s.fval for s in result.raw_samples),
            opt_sol,
        )
        for sample in result.raw_samples:
            self.assertEqual(sample.status, success)
        np.testing.assert_array_almost_equal(result.x, [0, 1])
        self.assertEqual(result.fval, opt_sol)
        self.assertEqual(result.status, success)
        np.testing.assert_array_almost_equal(result.samples[0].x, [0, 1])
        self.assertAlmostEqual(result.samples[0].fval, opt_sol)
        self.assertEqual(result.samples[0].status, success)
        np.testing.assert_array_almost_equal(result.raw_samples[0].x, [0, 0, 1, 0])
        # optimizer internally deals with minimization problem
        self.assertAlmostEqual(
            self.op_maximize.objective.sense.value * result.raw_samples[0].fval, opt_sol
        )
        self.assertEqual(result.raw_samples[0].status, success)
        # test bit ordering
        opt_sol = -2
        qaoa = QAOA(quantum_instance=quantum_instance, reps=2)
        min_eigen_optimizer = MinimumEigenOptimizer(qaoa)
        result = min_eigen_optimizer.solve(self.op_ordering)
        self.assertEqual(result.fval, opt_sol)
        np.testing.assert_array_almost_equal(result.x, [0, 1])
        self.assertEqual(result.status, success)
        result.raw_samples.sort(key=lambda x: x.probability, reverse=True)
        np.testing.assert_array_almost_equal(result.x, result.raw_samples[0].x)
        self.assertAlmostEqual(sum(s.probability for s in result.samples), 1, delta=1e-5)
        self.assertAlmostEqual(sum(s.probability for s in result.raw_samples), 1, delta=1e-5)
        self.assertAlmostEqual(min(s.fval for s in result.samples), -2)
        self.assertAlmostEqual(min(s.fval for s in result.samples if s.status == success), opt_sol)
        self.assertAlmostEqual(min(s.fval for s in result.raw_samples), opt_sol)
        for sample in result.raw_samples:
            self.assertEqual(sample.status, success)
        np.testing.assert_array_almost_equal(result.samples[0].x, [0, 1])
        self.assertAlmostEqual(result.samples[0].fval, opt_sol)
        self.assertEqual(result.samples[0].status, success)
        np.testing.assert_array_almost_equal(result.raw_samples[0].x, [0, 1])
        self.assertAlmostEqual(result.raw_samples[0].fval, opt_sol)
        self.assertEqual(result.raw_samples[0].status, success)

    @data("sv", "qasm")
    def test_samples_vqe(self, simulator):
        """Test samples for VQE"""
        # test minimize
        algorithm_globals.random_seed = 1
        quantum_instance = self.sv_simulator if simulator == "sv" else self.qasm_simulator
        opt_sol = -2
        success = OptimizationResultStatus.SUCCESS
        optimizer = SPSA(maxiter=100)
        ry_ansatz = TwoLocal(5, "ry", "cz", reps=3, entanglement="full")
        vqe_mes = VQE(ry_ansatz, optimizer=optimizer, quantum_instance=quantum_instance)
        vqe = MinimumEigenOptimizer(vqe_mes)
        results = vqe.solve(self.op_ordering)
        self.assertEqual(results.fval, opt_sol)
        np.testing.assert_array_almost_equal(results.x, [0, 1])
        self.assertEqual(results.status, success)
        results.raw_samples.sort(key=lambda x: x.probability, reverse=True)
        np.testing.assert_array_almost_equal(results.x, results.raw_samples[0].x)
        self.assertAlmostEqual(sum(s.probability for s in results.samples), 1, delta=1e-5)
        self.assertAlmostEqual(sum(s.probability for s in results.raw_samples), 1, delta=1e-5)
        self.assertAlmostEqual(min(s.fval for s in results.samples), -2)
        self.assertAlmostEqual(min(s.fval for s in results.samples if s.status == success), opt_sol)
        self.assertAlmostEqual(min(s.fval for s in results.raw_samples), opt_sol)
        for sample in results.raw_samples:
            self.assertEqual(sample.status, success)
        np.testing.assert_array_almost_equal(results.samples[0].x, [0, 1])
        self.assertAlmostEqual(results.samples[0].fval, opt_sol)
        self.assertEqual(results.samples[0].status, success)
        np.testing.assert_array_almost_equal(results.raw_samples[0].x, [0, 1])
        self.assertAlmostEqual(results.raw_samples[0].fval, opt_sol)
        self.assertEqual(results.raw_samples[0].status, success)


if __name__ == "__main__":
    unittest.main()
