# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test from_ising and to_ising"""

from contextlib import nullcontext
from test.optimization_test_case import QiskitOptimizationTestCase

import numpy as np
from ddt import data, ddt
from qiskit.opflow import I, OperatorBase, PauliSumOp, Z
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

from qiskit_optimization.exceptions import QiskitOptimizationError
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.translators import from_ising, to_ising


@ddt
class TestIsingTranslator(QiskitOptimizationTestCase):
    """Test from_ising and to_ising"""

    def op_from_list(self, lst, opflow):
        """generate an operator from a list"""
        if opflow:
            with self.assertWarns(DeprecationWarning):
                return PauliSumOp.from_list(lst)
        else:
            return SparsePauliOp.from_list(lst)

    def assert_deprecation_warning(self, opflow: bool):
        """assert deprecation warning if opflow is used"""
        if opflow:
            return self.assertWarns(DeprecationWarning)
        else:
            return nullcontext()

    @data(True, False, None)
    def test_to_ising(self, opflow):
        """test to_ising"""

        with self.subTest("minimize"):
            # minimize: x + x * y
            # subject to: x, y \in {0, 1}
            q_p = QuadraticProgram("test")
            q_p.binary_var(name="x")
            q_p.binary_var(name="y")
            q_p.minimize(linear={"x": 1}, quadratic={("x", "y"): 1})
            with self.assert_deprecation_warning(opflow in [None, True]):
                op, offset = to_ising(q_p, opflow)
            if opflow in [None, True]:
                self.assertIsInstance(op, OperatorBase)
            else:
                self.assertIsInstance(op, BaseOperator)
            op_ref = SparsePauliOp.from_list([("ZI", -0.25), ("IZ", -0.75), ("ZZ", 0.25)])
            np.testing.assert_allclose(op.to_matrix(), op_ref.to_matrix())
            self.assertAlmostEqual(offset, 0.75)

        with self.subTest("maximize"):
            # maximize: x + x * y
            # subject to: x, y \in {0, 1}
            q_p = QuadraticProgram("test")
            q_p.binary_var(name="x")
            q_p.binary_var(name="y")
            q_p.maximize(linear={"x": 1}, quadratic={("x", "y"): 1})
            with self.assert_deprecation_warning(opflow in [None, True]):
                op, offset = to_ising(q_p, opflow)
            if opflow in [None, True]:
                self.assertIsInstance(op, OperatorBase)
            else:
                self.assertIsInstance(op, BaseOperator)
            op_ref = SparsePauliOp.from_list([("ZI", 0.25), ("IZ", 0.75), ("ZZ", -0.25)])
            np.testing.assert_allclose(op.to_matrix(), op_ref.to_matrix())
            self.assertAlmostEqual(offset, -0.75)

    @data(True, False, None)
    def test_to_ising2(self, opflow):
        """test to_ising 2"""

        with self.subTest("minimize"):
            # minimize: 1 - 2 * x1 - 2 * x2 + 4 * x1 * x2
            # subject to: x, y \in {0, 1}
            q_p = QuadraticProgram("test")
            q_p.binary_var(name="x")
            q_p.binary_var(name="y")
            q_p.minimize(constant=1, linear={"x": -2, "y": -2}, quadratic={("x", "y"): 4})
            with self.assert_deprecation_warning(opflow in [None, True]):
                op, offset = to_ising(q_p, opflow)
            if opflow in [None, True]:
                self.assertIsInstance(op, OperatorBase)
            else:
                self.assertIsInstance(op, BaseOperator)
            op_ref = SparsePauliOp.from_list([("ZZ", 1.0)])
            np.testing.assert_allclose(op.to_matrix(), op_ref.to_matrix())
            self.assertAlmostEqual(offset, 0.0)

        with self.subTest("maximize"):
            # maximize: 1 - 2 * x1 - 2 * x2 + 4 * x1 * x2
            # subject to: x, y \in {0, 1}
            q_p = QuadraticProgram("test")
            q_p.binary_var(name="x")
            q_p.binary_var(name="y")
            q_p.maximize(constant=1, linear={"x": -2, "y": -2}, quadratic={("x", "y"): 4})
            with self.assert_deprecation_warning(opflow in [None, True]):
                op, offset = to_ising(q_p, opflow)
            if opflow in [None, True]:
                self.assertIsInstance(op, OperatorBase)
            else:
                self.assertIsInstance(op, BaseOperator)
            op_ref = SparsePauliOp.from_list([("ZZ", -1.0)])
            np.testing.assert_allclose(op.to_matrix(), op_ref.to_matrix())
            self.assertAlmostEqual(offset, 0.0)

    @data(True, False)
    def test_from_ising(self, opflow):
        """test from_ising"""
        # minimize: x + x * y
        # subject to: x, y \in {0, 1}
        op = self.op_from_list([("ZI", -0.25), ("IZ", -0.75), ("ZZ", 0.25)], opflow)
        with self.subTest("linear: True"):
            with self.assert_deprecation_warning(opflow):
                q_p = from_ising(op, 0.75, linear=True)
            self.assertEqual(q_p.get_num_vars(), op.num_qubits)
            self.assertEqual(q_p.get_num_linear_constraints(), 0)
            self.assertEqual(q_p.get_num_quadratic_constraints(), 0)
            self.assertAlmostEqual(q_p.objective.constant, 0)
            np.testing.assert_allclose(q_p.objective.linear.to_array(), [1, 0])
            np.testing.assert_allclose(q_p.objective.quadratic.to_array(), [[0, 1], [0, 0]])

        with self.subTest("linear: False"):
            with self.assert_deprecation_warning(opflow):
                q_p = from_ising(op, 0.75, linear=False)
            self.assertEqual(q_p.get_num_vars(), op.num_qubits)
            self.assertEqual(q_p.get_num_linear_constraints(), 0)
            self.assertEqual(q_p.get_num_quadratic_constraints(), 0)
            self.assertAlmostEqual(q_p.objective.constant, 0)
            np.testing.assert_allclose(q_p.objective.linear.to_array(), [0, 0])
            np.testing.assert_allclose(q_p.objective.quadratic.to_array(), [[1, 1], [0, 0]])

    @data(True, False)
    def test_from_ising2(self, opflow):
        """test from_ising 2"""
        # minimize: 1 - 2 * x1 - 2 * x2 + 4 * x1 * x2
        # subject to: x, y \in {0, 1}
        op = self.op_from_list([("ZZ", 1)], opflow)
        with self.subTest("linear: True"):
            with self.assert_deprecation_warning(opflow):
                q_p = from_ising(op, 0, linear=True)
            self.assertEqual(q_p.get_num_vars(), op.num_qubits)
            self.assertEqual(q_p.get_num_linear_constraints(), 0)
            self.assertEqual(q_p.get_num_quadratic_constraints(), 0)
            self.assertAlmostEqual(q_p.objective.constant, 1)
            np.testing.assert_allclose(q_p.objective.linear.to_array(), [-2, -2])
            np.testing.assert_allclose(q_p.objective.quadratic.to_array(), [[0, 4], [0, 0]])

        with self.subTest("linear: False"):
            with self.assert_deprecation_warning(opflow):
                q_p = from_ising(op, 0, linear=False)
            self.assertEqual(q_p.get_num_vars(), op.num_qubits)
            self.assertEqual(q_p.get_num_linear_constraints(), 0)
            self.assertEqual(q_p.get_num_quadratic_constraints(), 0)
            self.assertAlmostEqual(q_p.objective.constant, 1)
            np.testing.assert_allclose(q_p.objective.linear.to_array(), [0, 0])
            np.testing.assert_allclose(q_p.objective.quadratic.to_array(), [[-2, 4], [0, -2]])

    @data(True, False)
    def test_from_ising_pauli_with_invalid_paulis(self, opflow):
        """test from_ising with invalid Pauli terms"""
        with self.assertRaises(QiskitOptimizationError):
            op = self.op_from_list([("IX", 1)], opflow)
            with self.assert_deprecation_warning(opflow):
                _ = from_ising(op, 0)

        with self.assertRaises(QiskitOptimizationError):
            op = self.op_from_list([("IY", 1)], opflow)
            with self.assert_deprecation_warning(opflow):
                _ = from_ising(op, 0)

        with self.assertRaises(QiskitOptimizationError):
            op = self.op_from_list([("ZZZ", 1)], opflow)
            with self.assert_deprecation_warning(opflow):
                _ = from_ising(op, 0)

        with self.assertRaises(QiskitOptimizationError):
            op = self.op_from_list([("IZ", 1j)], opflow)
            with self.assert_deprecation_warning(opflow):
                _ = from_ising(op, 0)

    @data(True, False)
    def test_pauli_I_Z(self, opflow):
        """test from_ising and to_ising with Pauli I and Z"""
        with self.subTest("0 * I, linear=False"):
            with self.assert_deprecation_warning(opflow):
                operator = 0 * I if opflow else SparsePauliOp("I", 0)
                q_p = from_ising(operator, linear=False)
            self.assertEqual(q_p.get_num_vars(), 1)
            self.assertEqual(q_p.get_num_linear_constraints(), 0)
            self.assertEqual(q_p.get_num_quadratic_constraints(), 0)
            self.assertAlmostEqual(q_p.objective.constant, 0)
            np.testing.assert_allclose(q_p.objective.linear.to_array(), [0])
            np.testing.assert_allclose(q_p.objective.quadratic.to_array(), [[0]])

            with self.assert_deprecation_warning(opflow):
                op, offset = to_ising(q_p, opflow=opflow)
            np.testing.assert_allclose(op.to_matrix(), np.zeros((2, 2)))
            self.assertAlmostEqual(offset, 0)

        with self.subTest("0 * I, linear=True"):
            with self.assert_deprecation_warning(opflow):
                operator = 0 * I if opflow else SparsePauliOp("I", 0)
                q_p = from_ising(operator, linear=True)
            self.assertEqual(q_p.get_num_vars(), 1)
            self.assertEqual(q_p.get_num_linear_constraints(), 0)
            self.assertEqual(q_p.get_num_quadratic_constraints(), 0)
            self.assertAlmostEqual(q_p.objective.constant, 0)
            np.testing.assert_allclose(q_p.objective.linear.to_array(), [0])
            np.testing.assert_allclose(q_p.objective.quadratic.to_array(), [[0]])

            with self.assert_deprecation_warning(opflow):
                op, offset = to_ising(q_p, opflow=opflow)
            np.testing.assert_allclose(op.to_matrix(), np.zeros((2, 2)))
            self.assertAlmostEqual(offset, 0)

        with self.subTest("2 * I, linear=False"):
            with self.assert_deprecation_warning(opflow):
                operator = 2 * I if opflow else SparsePauliOp("I", 2)
                q_p = from_ising(operator, linear=False)
            self.assertEqual(q_p.get_num_vars(), 1)
            self.assertEqual(q_p.get_num_linear_constraints(), 0)
            self.assertEqual(q_p.get_num_quadratic_constraints(), 0)
            self.assertAlmostEqual(q_p.objective.constant, 2)
            np.testing.assert_allclose(q_p.objective.linear.to_array(), [0])
            np.testing.assert_allclose(q_p.objective.quadratic.to_array(), [[0]])

            with self.assert_deprecation_warning(opflow):
                op, offset = to_ising(q_p, opflow=opflow)
            np.testing.assert_allclose(op.to_matrix(), np.zeros((2, 2)))
            self.assertAlmostEqual(offset, 2)

        with self.subTest("2 * I, linear=True"):
            with self.assert_deprecation_warning(opflow):
                operator = 2 * I if opflow else SparsePauliOp("I", 2)
                q_p = from_ising(operator, linear=True)
            self.assertEqual(q_p.get_num_vars(), 1)
            self.assertEqual(q_p.get_num_linear_constraints(), 0)
            self.assertEqual(q_p.get_num_quadratic_constraints(), 0)
            self.assertAlmostEqual(q_p.objective.constant, 2)
            np.testing.assert_allclose(q_p.objective.linear.to_array(), [0])
            np.testing.assert_allclose(q_p.objective.quadratic.to_array(), [[0]])

            with self.assert_deprecation_warning(opflow):
                op, offset = to_ising(q_p, opflow=opflow)
            np.testing.assert_allclose(op.to_matrix(), np.zeros((2, 2)))
            self.assertAlmostEqual(offset, 2)

        with self.subTest("Z, linear=False"):
            with self.assert_deprecation_warning(opflow):
                operator = Z if opflow else Pauli("Z")
                q_p = from_ising(operator, linear=False)
            self.assertEqual(q_p.get_num_vars(), 1)
            self.assertEqual(q_p.get_num_linear_constraints(), 0)
            self.assertEqual(q_p.get_num_quadratic_constraints(), 0)
            self.assertAlmostEqual(q_p.objective.constant, 1)
            np.testing.assert_allclose(q_p.objective.linear.to_array(), [0])
            np.testing.assert_allclose(q_p.objective.quadratic.to_array(), [[-2]])

            with self.assert_deprecation_warning(opflow):
                op, offset = to_ising(q_p, opflow=opflow)
            np.testing.assert_allclose(op.to_matrix(), operator.to_matrix())
            self.assertAlmostEqual(offset, 0)

        with self.subTest("Z, linear=True"):
            with self.assert_deprecation_warning(opflow):
                operator = Z if opflow else Pauli("Z")
                q_p = from_ising(operator, linear=True)
            self.assertEqual(q_p.get_num_vars(), 1)
            self.assertEqual(q_p.get_num_linear_constraints(), 0)
            self.assertEqual(q_p.get_num_quadratic_constraints(), 0)
            self.assertAlmostEqual(q_p.objective.constant, 1)
            np.testing.assert_allclose(q_p.objective.linear.to_array(), [-2])
            np.testing.assert_allclose(q_p.objective.quadratic.to_array(), [[0]])

            with self.assert_deprecation_warning(opflow):
                op, offset = to_ising(q_p, opflow=opflow)
            np.testing.assert_allclose(op.to_matrix(), operator.to_matrix())
            self.assertAlmostEqual(offset, 0)

        with self.subTest("3 * II, linear=False"):
            with self.assert_deprecation_warning(opflow):
                operator = 3 * I ^ I if opflow else SparsePauliOp("II", 3)
                q_p = from_ising(operator, linear=False)
            self.assertEqual(q_p.get_num_vars(), 2)
            self.assertEqual(q_p.get_num_linear_constraints(), 0)
            self.assertEqual(q_p.get_num_quadratic_constraints(), 0)
            self.assertAlmostEqual(q_p.objective.constant, 3)
            np.testing.assert_allclose(q_p.objective.linear.to_array(), [0, 0])
            np.testing.assert_allclose(q_p.objective.quadratic.to_array(), [[0, 0], [0, 0]])

            with self.assert_deprecation_warning(opflow):
                op, offset = to_ising(q_p, opflow=opflow)
            np.testing.assert_allclose(op.to_matrix(), np.zeros((4, 4)))
            self.assertAlmostEqual(offset, 3)

        with self.subTest("3 * II, linear=True"):
            with self.assert_deprecation_warning(opflow):
                operator = 3 * I ^ I if opflow else SparsePauliOp("II", 3)
                q_p = from_ising(operator, linear=True)
            self.assertEqual(q_p.get_num_vars(), 2)
            self.assertEqual(q_p.get_num_linear_constraints(), 0)
            self.assertEqual(q_p.get_num_quadratic_constraints(), 0)
            self.assertAlmostEqual(q_p.objective.constant, 3)
            np.testing.assert_allclose(q_p.objective.linear.to_array(), [0, 0])
            np.testing.assert_allclose(q_p.objective.quadratic.to_array(), [[0, 0], [0, 0]])

            with self.assert_deprecation_warning(opflow):
                op, offset = to_ising(q_p, opflow=opflow)
            np.testing.assert_allclose(op.to_matrix(), np.zeros((4, 4)))
            self.assertAlmostEqual(offset, 3)

        with self.subTest("IZ, linear=False"):
            with self.assert_deprecation_warning(opflow):
                operator = I ^ Z if opflow else Pauli("IZ")
                q_p = from_ising(operator, linear=False)
            self.assertEqual(q_p.get_num_vars(), 2)
            self.assertEqual(q_p.get_num_linear_constraints(), 0)
            self.assertEqual(q_p.get_num_quadratic_constraints(), 0)
            self.assertAlmostEqual(q_p.objective.constant, 1)
            np.testing.assert_allclose(q_p.objective.linear.to_array(), [0, 0])
            np.testing.assert_allclose(q_p.objective.quadratic.to_array(), [[-2, 0], [0, 0]])

            with self.assert_deprecation_warning(opflow):
                op, offset = to_ising(q_p, opflow=opflow)
            np.testing.assert_allclose(op.to_matrix(), operator.to_matrix())
            self.assertAlmostEqual(offset, 0)

        with self.subTest("IZ, linear=True"):
            with self.assert_deprecation_warning(opflow):
                operator = I ^ Z if opflow else Pauli("IZ")
                q_p = from_ising(operator, linear=True)
            self.assertEqual(q_p.get_num_vars(), 2)
            self.assertEqual(q_p.get_num_linear_constraints(), 0)
            self.assertEqual(q_p.get_num_quadratic_constraints(), 0)
            self.assertAlmostEqual(q_p.objective.constant, 1)
            np.testing.assert_allclose(q_p.objective.linear.to_array(), [-2, 0])
            np.testing.assert_allclose(q_p.objective.quadratic.to_array(), [[0, 0], [0, 0]])

            with self.assert_deprecation_warning(opflow):
                op, offset = to_ising(q_p, opflow=opflow)
            np.testing.assert_allclose(op.to_matrix(), operator.to_matrix())
            self.assertAlmostEqual(offset, 0)

    @data(True, False)
    def test_to_ising_wo_variable(self, opflow):
        """test to_ising with problems without variables"""
        with self.subTest("empty problem"):
            q_p = QuadraticProgram()
            with self.assert_deprecation_warning(opflow):
                op, offset = to_ising(q_p, opflow)
            np.testing.assert_allclose(op.to_matrix(), np.zeros((2, 2)))
            self.assertAlmostEqual(offset, 0)

        with self.subTest("min 3"):
            q_p = QuadraticProgram()
            q_p.minimize(constant=3)
            with self.assert_deprecation_warning(opflow):
                op, offset = to_ising(q_p, opflow)
            np.testing.assert_allclose(op.to_matrix(), np.zeros((2, 2)))
            self.assertAlmostEqual(offset, 3)

        with self.subTest("max -1"):
            q_p = QuadraticProgram()
            q_p.maximize(constant=-1)
            with self.assert_deprecation_warning(opflow):
                op, offset = to_ising(q_p, opflow)
            np.testing.assert_allclose(op.to_matrix(), np.zeros((2, 2)))
            self.assertAlmostEqual(offset, 1)

    def test_warning(self):
        """Test warning message"""
        q_p = QuadraticProgram()
        with self.assertWarns((UserWarning, DeprecationWarning)):
            _ = to_ising(q_p)

        with self.assertWarns(DeprecationWarning):
            op = PauliSumOp.from_list([("Z", 1)])
        with self.assertWarns(DeprecationWarning):
            _ = from_ising(op)
