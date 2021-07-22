# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test SK model class"""
import numpy as np

from test.optimization_test_case import QiskitOptimizationTestCase

import networkx as nx

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import OptimizationResult, OptimizationResultStatus
from qiskit_optimization.applications.SK_model import SKModel
from qiskit_optimization.problems import QuadraticObjective, VarType


class TestSKModel(QiskitOptimizationTestCase):
    """Test SK model class"""

    def setUp(self):
        super().setUp()
        self._n = 2
        self._seed = 0
        self._graph = nx.convert_matrix.from_numpy_matrix(np.array([[0.0, -1.0], [-1.0, 0.0]]))

        op = QuadraticProgram()
        for _ in range(2):
            op.binary_var()

        self._result = OptimizationResult(
            x=[0, 0],
            fval=-1 / np.sqrt(2),
            variables=op.variables,
            status=OptimizationResultStatus.SUCCESS,
        )

    def test_to_quadratic_program(self):
        """Test to_quadratic_program"""
        problem = SKModel(self._n, np.random.RandomState(self._seed))
        qp = problem.to_quadratic_program()

        # Test name
        with self.subTest("Test name"):
            self.assertEqual(qp.name, "SK-model")

        # Test variables
        with self.subTest("Test variables"):
            self.assertEqual(qp.get_num_vars(), 2)
            for var in qp.variables:
                self.assertEqual(var.vartype, VarType.BINARY)

        # Test objective
        with self.subTest("Test objective"):
            obj = qp.objective
            self.assertEqual(obj.sense, QuadraticObjective.Sense.MINIMIZE)
            self.assertAlmostEqual(obj.constant, 1 / np.sqrt(2))
            obj_lin = obj.linear.to_dict()
            self.assertAlmostEqual(obj_lin[0], -np.sqrt(2))
            self.assertAlmostEqual(obj_lin[1], -np.sqrt(2))
            obj_quad = obj.quadratic.to_dict()
            self.assertEqual(len(obj_quad), 1)
            self.assertAlmostEqual(obj_quad[(0, 1)], 2 * np.sqrt(2))

        # Test constraint
        with self.subTest("Test constraints"):
            constraints_lin = qp.linear_constraints
            self.assertEqual(len(constraints_lin), 0)
            constraints_quad = qp.quadratic_constraints
            self.assertEqual(len(constraints_quad), 0)

    def test_interpret(self):
        """Test interpret"""
        sk = SKModel(2, np.random.RandomState(self._seed))
        energy, configuration = sk.interpret(self._result)
        self.assertEqual(configuration, [-1, -1])
        self.assertAlmostEqual(energy, -1 / np.sqrt(2))
