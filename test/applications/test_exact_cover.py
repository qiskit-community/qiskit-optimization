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

""" Test ExactCover class"""
from test.optimization_test_case import QiskitOptimizationTestCase

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import OptimizationResult, OptimizationResultStatus
from qiskit_optimization.applications.exact_cover import ExactCover
from qiskit_optimization.problems import Constraint, QuadraticObjective, VarType


class TestExactCover(QiskitOptimizationTestCase):
    """Test ExactCover class"""

    def setUp(self):
        """Set up for the tests"""
        super().setUp()
        self.total_set = [1, 2, 3, 4, 5]
        self.list_of_subsets = [[1, 4], [2, 3, 4], [1, 5], [2, 3]]
        op = QuadraticProgram()
        for _ in range(4):
            op.binary_var()
        self.result = OptimizationResult(
            x=[0, 1, 1, 0],
            fval=2,
            variables=op.variables,
            status=OptimizationResultStatus.SUCCESS,
        )

    def test_to_quadratic_program(self):
        """Test to_quadratic_program"""
        exact_cover = ExactCover(self.list_of_subsets)
        op = exact_cover.to_quadratic_program()
        # Test name
        self.assertEqual(op.name, "Exact cover")
        # Test variables
        self.assertEqual(op.get_num_vars(), 4)
        for var in op.variables:
            self.assertEqual(var.vartype, VarType.BINARY)
        # Test objective
        obj = op.objective
        self.assertEqual(obj.sense, QuadraticObjective.Sense.MINIMIZE)
        self.assertEqual(obj.constant, 0)
        self.assertDictEqual(obj.linear.to_dict(), {0: 1, 1: 1, 2: 1, 3: 1})
        self.assertEqual(obj.quadratic.to_dict(), {})
        # Test constraint
        lin = op.linear_constraints
        self.assertEqual(len(lin), len(self.total_set))
        for i, lin in enumerate(lin):
            self.assertEqual(lin.sense, Constraint.Sense.EQ)
            self.assertEqual(lin.rhs, 1)
            self.assertEqual(
                lin.linear.to_dict(),
                {j: 1 for j, subset in enumerate(self.list_of_subsets) if i + 1 in subset},
            )

    def test_interpret(self):
        """Test interpret"""
        exact_cover = ExactCover(self.list_of_subsets)
        self.assertEqual(exact_cover.interpret(self.result), [[2, 3, 4], [1, 5]])
