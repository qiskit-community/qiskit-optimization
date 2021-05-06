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

""" Test GraphPartinioning class"""
from test.optimization_test_case import QiskitOptimizationTestCase

import networkx as nx

from qiskit.exceptions import MissingOptionalLibraryError
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import OptimizationResult, OptimizationResultStatus
from qiskit_optimization.applications.max_cut import Maxcut
from qiskit_optimization.problems import QuadraticObjective, VarType


class TestMaxcut(QiskitOptimizationTestCase):
    """Test Maxcut class"""

    def setUp(self):
        super().setUp()
        self.graph = nx.gnm_random_graph(4, 6, 123)
        op = QuadraticProgram()
        for _ in range(4):
            op.binary_var()
        self.result = OptimizationResult(
            x=[1, 1, 0, 0],
            fval=4,
            variables=op.variables,
            status=OptimizationResultStatus.SUCCESS,
        )

    def test_to_quadratic_program(self):
        """Test to_quadratic_program"""
        maxcut = Maxcut(self.graph)
        op = maxcut.to_quadratic_program()
        # Test name
        self.assertEqual(op.name, "Max-cut")
        # Test variables
        self.assertEqual(op.get_num_vars(), 4)
        for var in op.variables:
            self.assertEqual(var.vartype, VarType.BINARY)
        # Test objective
        obj = op.objective
        self.assertEqual(obj.sense, QuadraticObjective.Sense.MAXIMIZE)
        self.assertEqual(obj.constant, 0)
        self.assertDictEqual(obj.linear.to_dict(), {0: 3.0, 1: 3.0, 2: 3.0, 3: 3.0})
        self.assertDictEqual(
            obj.quadratic.to_dict(),
            {
                (0, 1): -2.0,
                (0, 2): -2.0,
                (1, 2): -2.0,
                (0, 3): -2.0,
                (1, 3): -2.0,
                (2, 3): -2.0,
            },
        )
        # Test constraint
        lin = op.linear_constraints
        self.assertEqual(len(lin), 0)

    def test_interpret(self):
        """Test interpret"""
        maxcut = Maxcut(self.graph)
        self.assertEqual(maxcut.interpret(self.result), [[2, 3], [0, 1]])

    def test_node_color(self):
        """Test _node_color"""
        maxcut = Maxcut(self.graph)
        self.assertEqual(maxcut._node_color(self.result), ["b", "b", "r", "r"])

    def test_draw(self):
        """Test whether draw raises an error if matplotlib is not installed"""
        maxcut = Maxcut(self.graph)
        try:
            import matplotlib as _

            maxcut.draw()

        except ImportError:
            with self.assertRaises(MissingOptionalLibraryError):
                maxcut.draw()
