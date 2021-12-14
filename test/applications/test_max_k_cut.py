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

""" Test Maxkcut class"""

from test.optimization_test_case import QiskitOptimizationTestCase

import networkx as nx

from qiskit.exceptions import MissingOptionalLibraryError
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import OptimizationResult, OptimizationResultStatus
from qiskit_optimization.applications.max_k_cut import Maxkcut
from qiskit_optimization.problems import QuadraticObjective, VarType

try:
    import matplotlib as _

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


class TestMaxkcut(QiskitOptimizationTestCase):
    """Test Maxkcut class"""

    def setUp(self):
        super().setUp()
        self.graph = nx.gnm_random_graph(4, 5, 123)
        self.k = 3
        op = QuadraticProgram()
        for _ in range(12):
            op.binary_var()
        self.result = OptimizationResult(
            x=[0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
            fval=0,
            variables=op.variables,
            status=OptimizationResultStatus.SUCCESS,
        )

    def test_to_quadratic_program(self):
        """Test to_quadratic_program"""
        maxkcut = Maxkcut(self.graph, self.k)
        op = maxkcut.to_quadratic_program()
        # Test name
        self.assertEqual(op.name, "Max-k-cut")
        # Test variables
        self.assertEqual(op.get_num_vars(), 12)
        for var in op.variables:
            self.assertEqual(var.vartype, VarType.BINARY)
        # Test objective
        obj = op.objective
        self.assertEqual(obj.sense, QuadraticObjective.Sense.MINIMIZE)
        self.assertEqual(obj.constant, 4)
        self.assertDictEqual(
            obj.linear.to_dict(),
            {
                0: -2.0,
                1: -2.0,
                2: -2.0,
                3: -2.0,
                4: -2.0,
                5: -2.0,
                6: -2.0,
                7: -2.0,
                8: -2.0,
                9: -2.0,
                10: -2.0,
                11: -2.0,
            },
        )
        self.assertDictEqual(
            obj.quadratic.to_dict(),
            {
                (0, 0): 1.0,
                (0, 1): 2.0,
                (1, 1): 1.0,
                (0, 2): 2.0,
                (1, 2): 2.0,
                (2, 2): 1.0,
                (0, 3): 1.0,
                (3, 3): 1.0,
                (1, 4): 1.0,
                (3, 4): 2.0,
                (4, 4): 1.0,
                (2, 5): 1.0,
                (3, 5): 2.0,
                (4, 5): 2.0,
                (5, 5): 1.0,
                (0, 6): 1.0,
                (3, 6): 1.0,
                (6, 6): 1.0,
                (1, 7): 1.0,
                (4, 7): 1.0,
                (6, 7): 2.0,
                (7, 7): 1.0,
                (2, 8): 1.0,
                (5, 8): 1.0,
                (6, 8): 2.0,
                (7, 8): 2.0,
                (8, 8): 1.0,
                (0, 9): 1.0,
                (6, 9): 1.0,
                (9, 9): 1.0,
                (1, 10): 1.0,
                (7, 10): 1.0,
                (9, 10): 2.0,
                (10, 10): 1.0,
                (2, 11): 1.0,
                (8, 11): 1.0,
                (9, 11): 2.0,
                (10, 11): 2.0,
                (11, 11): 1.0,
            },
        )
        # Test constraint
        lin = op.linear_constraints
        self.assertEqual(len(lin), 0)

    def test_interpret(self):
        """Test interpret"""
        maxkcut = Maxkcut(self.graph, self.k)
        self.assertEqual(maxkcut.interpret(self.result), [[1, 3], [0], [2]])

    def test_node_color(self):
        """Test _node_color"""
        # default colors
        if _HAS_MATPLOTLIB:
            maxkcut = Maxkcut(self.graph, self.k)
            self.assertEqual(
                [[round(num, 2) for num in i] for i in maxkcut._node_color(self.result.x)],
                [
                    [0.5, 1.0, 0.7, 1.0],
                    [0.5, 0.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0, 1.0],
                    [0.5, 0.0, 1.0, 1.0],
                ],
            )
        # given colors
        maxkcut = Maxkcut(self.graph, self.k, colors=["r", "g", "b"])
        self.assertEqual(
            [[round(num, 2) for num in i] for i in maxkcut._node_color(self.result.x)],
            [
                [0.0, 0.5, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
            ],
        )

    def test_draw(self):
        """Test whether draw raises an error if matplotlib is not installed"""
        maxkcut = Maxkcut(self.graph, self.k)
        try:
            import matplotlib as _

            maxkcut.draw()

        except ImportError:
            with self.assertRaises(MissingOptionalLibraryError):
                maxkcut.draw()
