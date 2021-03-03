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

""" Test NumberPartitioning class"""

import random

import networkx as nx

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import (OptimizationResult,
                                            OptimizationResultStatus)
from qiskit_optimization.applications.ising.number_partitioning import NumberPartitioning
from qiskit_optimization.problems import (Constraint, QuadraticObjective, VarType)
from test.optimization_test_case import QiskitOptimizationTestCase


class TestNumberPartitioning(QiskitOptimizationTestCase):
    """ Test NumberPartitioning class"""

    def setUp(self):
        super().setUp()
        self.num_set = [8, 7, 6, 5, 4]
        qp = QuadraticProgram()
        for i in range(5):
            qp.binary_var()
        self.result = OptimizationResult(
            x=[1, 1, 0, 0, 0], fval=0, variables=qp.variables,
            status=OptimizationResultStatus.SUCCESS)

    def test_to_quadratic_program(self):
        """Test to_quadratic_program"""
        number_partitioning = NumberPartitioning(self.num_set)
        qp = number_partitioning.to_quadratic_program()
        # Test name
        self.assertEqual(qp.name, "Number partitioning")
        # Test variables
        self.assertEqual(qp.get_num_vars(), 5)
        for var in qp.variables:
            self.assertEqual(var.vartype, VarType.BINARY)
        # Test objective
        obj = qp.objective
        self.assertEqual(obj.sense, QuadraticObjective.Sense.MINIMIZE)
        self.assertEqual(obj.constant, 0)
        self.assertDictEqual(obj.linear.to_dict(), {})
        self.assertEqual(obj.quadratic.to_dict(), {})
        # Test constraint
        lin = qp.linear_constraints
        self.assertEqual(len(lin), 1)
        self.assertEqual(lin[0].sense, Constraint.Sense.EQ)
        self.assertEqual(lin[0].rhs, -30)
        self.assertEqual(lin[0].linear.to_dict(), {
                             i: -2*num for i, num in enumerate(self.num_set)
                             })

    def test_interpret(self):
        """Test interpret"""
        number_partitioning = NumberPartitioning(self.num_set)
        self.assertEqual(number_partitioning.interpret(self.result), [[6, 5, 4], [8, 7]])
