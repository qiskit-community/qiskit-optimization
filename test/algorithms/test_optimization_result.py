# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test OptimizationResult """

from test.optimization_test_case import QiskitOptimizationTestCase

import numpy as np

from qiskit_optimization.algorithms import OptimizationResult, OptimizationResultStatus
from qiskit_optimization.problems import QuadraticProgram


class TestOptimizationResult(QiskitOptimizationTestCase):
    """Test OptimizationResult"""

    def test_init(self):
        """test init"""
        q_p = QuadraticProgram()
        q_p.integer_var_list(3)
        result = OptimizationResult(
            x=[1, 2, 3], fval=10, variables=q_p.variables, status=OptimizationResultStatus.SUCCESS
        )
        np.testing.assert_allclose(result.x, [1, 2, 3])
        self.assertAlmostEqual(result.fval, 10)
        self.assertEqual(result.status, OptimizationResultStatus.SUCCESS)
        self.assertIsNone(result.raw_results)
        self.assertEqual(len(result.samples), 1)
        sample = result.samples[0]
        np.testing.assert_allclose(sample.x, [1, 2, 3])
        self.assertAlmostEqual(sample.fval, 10)
        self.assertEqual(sample.status, OptimizationResultStatus.SUCCESS)
        self.assertAlmostEqual(sample.probability, 1)

    def test_str_repr(self):
        """test str and repr"""

        with self.subTest("success"):
            q_p = QuadraticProgram()
            q_p.integer_var_list(3)
            result = OptimizationResult(
                x=[1.0, 2.0, 3.0],
                fval=10.1,
                variables=q_p.variables,
                status=OptimizationResultStatus.SUCCESS,
            )
            expected = "fval=10.1, x0=1.0, x1=2.0, x2=3.0, status=SUCCESS"
            self.assertEqual(str(result), expected)
            self.assertEqual(repr(result), f"<OptimizationResult: {expected}>")

        with self.subTest("failure"):
            q_p = QuadraticProgram()
            q_p.integer_var_list(3)
            result = OptimizationResult(
                x=[-1.0, 2.0, -3.0],
                fval=10.0,
                variables=q_p.variables,
                status=OptimizationResultStatus.FAILURE,
            )
            expected = "fval=10.0, x0=-1.0, x1=2.0, x2=-3.0, status=FAILURE"
            self.assertEqual(str(result), expected)
            self.assertEqual(repr(result), f"<OptimizationResult: {expected}>")

        with self.subTest("infeasible"):
            q_p = QuadraticProgram()
            q_p.integer_var_list(3, name="y", key_format="_{}")
            result = OptimizationResult(
                x=[1.0, 2.0, -3.0],
                fval=11.0,
                variables=q_p.variables,
                status=OptimizationResultStatus.INFEASIBLE,
            )
            expected = "fval=11.0, y_0=1.0, y_1=2.0, y_2=-3.0, status=INFEASIBLE"
            self.assertEqual(str(result), expected)
            self.assertEqual(repr(result), f"<OptimizationResult: {expected}>")

    def test_prettyprint(self):
        """test prettyprint"""

        with self.subTest("success"):
            q_p = QuadraticProgram()
            q_p.integer_var_list(3)
            result = OptimizationResult(
                x=[1.0, 2.0, 3.0],
                fval=10.1,
                variables=q_p.variables,
                status=OptimizationResultStatus.SUCCESS,
            )
            expected = "\n".join(
                [
                    "objective function value: 10.1",
                    "variable values: x0=1.0, x1=2.0, x2=3.0",
                    "status: SUCCESS",
                ]
            )
            self.assertEqual(result.prettyprint(), expected)

        with self.subTest("failure"):
            q_p = QuadraticProgram()
            q_p.integer_var_list(3)
            result = OptimizationResult(
                x=[-1.0, 2.0, -3.0],
                fval=10.0,
                variables=q_p.variables,
                status=OptimizationResultStatus.FAILURE,
            )
            expected = "\n".join(
                [
                    "objective function value: 10.0",
                    "variable values: x0=-1.0, x1=2.0, x2=-3.0",
                    "status: FAILURE",
                ]
            )
            self.assertEqual(result.prettyprint(), expected)

        with self.subTest("infeasible"):
            q_p = QuadraticProgram()
            q_p.integer_var_list(3, name="y", key_format="_{}")
            result = OptimizationResult(
                x=[1.0, 2.0, -3.0],
                fval=11.0,
                variables=q_p.variables,
                status=OptimizationResultStatus.INFEASIBLE,
            )
            expected = "\n".join(
                [
                    "objective function value: 11.0",
                    "variable values: y_0=1.0, y_1=2.0, y_2=-3.0",
                    "status: INFEASIBLE",
                ]
            )
            self.assertEqual(result.prettyprint(), expected)
