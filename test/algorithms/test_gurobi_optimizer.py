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

""" Test Gurobi Optimizer """

import unittest
from test.optimization_test_case import QiskitOptimizationTestCase, requires_extra_library

from ddt import data, ddt

from qiskit_optimization.algorithms import GurobiOptimizer
from qiskit_optimization.problems import QuadraticProgram


@ddt
class TestGurobiOptimizer(QiskitOptimizationTestCase):
    """Gurobi Optimizer Tests."""

    @data(
        ("op_ip1.lp", [0, 2], 6), ("op_mip1.lp", [1, 1, 0], 6), ("op_lp1.lp", [0.25, 1.75], 5.8750)
    )
    @requires_extra_library
    def test_gurobi_optimizer(self, config):
        """Gurobi Optimizer Test"""
        # unpack configuration
        gurobi_optimizer = GurobiOptimizer(disp=False)
        filename, x, fval = config

        # load optimization problem
        problem = QuadraticProgram()
        lp_file = self.get_resource_path(filename, "algorithms/resources")
        problem.read_from_lp_file(lp_file)

        # solve problem with gurobi
        result = gurobi_optimizer.solve(problem)

        # analyze results
        self.assertAlmostEqual(result.fval, fval)
        for i in range(problem.get_num_vars()):
            self.assertAlmostEqual(result.x[i], x[i])


if __name__ == "__main__":
    unittest.main()
