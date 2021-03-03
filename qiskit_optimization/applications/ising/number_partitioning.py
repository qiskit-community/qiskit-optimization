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

"""
Convert vertex cover instances into Pauli list
Deal with Gset format. See https://web.stanford.edu/~yyye/yyye/Gset/
"""
import copy

import networkx as nx
import numpy as np
from docplex.mp.model import Model

from .base_application import BaseApplication
from qiskit_optimization.problems.quadratic_program import QuadraticProgram


class NumberPartitioning(BaseApplication):

    def __init__(self, num_set):
        self._num_set = copy.deepcopy(num_set)

    def to_quadratic_program(self):
        mdl = Model(name='Number partitioning')
        x = {i: mdl.binary_var(name='x_{0}'.format(i)) for i in range(len(self._num_set))}
        mdl.add_constraint(mdl.sum(num * (-2 * x[i] + 1)
                                   for i, num in enumerate(self._num_set)) == 0)
        qp = QuadraticProgram()
        qp.from_docplex(mdl)
        return qp

    def interpret(self, result):
        num_subsets = [[], []]
        for i, value in enumerate(result.x):
            if value == 0:
                num_subsets[0].append(self._num_set[i])
            else:
                num_subsets[1].append(self._num_set[i])
        return num_subsets
