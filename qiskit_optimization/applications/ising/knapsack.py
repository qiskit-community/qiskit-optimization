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


class Knapsack(BaseApplication):

    def __init__(self, values, weights, max_weight):
        self._values = copy.deepcopy(values)
        self._weights = copy.deepcopy(weights)
        self._max_weight = copy.deepcopy(max_weight)

    def to_quadratic_program(self):
        mdl = Model(name='K napsack')
        x = {i: mdl.binary_var(name='x_{0}'.format(i)) for i in range(len(self._values))}
        mdl.maximize(mdl.sum(self._values[i]*x[i] for i in x))
        mdl.add_constraint(mdl.sum(self._weights[i] * x[i] for i in x) <= self._max_weight)
        qp = QuadraticProgram()
        qp.from_docplex(mdl)
        return qp

    def interpret(self, result):
        return [i for i, value in enumerate(result.x) if value]
