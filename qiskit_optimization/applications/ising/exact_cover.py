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


class ExactCover(BaseApplication):

    def __init__(self, subsets):
        self._subsets = copy.deepcopy(subsets)
        self._set = []
        for sub in self._subsets:
            self._set.extend(sub)
        self._set = np.unique(self._set)

    def to_quadratic_program(self):
        mdl = Model(name='exact_cover')
        x = {i: mdl.binary_var(name='x_{0}'.format(i)) for i in range(len(self._subsets))}
        mdl.minimize(mdl.sum(x[i] for i in x))
        for e in self._set:
            mdl.add_constraint(mdl.sum(x[i] for i, sub in enumerate(self._subsets)
                                       if e in sub) == 1)
        qp = QuadraticProgram()
        qp.from_docplex(mdl)
        return qp

    def interpret(self, result):
        sub = []
        for i, value in enumerate(result.x):
            if value:
                sub.append(self._subsets[i])
        return sub
