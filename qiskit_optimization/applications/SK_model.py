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

from typing import Optional, Union, List

import networkx as nx
import numpy as np
from docplex.mp.model import Model

from .graph_optimization_application import OptimizationApplication
from ..algorithms import OptimizationResult
from ..problems.quadratic_program import QuadraticProgram
from ..translators import from_docplex_mp


class SKModel(OptimizationApplication):
    """Optimization application for the "Sherrington Kirkpatrick (SK) model" [1].

    The SK Hamiltonian over n spins is given as:
        H(x)=-1/\sqrt{n} \sum_{i<j} w_{i,j}x_ix_j, where x_i\in\pm 1
    and w_{i,j}\in\pm 1 are chosen independently and uniformly at random.
    Notice that there are other variants e.g., with w_{i,j} chosen from the normal distribution
    with mean 0 and variance 1.

    References:
        [1]: Dmitry Panchenko. "The Sherrington-Kirkpatrick model: an overview",
        https://arxiv.org/abs/1211.1094
    """

    def __init__(self, n: int, rng: Optional[np.random.RandomState] = np.random):
        """
        Constructor for the SK model.

        Args:
            n: number of sites
            rng: numpy pseudo-random number generator
        """
        self._rng = rng
        self._n = n
        self._graph = nx.complete_graph(self._n)

        self.new_instance()

    def new_instance(self) -> None:
        """Generate a new instance of the SK model."""
        for _, _, d in self._graph.edges(data=True):
            d['weight'] = self._rng.choice([-1, 1])

    def to_quadratic_program(self) -> QuadraticProgram:
        """Convert an SK model problem instance into a
        :class:`~qiskit_optimization.problems.QuadraticProgram`.

        Returns:
            The :class:`~qiskit_optimization.problems.QuadraticProgram` created
            from the SK problem instance.
        """
        mdl = Model(name="SK-model")
        x = {
            i: mdl.binary_var(name="x_{0}".format(i)) for i in range(self._graph.number_of_nodes())
        }

        objective = mdl.sum(
            -1 / np.sqrt(self._n) * self._graph.edges[i, j]["weight"] * (2 * x[i] - 1) * (2 * x[j] - 1)
            for i, j in self._graph.edges
        )
        # we converted the standard H(x)=-1/\sqrt{n} \sum w_{ij}x_ix_j, where x_i\in\pm 1 to binary.

        mdl.minimize(objective)
        return from_docplex_mp(mdl)

    def interpret(self, result: Union[OptimizationResult, np.ndarray]) -> List[List[int]]:
        """Interpret a result as energy and configuration of spins

        Args:
            result : The calculated result of the problem

        Returns:
            [energy, configuration of spins]
        """
        configuration = list(map(lambda x: 2 * x - 1, self._result_to_x(result)))
        return [result.fval, configuration]

    @property
    def graph(self) -> nx.Graph:
        """Getter of the graph
        Returns:
            A graph for a problem
        """
        return self._graph

    def _set_rng(self, rng):
        self._rng = rng

    rng = property(fset=_set_rng)
