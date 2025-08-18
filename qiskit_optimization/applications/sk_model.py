# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An application class for the Sherrington Kirkpatrick (SK) model."""
from __future__ import annotations


import networkx as nx
import numpy as np
from docplex.mp.model import Model

from .graph_optimization_application import OptimizationApplication
from ..algorithms import OptimizationResult
from ..problems.quadratic_program import QuadraticProgram
from ..translators import from_docplex_mp


class SKModel(OptimizationApplication):
    r"""Optimization application of the "Sherrington Kirkpatrick (SK) model" [1].

    The SK Hamiltonian over n spins is given as:
    :math:`H(x)=-1/\sqrt{n} \sum_{i<j} w_{i,j}x_ix_j`,
    where :math:`x_i\in\{\pm 1\}` is the configuration of spins and
    :math:`w_{i,j}\in\{\pm 1\}` is a disorder chosen independently and uniformly at random.
    Notice that there are other variants of disorders e.g., with :math:`w_{i,j}` chosen from
    the normal distribution with mean 0 and variance 1.

    References:
        [1]: Dmitry Panchenko. "The Sherrington-Kirkpatrick model: an overview",
        https://arxiv.org/abs/1211.1094
    """

    def __init__(self, num_sites: int, rng_or_seed: np.random.Generator | int | None = None):
        """
        Args:
            num_sites: number of sites
            rng_or_seed: NumPy pseudo-random number generator or seed for np.random.default_rng(<seed>)
                or None. None results in usage of np.random.default_rng().
        """
        if isinstance(rng_or_seed, np.random.Generator):
            self._rng = rng_or_seed
        else:
            self._rng = np.random.default_rng(rng_or_seed)

        self._num_sites = num_sites
        self._graph = nx.complete_graph(self._num_sites)

        self.disorder()

    def disorder(self) -> None:
        """Generate a new disorder of the SK model."""
        for _, _, edge_data in self._graph.edges(data=True):
            edge_data["weight"] = self._rng.choice([-1, 1])

    def to_quadratic_program(self) -> QuadraticProgram:
        """Convert an SK model problem instance into a
        :class:`~qiskit_optimization.problems.QuadraticProgram`.

        Returns:
            The :class:`~qiskit_optimization.problems.QuadraticProgram` created
            from the SK problem instance.
        """
        mdl = Model(name="SK-model")
        x = mdl.binary_var_list(self._graph.number_of_nodes())

        objective = mdl.sum(
            -1
            / np.sqrt(self._num_sites)
            * self._graph.edges[i, j]["weight"]
            * (2 * x[i] - 1)
            * (2 * x[j] - 1)
            for i, j in self._graph.edges
        )
        # we converted the standard H(x)=-1/\sqrt{n} \sum w_{ij}x_ix_j, where x_i\in\pm 1 to binary.

        mdl.minimize(objective)
        return from_docplex_mp(mdl)

    def interpret(self, result: OptimizationResult | np.ndarray) -> list[int]:
        """Interpret a result as configuration of spins.

        Args:
            result : The calculated result of the problem.

        Returns:
            configuration of spins
        """
        configuration = [2 * x - 1 for x in self._result_to_x(result)]
        return configuration

    @property
    def graph(self) -> nx.Graph:
        """Getter of the graph representation.
        Returns:
            A graph for a problem.
        """
        return self._graph

    @property
    def num_sites(self) -> int:
        """Getter of the number of sites.
        Returns:
            Number of sites.
        """
        return self._num_sites
