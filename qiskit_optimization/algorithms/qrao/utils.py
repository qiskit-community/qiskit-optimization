# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility functions related to Quantum Random Access Optimization"""

from typing import Optional

import numpy as np
import networkx as nx
from docplex.mp.model import Model

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp


def get_random_maxcut_docplex_model(
    *,
    num_nodes: int = 6,
    degree: int = 3,
    seed: Optional[int] = None,
    weight: int = 1,
    draw: bool = False,
) -> Model:
    """Prepare a random DOcplex max-cut model

    Args:

        num_nodes: The number of vertices in the graph

        degree: The degree of each node in the graph

        seed: The seed to use for randomness

        weight: If `-1`, each edge will randomly have weight `-1` or `1`.
            Otherwise, each graph edge will have a random integer weight
            between `1` and `weight`, inclusive.  (It follows that if `weight
            == 1`, all edge weights are `1`).

        draw: If `True`, will call `nx.draw()` on the generated graph before
            returning.

    """
    rng = np.random.RandomState(seed)
    graph = nx.random_regular_graph(d=degree, n=num_nodes, seed=rng)
    edges = np.zeros((num_nodes, num_nodes))
    for i, j in graph.edges():
        if weight == 1:
            w = 1
        elif weight == -1:
            w = rng.choice((-1, 1))
        else:
            w = rng.randint(1, weight + 1)
        edges[i, j] = edges[j, i] = w

    mod = Model("maxcut")
    nodes = list(range(num_nodes))
    var = [mod.binary_var(name="x" + str(i)) for i in nodes]
    mod.maximize(
        mod.sum(edges[i, j] * (var[i] + var[j] - 2 * var[i] * var[j]) for i in nodes for j in nodes)
    )

    if draw:  # pragma: no cover (tested by treon)
        nx.draw(graph, with_labels=True, font_color="whitesmoke")

    return mod


def get_random_maxcut_qp(*args, **kwargs) -> QuadraticProgram:
    """Prepare a random max-cut `QuadraticProgram`, using the same arguments as
    :func:`get_random_maxcut_docplex_model`.

    """
    mod = get_random_maxcut_docplex_model(*args, **kwargs)
    return from_docplex_mp(mod)
