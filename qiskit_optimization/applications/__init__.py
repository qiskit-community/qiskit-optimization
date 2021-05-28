# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Optimization applications (:mod:`qiskit_optimization.applications`)
===================================================================

.. currentmodule:: qiskit_optimization.applications

Applications for common optimization problems.

Base classes for applications
=======================================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   OptimizationApplication
   GraphOptimizationApplication

Applications
======================
.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Clique
   ExactCover
   GraphPartition
   Knapsack
   Maxcut
   NumberPartition
   SetPacking
   StableSet
   Tsp
   VehicleRouting
   VertexCover
"""

from .clique import Clique
from .exact_cover import ExactCover
from .graph_optimization_application import GraphOptimizationApplication
from .graph_partition import GraphPartition
from .knapsack import Knapsack
from .max_cut import Maxcut
from .number_partition import NumberPartition
from .optimization_application import OptimizationApplication
from .set_packing import SetPacking
from .stable_set import StableSet
from .tsp import Tsp
from .vehicle_routing import VehicleRouting
from .vertex_cover import VertexCover

_all__ = [
    "Clique",
    "ExactCover",
    "GraphOptimizationApplication",
    "Knapsack",
    "Maxcut",
    "NumberPartition",
    "OptimizationApplication",
    "SetPacking",
    "StableSet",
    "Tsp",
    "VehicleRouting",
    "VertexCover",
]
