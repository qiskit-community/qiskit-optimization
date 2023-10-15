# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Optimization algorithms (:mod:`qiskit_optimization.algorithms`)
===============================================================

Algorithms for optimization problems.

.. currentmodule:: qiskit_optimization.algorithms

Base classes for algorithms and results
---------------------------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   OptimizationAlgorithm
   MultiStartOptimizer
   OptimizationResult
   BaseAggregator

Algorithms and results
----------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ADMMOptimizationResult
   ADMMOptimizer
   ADMMParameters
   ADMMState
   CobylaOptimizer
   CplexOptimizer
   GoemansWilliamsonOptimizer
   GoemansWilliamsonOptimizationResult
   GroverOptimizationResult
   GroverOptimizer
   GurobiOptimizer
   IntermediateResult
   MeanAggregator
   MinimumEigenOptimizationResult
   MinimumEigenOptimizer
   OptimizationResultStatus
   RecursiveMinimumEigenOptimizationResult
   RecursiveMinimumEigenOptimizer
   ScipyMilpOptimizer
   SlsqpOptimizationResult
   SlsqpOptimizer
   SolutionSample
   WarmStartQAOAOptimizer
   WarmStartQAOAFactory

Submodules
----------

.. autosummary::
   :toctree:

   qrao

"""

from .admm_optimizer import (
    ADMMOptimizer,
    ADMMOptimizationResult,
    ADMMState,
    ADMMParameters,
)
from .cobyla_optimizer import CobylaOptimizer
from .cplex_optimizer import CplexOptimizer
from .goemans_williamson_optimizer import (
    GoemansWilliamsonOptimizer,
    GoemansWilliamsonOptimizationResult,
)
from .grover_optimizer import GroverOptimizer, GroverOptimizationResult
from .gurobi_optimizer import GurobiOptimizer
from .minimum_eigen_optimizer import (
    MinimumEigenOptimizer,
    MinimumEigenOptimizationResult,
)
from .multistart_optimizer import MultiStartOptimizer
from .optimization_algorithm import (
    OptimizationAlgorithm,
    OptimizationResult,
    OptimizationResultStatus,
    SolutionSample,
)
from .recursive_minimum_eigen_optimizer import (
    RecursiveMinimumEigenOptimizer,
    RecursiveMinimumEigenOptimizationResult,
    IntermediateResult,
)
from .scipy_milp_optimizer import ScipyMilpOptimizer
from .slsqp_optimizer import SlsqpOptimizer, SlsqpOptimizationResult
from .warm_start_qaoa_optimizer import (
    BaseAggregator,
    MeanAggregator,
    WarmStartQAOAFactory,
    WarmStartQAOAOptimizer,
)

__all__ = [
    "ADMMOptimizer",
    "ADMMOptimizationResult",
    "ADMMState",
    "ADMMParameters",
    "OptimizationAlgorithm",
    "OptimizationResult",
    "OptimizationResultStatus",
    "BaseAggregator",
    "CplexOptimizer",
    "CobylaOptimizer",
    "GoemansWilliamsonOptimizer",
    "GoemansWilliamsonOptimizationResult",
    "GroverOptimizer",
    "GroverOptimizationResult",
    "GurobiOptimizer",
    "MeanAggregator",
    "MinimumEigenOptimizer",
    "MinimumEigenOptimizationResult",
    "MultiStartOptimizer",
    "RecursiveMinimumEigenOptimizer",
    "RecursiveMinimumEigenOptimizationResult",
    "IntermediateResult",
    "ScipyMilpOptimizer",
    "SlsqpOptimizer",
    "SlsqpOptimizationResult",
    "SolutionSample",
    "WarmStartQAOAOptimizer",
    "WarmStartQAOAFactory",
]
