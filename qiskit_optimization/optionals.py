# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Additional optional constants.
"""

from qiskit.utils import LazyImportTester

HAS_CPLEX = LazyImportTester(
    {
        "cplex": ("Cplex",),
    },
    name="CPLEX",
    install="pip install 'qiskit-optimization[cplex]'",
)

HAS_CVXPY = LazyImportTester(
    {
        "cvxpy": ("DCPError", "DGPError", "SolverError"),
    },
    name="CVXPY",
    install="pip install 'qiskit-optimization[cvx]'",
)

HAS_GUROBIPY = LazyImportTester(
    {
        "gurobipy": ("Model",),
    },
    name="Gurobi",
    install="pip install 'qiskit-optimization[gurobi]'",
)

HAS_MATPLOTLIB = LazyImportTester(
    {
        "matplotlib.pyplot": ("Figure",),
    },
    name="Matplotlib",
    install="pip install 'qiskit-optimization[matplotlib]'",
)
