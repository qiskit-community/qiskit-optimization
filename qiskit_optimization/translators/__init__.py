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

"""
Quadratic program translators (:mod:`qiskit_optimization.translators`)
======================================================================

.. currentmodule:: qiskit_optimization.translators

Translators between :class:`~qiskit_optimization.problems.QuadraticProgram` and
other optimization models.

Translators
======================
.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   from_docplex_mp
   to_docplex_mp
   from_gurobipy
   to_gurobipy
   from_ising
   to_ising
"""

from .docplex_mp import from_docplex_mp, to_docplex_mp
from .gurobipy import from_gurobipy, to_gurobipy
from .ising import from_ising, to_ising

_all = [
    "from_docplex_mp",
    "to_docplex_mp",
    "from_gurobipy",
    "to_gurobipy",
    "from_ising",
    "to_ising",
]
