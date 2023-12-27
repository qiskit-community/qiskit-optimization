# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2023.
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
other optimization models or other objects.

Translators
----------------------
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
from .file_io import (
    export_as_lp_string,
    export_as_mps_string,
    read_from_lp_file,
    read_from_mps_file,
    write_to_lp_file,
    write_to_mps_file,
)
from .gurobipy import from_gurobipy, to_gurobipy
from .ising import from_ising, to_ising

__all__ = [
    "from_docplex_mp",
    "to_docplex_mp",
    "from_gurobipy",
    "to_gurobipy",
    "from_ising",
    "to_ising",
    "export_as_lp_string",
    "export_as_mps_string",
    "read_from_lp_file",
    "read_from_mps_file",
    "write_to_lp_file",
    "write_to_mps_file",
]
