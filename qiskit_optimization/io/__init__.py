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
Quadratic program input/output (:mod:`qiskit_optimization.io`)
======================================================================

.. currentmodule:: qiskit_optimization.io

Input and output of :class:`~qiskit_optimization.problems.QuadraticProgram`.
Users can read a file to generate a quadratic program and write a quadratic program
to a file.

Input/output
======================
.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

    read_lp
    write_lp
"""

from .lp import read_lp, write_lp

_all = ["read_lp", "write_lp"]
