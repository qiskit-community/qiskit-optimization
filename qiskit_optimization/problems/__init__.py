# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Optimization problems (:mod:`qiskit_optimization.problems`)
===========================================================

.. currentmodule:: qiskit_optimization.problems

Quadratic program
=================
Structures for defining an optimization problem.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QuadraticProgram

Note:
    The following classes are not intended to be instantiated directly.
    Objects of these types are available within an instantiated :class:`QuadraticProgram`.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Constraint
   LinearExpression
   LinearConstraint
   QuadraticExpression
   QuadraticConstraint
   QuadraticObjective
   QuadraticProgramElement
   Variable

"""

from .constraint import Constraint
from .linear_constraint import LinearConstraint
from .linear_expression import LinearExpression
from .quadratic_constraint import QuadraticConstraint
from .quadratic_expression import QuadraticExpression
from .quadratic_objective import QuadraticObjective
from .quadratic_program import QuadraticProgram
from .quadratic_program_element import QuadraticProgramElement
from .variable import Variable, VarType

__all__ = [
    "Constraint",
    "LinearExpression",
    "LinearConstraint",
    "QuadraticExpression",
    "QuadraticConstraint",
    "QuadraticObjective",
    "QuadraticProgram",
    "QuadraticProgramElement",
    "Variable",
    "VarType",
]
