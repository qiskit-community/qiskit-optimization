# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Introduced new type to maintain readability."""

from typing import Optional, TypeVar, Union

_T = TypeVar("_T")  # Pylint does not allow single character class names.
ListOrDict = Union[list[Optional[_T]], dict[str, _T]]
