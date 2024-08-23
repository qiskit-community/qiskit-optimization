# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Exception and warnings for errors raised by Algorithms module."""

from qiskit.exceptions import QiskitError


class AlgorithmError(QiskitError):
    """For Algorithm specific errors."""

    pass


class QiskitAlgorithmsWarning(UserWarning):
    """Base class for warnings raised by Qiskit Algorithms."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(" ".join(message))
        self.message = " ".join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)


class QiskitAlgorithmsOptimizersWarning(QiskitAlgorithmsWarning):
    """For Algorithm specific warnings."""

    pass
