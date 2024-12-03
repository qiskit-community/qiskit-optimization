# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test set batching."""

from test import QiskitAlgorithmsTestCase

from qiskit_optimization.optimizers import SPSA, COBYLA

from qiskit_optimization.utils.set_batching import _set_default_batchsize


class TestSetBatching(QiskitAlgorithmsTestCase):
    """Set Batching tests."""

    def test_set_default_batchsize_updates(self):
        """Test that the default batchsize is set when it is None."""
        # Create an instance of SPSA with _max_evals_grouped as None
        optimizer = SPSA()
        optimizer._max_evals_grouped = None  # Directly set the private variable for testing
        # Call the function
        updated = _set_default_batchsize(optimizer)
        # Check that the batch size was updated
        self.assertTrue(updated)
        self.assertEqual(optimizer._max_evals_grouped, 50)

    def test_set_default_batchsize_no_update(self):
        """Test that the batchsize is not updated when it is already set."""
        # Create an instance of SPSA with _max_evals_grouped already set
        optimizer = SPSA()
        optimizer._max_evals_grouped = 10  # Already set to a value
        # Call the function
        updated = _set_default_batchsize(optimizer)
        # Check that the batch size was not updated
        self.assertFalse(updated)
        self.assertEqual(optimizer._max_evals_grouped, 10)

    def test_set_default_batchsize_not_spsa(self):
        """Test that the function does not update when not an SPSA instance."""
        # Create a mock optimizer that is not an instance of SPSA
        optimizer = COBYLA()
        optimizer._max_evals_grouped = None  # COBYLA doesn't need the actual implementation
        # Call the function
        updated = _set_default_batchsize(optimizer)
        # Check that the batch size was not updated
        self.assertFalse(updated)
