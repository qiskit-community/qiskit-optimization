# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test AlgorithmResult"""

import unittest

from test import QiskitAlgorithmsTestCase

from qiskit_optimization.algorithm_result import AlgorithmResult


class TestAlgorithmResult(AlgorithmResult):
    """Concrete subclass for testing purposes"""

    def __init__(self, data):
        self.data = data
        self.name = "Test Result"


class TestAlgorithmResultMethods(QiskitAlgorithmsTestCase):
    """AlgorithmResult tests."""

    def setUp(self):
        """Setting up initial test objects"""
        self.result1 = TestAlgorithmResult({"value1": 10, "value2": 20})
        self.result2 = TestAlgorithmResult({"value1": 100, "value2": 200})
        self.result3 = TestAlgorithmResult({"value3": 300})

    def test_str_method(self):
        """Test the __str__ method"""
        expected_str = "{'data': {'value1': 100, 'value2': 200}, 'name': 'Test Result'}"
        self.assertEqual(
            self.result1.__str__(), expected_str  # pylint: disable=unnecessary-dunder-call
        )

    def test_combine_with_another_result(self):
        """Test the combine method with another result"""
        self.result1.combine(self.result2)
        self.assertEqual(self.result1.data, {"value1": 100, "value2": 200})
        self.assertEqual(self.result1.name, "Test Result")

    def test_combine_with_new_property(self):
        """Test combining with a result that has a new property"""
        self.result1.combine(self.result3)
        self.assertEqual(self.result1.data, {"value3": 300})
        self.assertEqual(self.result1.name, "Test Result")

    def test_combine_with_none_raises_error(self):
        """Test that passing None to combine raises TypeError"""
        with self.assertRaises(TypeError):
            self.result1.combine(None)

    def test_combine_with_self_does_nothing(self):
        """Test that combining with self doesn't change anything"""
        original_data = self.result1.data.copy()
        self.result1.combine(self.result1)
        self.assertEqual(self.result1.data, original_data)


if __name__ == "__main__":
    unittest.main()
