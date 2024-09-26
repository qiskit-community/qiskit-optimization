# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Optimizers"""

import unittest
from test import QiskitAlgorithmsTestCase

from typing import Optional, List, Tuple
from ddt import ddt, data, unpack
import numpy as np
from scipy.optimize import rosen, rosen_der

from qiskit_optimization.optimizers import (
    COBYLA,
    NELDER_MEAD,
    Optimizer,
    SPSA,
    SciPyOptimizer,
    OptimizerSupportLevel,
)
from qiskit_optimization.utils import algorithm_globals


@ddt
class TestOptimizers(QiskitAlgorithmsTestCase):
    """Test Optimizers"""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 52
        self.optimizer = SPSA()
        self.optimizer_2 = SPSA()
        self.optimizer_2.set_options(tolerance=1e-6, maxiter=100, method="SPSA")

    def run_optimizer(
        self,
        optimizer: Optimizer,
        max_nfev: int,
        grad: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ):
        """Test the optimizer.

        Args:
            optimizer: The optimizer instance to test.
            max_nfev: The maximal allowed number of function evaluations.
            grad: Whether to pass the gradient function as input.
            bounds: Optimizer bounds.
        """
        x_0 = np.asarray([1.3, 0.7, 0.8, 1.9, 1.2])
        jac = rosen_der if grad else None

        res = optimizer.minimize(rosen, x_0, jac, bounds)
        x_opt = res.x
        nfev = res.nfev

        np.testing.assert_array_almost_equal(x_opt, [1.0] * len(x_0), decimal=2)
        self.assertLessEqual(nfev, max_nfev)

    def test_cobyla(self):
        """cobyla test"""
        optimizer = COBYLA(maxiter=100000, tol=1e-06)
        self.run_optimizer(optimizer, max_nfev=100000)

    def test_nelder_mead(self):
        """nelder mead test"""
        optimizer = NELDER_MEAD(maxfev=10000, tol=1e-06)
        self.run_optimizer(optimizer, max_nfev=10000)

    @unittest.skip("Skipping SPSA as it does not do well on non-convex rozen")
    def test_spsa(self):
        """spsa test"""
        optimizer = SPSA(maxiter=10000)
        self.run_optimizer(optimizer, max_nfev=100000)

    def test_scipy_optimizer(self):
        """scipy_optimizer test"""
        optimizer = SciPyOptimizer("BFGS", options={"maxiter": 1000})
        self.run_optimizer(optimizer, max_nfev=10000)

    def test_scipy_optimizer_callback(self):
        """scipy_optimizer callback test"""
        values = []

        def callback(x):
            values.append(x)

        optimizer = SciPyOptimizer("BFGS", options={"maxiter": 1000}, callback=callback)
        self.run_optimizer(optimizer, max_nfev=10000)
        self.assertTrue(values)  # Check the list is nonempty.

    def test_scipy_optimizer_parse_bounds(self):
        """
        Test the parsing of bounds in SciPyOptimizer.minimize method. Verifies that the bounds are
        correctly parsed and set within the optimizer object.

        Raises:
            AssertionError: If any of the assertions fail.
            AssertionError: If a TypeError is raised unexpectedly while parsing bounds.

        """
        try:
            # Initialize SciPyOptimizer instance with SLSQP method
            optimizer = SciPyOptimizer("SLSQP")

            # Call minimize method with a simple lambda function and bounds
            optimizer.minimize(lambda x: -x, 1.0, bounds=[(0.0, 1.0)])

            # Assert that "bounds" is not present in optimizer options and kwargs
            self.assertFalse("bounds" in optimizer._options)
            self.assertFalse("bounds" in optimizer._kwargs)

        except TypeError:
            # This would give: https://github.com/qiskit-community/qiskit-machine-learning/issues/570
            self.fail(
                "TypeError was raised unexpectedly when parsing bounds in SciPyOptimizer.minimize(...)."
            )

        # Finally, expect exceptions if bounds are parsed incorrectly, i.e. differently than as in Scipy
        with self.assertRaises(RuntimeError):
            _ = SciPyOptimizer("SLSQP", bounds=[(0.0, 1.0)])
        with self.assertRaises(RuntimeError):
            _ = SciPyOptimizer("SLSQP", options={"bounds": [(0.0, 1.0)]})

    def test_gradient_num_diff(self):
        """Test the gradient_num_diff function."""

        # Define a simple quadratic function and its gradient
        def func(x):
            return (x[0] - 2) ** 2 + (x[1] - 3) ** 2

        def expected_gradient(x):
            return np.array([2 * (x[0] - 2), 2 * (x[1] - 3)])

        # Set the point around which we compute the gradient
        x_center = np.array([1.0, 1.0])
        epsilon = 1e-5  # Small perturbation for numerical differentiation

        # Compute the numerical gradient using the optimizer method
        numerical_gradient = self.optimizer.gradient_num_diff(x_center, func, epsilon)

        # Compute the expected gradient
        expected_grad = expected_gradient(x_center)

        # Assert that the computed gradient is close to the expected gradient
        np.testing.assert_allclose(numerical_gradient, expected_grad, rtol=1e-5, atol=1e-8)

    def test_set_options(self):
        """Test the set_options method."""

        # Define some options to set
        options = {"max_iter": 100, "tolerance": 1e-6, "verbose": True}

        # Set options using the set_options method
        self.optimizer.set_options(**options)

        # Assert that the options dictionary is updated correctly
        for key, value in options.items():
            self.assertIn(key, self.optimizer._options)
            self.assertEqual(self.optimizer._options[key], value)

        # Test updating an existing option
        self.optimizer.set_options(max_iter=200)
        self.assertEqual(self.optimizer._options["max_iter"], 200)

    def test_wrap_function(self):
        """Test the wrap_function method."""

        # Define a simple function to test
        def simple_function(x, y):
            return x + y

        # Wrap the function, injecting the argument (5,)
        wrapped_function = self.optimizer.wrap_function(simple_function, (5,))

        # Call the wrapped function with a single argument
        result = wrapped_function(10)  # Should compute 10 + 5

        # Assert that the result is as expected
        self.assertEqual(result, 15)

    def test_wrap_function_with_multiple_args(self):
        """Test wrap_function with multiple injected args."""

        # Define a simple function to test
        def multiply_function(a, b, c):
            return a * b * c

        # Wrap the function, injecting the arguments (2, 3)
        wrapped_function = self.optimizer.wrap_function(multiply_function, (2, 3))

        # Call the wrapped function with a single argument
        result = wrapped_function(4)  # Should compute 4 * 2 * 3

        # Assert that the result is as expected
        self.assertEqual(result, 24)

    def test_setting(self):
        """Test the setting property."""

        actual_output = self.optimizer.setting

        # Check if key parts are present in the settings output
        self.assertIn("Optimizer: SPSA", actual_output)
        self.assertIn("max_evals_grouped: None", actual_output)

        # Optional: check for specific support levels if required
        self.assertIn("gradient_support_level", actual_output)
        self.assertIn("bounds_support_level", actual_output)
        self.assertIn("initial_point_support_level", actual_output)

    def test_gradient_support_level(self):
        """Test for gradient support level property"""
        self.assertEqual(self.optimizer.gradient_support_level, OptimizerSupportLevel.ignored)

    def test_is_gradient_ignored(self):
        """Test for is_gradient_ignored property"""
        self.assertTrue(self.optimizer.is_gradient_ignored)

    def test_is_gradient_supported(self):
        """Test for is_gradient_supported property"""
        self.assertTrue(self.optimizer.is_gradient_supported)

    def test_is_gradient_required(self):
        """Test for is_gradient_required property"""
        self.assertFalse(self.optimizer.is_gradient_required)

    def test_bounds_support_level(self):
        """Test for bounds support level property"""
        self.assertNotEqual(self.optimizer.bounds_support_level, OptimizerSupportLevel.supported)

    def test_is_bounds_ignored(self):
        """Test for is_bounds_ignored property"""
        self.assertTrue(self.optimizer.is_bounds_ignored)

    def test_is_bounds_supported(self):
        """Test for is_bounds_supported property"""
        self.assertTrue(self.optimizer.is_bounds_supported)

    def test_is_bounds_required(self):
        """Test for is_bounds_required property"""
        self.assertFalse(self.optimizer.is_bounds_required)

    def test_initial_point_support_level(self):
        """Test for initial point support level property"""
        self.assertEqual(self.optimizer.initial_point_support_level, OptimizerSupportLevel.required)

    def test_is_initial_point_ignored(self):
        """Test for is_initial_point_ignored property"""
        self.assertFalse(self.optimizer.is_initial_point_ignored)

    def test_is_initial_point_supported(self):
        """Test for is_initial_point_supported property"""
        self.assertTrue(self.optimizer.is_initial_point_supported)

    def test_is_initial_point_required(self):
        """Test for is_initial_point_required property"""
        self.assertTrue(self.optimizer.is_initial_point_required)

    def test_set_max_evals_grouped(self):
        """Test for set_max_evals_grouped method"""
        self.optimizer.set_max_evals_grouped(10)
        self.assertEqual(self.optimizer._max_evals_grouped, 10)


@ddt
class TestOptimizerSerialization(QiskitAlgorithmsTestCase):
    """Tests concerning the serialization of optimizers."""

    @data(
        ("COBYLA", {"maxiter": 10}),
        ("NELDER_MEAD", {"maxiter": 0}),
        ("dogleg", {"maxiter": 100}),
        ("trust-constr", {"maxiter": 10}),
        ("trust-ncg", {"maxiter": 100}),
        ("trust-exact", {"maxiter": 120}),
        ("trust-krylov", {"maxiter": 150}),
    )
    @unpack
    def test_scipy(self, method, options):
        """Test the SciPyOptimizer is serializable."""

        optimizer = SciPyOptimizer(method, options=options)
        serialized = optimizer.settings
        from_dict = SciPyOptimizer(**serialized)

        self.assertEqual(from_dict._method, method.lower())
        self.assertEqual(from_dict._options, options)

    def test_independent_reconstruction(self):
        """Test the SciPyOptimizers don't reset all settings upon creating a new instance.

        COBYLA is used as representative example here."""

        kwargs = {"coffee": "without sugar"}
        options = {"tea": "with milk"}
        optimizer = COBYLA(maxiter=1, options=options, **kwargs)
        serialized = optimizer.settings
        from_dict = COBYLA(**serialized)

        with self.subTest(msg="test attributes"):
            self.assertEqual(from_dict.settings["maxiter"], 1)

        with self.subTest(msg="test options"):
            # options should only contain values that are *not* already in the initializer
            # (e.g. should not contain maxiter)
            self.assertEqual(from_dict.settings["options"], {"tea": "with milk"})

        with self.subTest(msg="test kwargs"):
            self.assertEqual(from_dict.settings["coffee"], "without sugar")

        with self.subTest(msg="option ids differ"):
            self.assertNotEqual(id(serialized["options"]), id(from_dict.settings["options"]))

    def test_spsa(self):
        """Test SPSA optimizer is serializable."""
        options = {
            "maxiter": 100,
            "blocking": True,
            "allowed_increase": 0.1,
            "second_order": True,
            "learning_rate": 0.02,
            "perturbation": 0.05,
            "regularization": 0.1,
            "resamplings": 2,
            "perturbation_dims": 5,
            "trust_region": False,
            "initial_hessian": None,
            "lse_solver": None,
            "hessian_delay": 0,
            "callback": None,
            "termination_checker": None,
        }
        spsa = SPSA(**options)

        self.assertDictEqual(spsa.settings, options)

    def test_spsa_custom_iterators(self):
        """Test serialization works with custom iterators for learning rate and perturbation."""
        rate = 0.99

        def powerlaw():
            n = 0
            while True:
                yield rate**n
                n += 1

        def steps():
            n = 1
            divide_after = 20
            epsilon = 0.5
            while True:
                yield epsilon
                n += 1
                if n % divide_after == 0:
                    epsilon /= 2

        learning_rate = powerlaw()
        expected_learning_rate = np.array([next(learning_rate) for _ in range(200)])

        perturbation = steps()
        expected_perturbation = np.array([next(perturbation) for _ in range(200)])

        spsa = SPSA(maxiter=200, learning_rate=powerlaw, perturbation=steps)
        settings = spsa.settings

        self.assertTrue(np.allclose(settings["learning_rate"], expected_learning_rate))
        self.assertTrue(np.allclose(settings["perturbation"], expected_perturbation))


if __name__ == "__main__":
    unittest.main()
