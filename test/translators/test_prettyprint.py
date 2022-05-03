# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test prettyprint"""

from test.optimization_test_case import QiskitOptimizationTestCase

from qiskit_optimization.exceptions import QiskitOptimizationError
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.translators.prettyprint import prettyprint


class TestPrettyprint(QiskitOptimizationTestCase):
    """Test prettyprint"""

    @staticmethod
    def _convert(out: str):
        print('"\\n".join([')
        for line in out.split("\n"):
            print(f'"{line}",')
        print("])")

    def test_prettyprint(self):
        """test prettyprint"""

        with self.subTest("empty"):
            q_p = QuadraticProgram()
            expected = "\n".join(
                [
                    "Problem name: ",
                    "",
                    "Minimize",
                    "  0",
                    "",
                    "Subject to",
                    "  No constraints",
                    "",
                    "  No variables",
                    "",
                ]
            )
            self.assertEqual(prettyprint(q_p), expected)
            self.assertEqual(q_p.prettyprint(), expected)

        with self.subTest("minimize 1"):
            q_p = QuadraticProgram("test")
            q_p.binary_var(name="x")
            q_p.binary_var(name="y")
            q_p.minimize(linear={"x": 1}, quadratic={("x", "y"): 1})
            expected = "\n".join(
                [
                    "Problem name: test",
                    "",
                    "Minimize",
                    "  x*y + x",
                    "",
                    "Subject to",
                    "  No constraints",
                    "",
                    "  Binary variables (2)",
                    "    x y",
                    "",
                ]
            )
            self.assertEqual(prettyprint(q_p), expected)
            self.assertEqual(q_p.prettyprint(), expected)

        with self.subTest("maximize 1"):
            q_p = QuadraticProgram("test")
            q_p.integer_var(lowerbound=5, upperbound=10, name="x")
            q_p.continuous_var(lowerbound=-1, name="y")
            q_p.maximize(linear={"x": -1, "y": 2}, constant=3)
            q_p.quadratic_constraint({}, {("x", "y"): -1}, "<=", 4)
            expected = "\n".join(
                [
                    "Problem name: test",
                    "",
                    "Maximize",
                    "  -x + 2*y + 3",
                    "",
                    "Subject to",
                    "  Quadratic constraints (1)",
                    "    -x*y <= 4  'q0'",
                    "",
                    "  Integer variables (1)",
                    "    5 <= x <= 10",
                    "",
                    "  Continuous variables (1)",
                    "    -1 <= y",
                    "",
                ]
            )
            self.assertEqual(prettyprint(q_p), expected)
            self.assertEqual(q_p.prettyprint(), expected)

        with self.subTest("minimize 2"):
            q_p = QuadraticProgram("my problem")
            q_p.binary_var("x")
            q_p.integer_var(-1, 5, "y")
            q_p.continuous_var(-1, 5, "z")
            q_p.minimize(1, {"x": 1, "y": -1, "z": 10}, {("x", "x"): 0.5, ("y", "z"): -1})
            q_p.linear_constraint({"x": 1, "y": 2}, "==", 1, "lin_eq")
            q_p.linear_constraint({"x": 1, "y": 2}, "<=", 1, "lin_leq")
            q_p.linear_constraint({"x": 1, "y": 2}, ">=", 1, "lin_geq")
            q_p.quadratic_constraint(
                {"x": 1, "y": 1},
                {("x", "x"): 1, ("y", "z"): -1, ("z", "z"): 2},
                "==",
                1,
                "quad_eq",
            )
            q_p.quadratic_constraint(
                {"x": 1, "y": 1},
                {("x", "x"): 1, ("y", "z"): -1, ("z", "z"): 2},
                "<=",
                1,
                "quad_leq",
            )
            q_p.quadratic_constraint(
                {"x": 1, "y": 1},
                {("x", "x"): 1, ("y", "z"): -1, ("z", "z"): 2},
                ">=",
                1,
                "quad_geq",
            )
            expected = "\n".join(
                [
                    "Problem name: my problem",
                    "",
                    "Minimize",
                    "  0.5*x^2 - y*z + x - y + 10*z + 1",
                    "",
                    "Subject to",
                    "  Linear constraints (3)",
                    "    x + 2*y == 1  'lin_eq'",
                    "    x + 2*y <= 1  'lin_leq'",
                    "    x + 2*y >= 1  'lin_geq'",
                    "",
                    "  Quadratic constraints (3)",
                    "    x^2 - y*z + 2*z^2 + x + y == 1  'quad_eq'",
                    "    x^2 - y*z + 2*z^2 + x + y <= 1  'quad_leq'",
                    "    x^2 - y*z + 2*z^2 + x + y >= 1  'quad_geq'",
                    "",
                    "  Integer variables (1)",
                    "    -1 <= y <= 5",
                    "",
                    "  Continuous variables (1)",
                    "    -1 <= z <= 5",
                    "",
                    "  Binary variables (1)",
                    "    x",
                    "",
                ]
            )
            self.assertEqual(prettyprint(q_p), expected)
            self.assertEqual(q_p.prettyprint(), expected)

        with self.subTest("maximize 2"):
            q_p = QuadraticProgram("problem 1")
            q_p.integer_var(-1, 2, "x")
            q_p.integer_var(-1, 2, "y")
            q_p.continuous_var(-1, name="z")
            q_p.binary_var("u")
            q_p.binary_var("very_long_variable")
            q_p.binary_var_list(10)
            q_p.integer_var_list(3)
            q_p.continuous_var_list(3)
            q_p.minimize(constant=3, linear={"x": 2, "y": 3}, quadratic={("u", "x"): -1})
            q_p.linear_constraint({"x": 1, "y": -2}, ">=", 2, name="lin_GE")
            q_p.linear_constraint({"x": 2, "y": -1}, "==", 1, name="lin_EQ")
            q_p.quadratic_constraint(
                {"x": 1, "u": 1}, {(3, 4): 1, (5, 6): -1}, "<=", 1, name="quad_LE"
            )
            q_p.quadratic_constraint({"x": 2, "y": -1}, {("z", "z"): -1}, "<=", 1)
            expected = "\n".join(
                [
                    "Problem name: problem 1",
                    "",
                    "Minimize",
                    "  -x*u + 2*x + 3*y + 3",
                    "",
                    "Subject to",
                    "  Linear constraints (2)",
                    "    x - 2*y >= 2  'lin_GE'",
                    "    2*x - y == 1  'lin_EQ'",
                    "",
                    "  Quadratic constraints (2)",
                    "    u*very_long_variable - x5*x6 + u + x <= 1  'quad_LE'",
                    "    -z^2 + 2*x - y <= 1  'q1'",
                    "",
                    "  Integer variables (5)",
                    "    -1 <= x <= 2",
                    "    -1 <= y <= 2",
                    "    0 <= x15",
                    "    0 <= x16",
                    "    0 <= x17",
                    "",
                    "  Continuous variables (4)",
                    "    -1 <= z",
                    "    0 <= x18",
                    "    0 <= x19",
                    "    0 <= x20",
                    "",
                    "  Binary variables (12)",
                    "    u very_long_variable x5 x6 x7 x8 x9 x10 x11 x12 x13 x14",
                    "",
                ]
            )
            self.assertEqual(prettyprint(q_p), expected)
            self.assertEqual(q_p.prettyprint(), expected)

    def test_prettyprint2(self):
        """test prettyprint with special characters (' ', '+', '-', '*')"""
        q_p = QuadraticProgram("+ test - test *")
        q_p.binary_var(" x ")
        q_p.continuous_var(1e-10, 1e10, "+y+")
        q_p.continuous_var(-1e10, -1e-10, "-y-")
        q_p.integer_var(10, 100, "*z*")
        q_p.minimize(1, [1, 2, 3, 4])
        expected = "\n".join(
            [
                "Problem name: + test - test *",
                "",
                "Minimize",
                "   x  + 4**z* + 2*+y+ + 3*-y- + 1",
                "",
                "Subject to",
                "  No constraints",
                "",
                "  Integer variables (1)",
                "    10 <= *z* <= 100",
                "",
                "  Continuous variables (2)",
                "    1e-10 <= +y+ <= 10000000000",
                "    -10000000000 <= -y- <= -1e-10",
                "",
                "  Binary variables (1)",
                "     x ",
                "",
            ]
        )
        self.assertEqual(prettyprint(q_p), expected)
        self.assertEqual(q_p.prettyprint(), expected)

    def test_error(self):
        """Test error case due to non-printable names"""

        name = "\n"
        with self.subTest("problem name - func"), self.assertRaises(QiskitOptimizationError):
            q_p = QuadraticProgram(name)
            _ = prettyprint(q_p)

        with self.subTest("problem name - meth"), self.assertRaises(QiskitOptimizationError):
            q_p = QuadraticProgram(name)
            _ = q_p.prettyprint()

        with self.subTest("linear variable name - func"), self.assertRaises(
            QiskitOptimizationError
        ):
            q_p = QuadraticProgram()
            q_p.binary_var(name)
            _ = prettyprint(q_p)

        with self.subTest("linear variable name - meth"), self.assertRaises(
            QiskitOptimizationError
        ):
            q_p = QuadraticProgram()
            q_p.binary_var(name)
            _ = q_p.prettyprint()

        with self.subTest("linear constraint name - func"), self.assertRaises(
            QiskitOptimizationError
        ):
            q_p = QuadraticProgram()
            q_p.linear_constraint(name=name)
            _ = prettyprint(q_p)

        with self.subTest("linear constraint name - meth"), self.assertRaises(
            QiskitOptimizationError
        ):
            q_p = QuadraticProgram()
            q_p.linear_constraint(name=name)
            _ = q_p.prettyprint()

        with self.subTest("quadratic constraint name - func"), self.assertRaises(
            QiskitOptimizationError
        ):
            q_p = QuadraticProgram()
            q_p.quadratic_constraint(name=name)
            _ = prettyprint(q_p)

        with self.subTest("quadratic constraint name - meth"), self.assertRaises(
            QiskitOptimizationError
        ):
            q_p = QuadraticProgram()
            q_p.quadratic_constraint(name=name)
            _ = q_p.prettyprint()
