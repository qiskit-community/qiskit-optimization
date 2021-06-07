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

"""LP file reader and writer"""

from typing import TYPE_CHECKING, Any

from docplex.mp.model_reader import ModelReader

from qiskit.exceptions import MissingOptionalLibraryError

from .docplex_mp import DocplexMpTranslator
from .model_translator import ModelTranslator

if TYPE_CHECKING:
    # pylint: disable=cyclic-import
    from qiskit_optimization.problems.quadratic_program import QuadraticProgram

try:
    import cplex  # pylint: disable=unused-import

    _HAS_CPLEX = True
except ImportError:
    _HAS_CPLEX = False


class LPFileTranslator(ModelTranslator):
    """Translator between a LP file and a quadratic program"""

    def __init__(self, output_filename: str = ""):
        """
        Args:
            output_filename: The filename of the file the model is written to.
              If filename is a directory, file name 'my_problem.lp' is appended.
              If filename does not end with '.lp', suffix '.lp' is appended.
        """
        self._filename = output_filename

    def is_installed(self) -> bool:
        return _HAS_CPLEX

    def is_compatible(self, model: Any) -> bool:
        """Checks whether a file name is a string or not

        Args:
            model: a LP file name to be read.

        Returns:
            Returns True if the file name is string, False otherwise.
        """
        return isinstance(model, str) and model.lower().endswith(".lp")

    def from_qp(self, quadratic_program: "QuadraticProgram") -> None:
        """Write a docplex model to a LP file.

        Args:
            quadratic_program: The quadratic program to be translated

        Raises:
            QiskitOptimizationError: if non-supported elements (should never happen).
        """
        mdl = DocplexMpTranslator().from_qp(quadratic_program)
        mdl.export_as_lp(self._filename)

    def to_qp(self, model: Any) -> "QuadraticProgram":
        """Read a LP file to generate ``QuadraticProgram``.

        Args:
            model: a LP file name to be read.

        Returns:
            The quadratic program corresponding to the model.

        Raises:
            QiskitOptimizationError: if the model contains unsupported elements.
            MissingOptionalLibraryError: if CPLEX is not installed.
        """

        if not _HAS_CPLEX:
            raise MissingOptionalLibraryError(
                libname="CPLEX",
                name="LPFileTranslator",
                pip_install="pip install 'qiskit-optimization[cplex]'",
            )

        def _parse_problem_name(filename: str) -> str:
            # Because docplex model reader uses the base name as model name,
            # we parse the model name in the LP file manually.
            # https://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.model_reader.html
            prefix = "\\Problem name:"
            model_name = ""
            with open(filename) as file:
                for line in file:
                    if line.startswith(prefix):
                        model_name = line[len(prefix) :].strip()
                    if not line.startswith("\\"):
                        break
            return model_name

        filename = model
        mdl = ModelReader().read(filename, model_name=_parse_problem_name(filename))
        return DocplexMpTranslator().to_qp(mdl)
