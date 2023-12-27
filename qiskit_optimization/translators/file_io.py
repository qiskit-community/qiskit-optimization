# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""LP/MPS-file import / export for Quadratic Programs."""

import os
import logging
from typing import List, Callable
from gzip import open as gzip_open
from pathlib import Path
from tempfile import NamedTemporaryFile

from docplex.mp.model_reader import ModelReader

import qiskit_optimization.optionals as _optionals
from ..problems.quadratic_program import QuadraticProgram

logger = logging.getLogger(__name__)


def export_as_lp_string(q_p: QuadraticProgram) -> str:
    """Returns the quadratic program as a string of LP format.

    Args:
        q_p: The quadratic program to be exported.

    Returns:
        A string representing the quadratic program.
    """
    from .docplex_mp import to_docplex_mp

    return to_docplex_mp(q_p).export_as_lp_string()


@_optionals.HAS_CPLEX.require_in_call
def export_as_mps_string(q_p: QuadraticProgram) -> str:
    """Returns the quadratic program as a string of LP format.

    Args:
        q_p: The quadratic program to be exported.

    Returns:
        A string representing the quadratic program.
    """
    from .docplex_mp import to_docplex_mp

    return to_docplex_mp(q_p).export_as_mps_string()


@_optionals.HAS_CPLEX.require_in_call
def _read_from_file(
    filename: str, extensions: List[str], name_parse_fun: Callable
) -> QuadraticProgram:
    """Loads a quadratic program from an LP or MPS file. Also deals with
    gzip'ed files.

    Args:
        filename: The filename of the file to be loaded.
        name_parse_fun: Function that parses the model name from the input file.

    Raises:
        IOError: If the file type is not recognized, not supported or the file is not found.

    Note:
        This method requires CPLEX to be installed and present in ``PYTHONPATH``.
    """

    # check whether this file type is supported
    extension = "".join(Path(filename).suffixes)
    main_extension = extension
    if main_extension.endswith(".gz"):
        main_extension = main_extension[:-3]
    if main_extension not in extensions:
        raise IOError("File type not supported for model reading.")

    # uncompress and parse
    if extension.endswith(".gz"):
        with gzip_open(filename, "rb") as compressed:
            # requires delete=False to avoid permission issues under windows
            with NamedTemporaryFile(suffix=extension[:-3], delete=False) as uncompressed:
                uncompressed.write(compressed.read())
                uncompressed.seek(0)
                uncompressed.flush()

                model = ModelReader().read(
                    uncompressed.name,
                    model_name=name_parse_fun(uncompressed.name)
                    if name_parse_fun is not None
                    else None,
                )
                uncompressed.close()
                os.unlink(uncompressed.name)

    else:
        model = ModelReader().read(
            filename,
            model_name=name_parse_fun(filename) if name_parse_fun is not None else None,
        )

    # pylint: disable=cyclic-import
    from ..translators.docplex_mp import from_docplex_mp

    return from_docplex_mp(model)


@_optionals.HAS_CPLEX.require_in_call
def read_from_lp_file(filename: str) -> QuadraticProgram:
    """Loads the quadratic program from a LP file (may be gzip'ed).

    Args:
        filename: The filename of the file to be loaded.

    Raises:
        IOError: If the file type is not recognized, not supported or the file is not found.

    Note:
        This method requires CPLEX to be installed and present in ``PYTHONPATH``.
    """

    def _parse_problem_name(filename: str) -> str:
        # Because docplex model reader uses the base name as model name,
        # we parse the model name in the LP file manually.
        # https://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.model_reader.html
        prefix = "\\Problem name:"
        model_name = ""
        with open(filename, encoding="utf8") as file:
            for line in file:
                if line.startswith(prefix):
                    model_name = line[len(prefix) :].strip()
                if not line.startswith("\\"):
                    break
        return model_name

    return _read_from_file(filename, [".lp"], _parse_problem_name)


@_optionals.HAS_CPLEX.require_in_call
def read_from_mps_file(filename: str) -> QuadraticProgram:
    """Loads the quadratic program from a MPS file (may be gzip'ed).

    Args:
        filename: The filename of the file to be loaded.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file type is not recognized or not supported.

    Note:
        This method requires CPLEX to be installed and present in ``PYTHONPATH``.
    """

    def _parse_problem_name(filename: str) -> str:
        # Because docplex model reader uses the base name as model name,
        # we parse the model name in the LP file manually.
        # https://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.model_reader.html
        prefix = "NAME "
        model_name = ""
        with open(filename, encoding="utf8") as file:
            for line in file:
                if line.startswith(prefix):
                    model_name = line[len(prefix) :].strip()
                    break
        return model_name

    return _read_from_file(filename, [".mps"], _parse_problem_name)


def write_to_lp_file(q_p: QuadraticProgram, filename: str):
    """Writes the quadratic program to an LP file.

    Args:
        q_p: The quadratic program to be exported.
        filename: The filename of the file the model is written to.
            If filename is a directory, file name 'my_problem.lp' is appended.
            If filename does not end with '.lp', suffix '.lp' is appended.

    Raises:
        OSError: If this cannot open a file.
        DOcplexException: If filename is an empty string
    """
    from .docplex_mp import to_docplex_mp

    mdl = to_docplex_mp(q_p)
    mdl.export_as_lp(filename)


@_optionals.HAS_CPLEX.require_in_call
def write_to_mps_file(q_p: QuadraticProgram, filename: str):
    """Writes the quadratic program to an MPS file.

    Args:
        q_p: The quadratic program to be exported.
        filename: The filename of the file the model is written to.
            If filename is a directory, file name 'my_problem.mps' is appended.
            If filename does not end with '.mps', suffix '.mps' is appended.

    Raises:
        OSError: If this cannot open a file.
        DOcplexException: If filename is an empty string
    """
    from .docplex_mp import to_docplex_mp

    mdl = to_docplex_mp(q_p)
    full_path = mdl.export_as_mps(filename)

    # docplex does not write the model's name out, so we do this here manually
    with open(full_path, "r", encoding="utf8") as mps_file:
        txt = mps_file.read()

    with open(full_path, "w", encoding="utf8") as mps_file:
        for line in txt.splitlines():
            if line.startswith("NAME"):
                mps_file.write(f"NAME {q_p.name}\n")
            else:
                mps_file.write(line + "\n")
