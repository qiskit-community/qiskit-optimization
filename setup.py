# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import setuptools
import inspect
import sys
import os
import re

with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

if not hasattr(setuptools, "find_namespace_packages") or not inspect.ismethod(
    setuptools.find_namespace_packages
):
    print(
        "Your setuptools version:'{}' does not support PEP 420 (find_namespace_packages). "
        "Upgrade it to version >='40.1.0' and repeat install.".format(setuptools.__version__)
    )
    sys.exit(1)

VERSION_PATH = os.path.join(os.path.dirname(__file__), "qiskit_optimization", "VERSION.txt")
with open(VERSION_PATH, "r") as version_file:
    VERSION = version_file.read().strip()

# Read long description from README.
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")
with open(README_PATH) as readme_file:
    README = re.sub(
        "<!--- long-description-skip-begin -->.*<!--- long-description-skip-end -->",
        "",
        readme_file.read(),
        flags=re.S | re.M,
    )

setuptools.setup(
    name="qiskit-optimization",
    version=VERSION,
    description="Qiskit Optimization: A library for optimization applications using quantum computing",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/qiskit-community/qiskit-optimization",
    author="Qiskit Optimization Development Team",
    author_email="qiskit@us.ibm.com",
    license="Apache-2.0",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qiskit sdk quantum optimization",
    packages=setuptools.find_packages(include=["qiskit_optimization", "qiskit_optimization.*"]),
    install_requires=REQUIREMENTS,
    include_package_data=True,
    python_requires=">=3.8",
    extras_require={
        "cplex": ["cplex; python_version < '3.12' and platform_machine != 'arm64'"],
        "cvx": ["cvxpy"],
        "matplotlib": ["matplotlib"],
        "gurobi": ["gurobipy"],
    },
    project_urls={
        "Bug Tracker": "https://github.com/qiskit-community/qiskit-optimization/issues",
        "Documentation": "https://qiskit-community.github.io/qiskit-optimization/",
        "Source Code": "https://github.com/qiskit-community/qiskit-optimization",
    },
    zip_safe=False,
)
