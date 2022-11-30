# Qiskit Optimization

[![License](https://img.shields.io/github/license/Qiskit/qiskit-optimization.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)<!--- long-description-skip-begin -->[![Build Status](https://github.com/Qiskit/qiskit-optimization/workflows/Optimization%20Unit%20Tests/badge.svg?branch=main)](https://github.com/Qiskit/qiskit-optimization/actions?query=workflow%3A"Optimization%20Unit%20Tests"+branch%3Amain+event%3Apush)[![](https://img.shields.io/github/release/Qiskit/qiskit-optimization.svg?style=popout-square)](https://github.com/Qiskit/qiskit-optimization/releases)[![](https://img.shields.io/pypi/dm/qiskit-optimization.svg?style=popout-square)](https://pypi.org/project/qiskit-optimization/)[![Coverage Status](https://coveralls.io/repos/github/Qiskit/qiskit-optimization/badge.svg?branch=main)](https://coveralls.io/github/Qiskit/qiskit-optimization?branch=main)<!--- long-description-skip-end -->

**Qiskit Optimization** is an open-source framework that covers the whole range from high-level modeling of optimization
problems, with automatic conversion of problems to different required representations, to a suite
of easy-to-use quantum optimization algorithms that are ready to run on classical simulators,
as well as on real quantum devices via Qiskit.

The Optimization module enables easy, efficient modeling of optimization problems using
[docplex](https://ibmdecisionoptimization.github.io/docplex-doc/).
A uniform interface as well as automatic conversion between different problem representations
allows users to solve problems using a large set of algorithms, from variational quantum algorithms,
such as the Quantum Approximate Optimization Algorithm QAOA, to Grover Adaptive Search using the
GroverOptimizer
leveraging fundamental algorithms provided by Terra. Furthermore, the modular design
of the optimization module allows it to be easily extended and facilitates rapid development and
testing of new algorithms. Compatible classical optimizers are also provided for testing,
validation, and benchmarking.

## Installation

We encourage installing Qiskit Optimization via the pip tool (a python package manager).

```bash
pip install qiskit-optimization
```

**pip** will handle all dependencies automatically and you will always install the latest
(and well-tested) version.

If you want to work on the very latest work-in-progress versions, either to try features ahead of
their official release or if you want to contribute to Optimization, then you can install from source.
To do this follow the instructions in the
 [documentation](https://qiskit.org/documentation/optimization/getting_started.html#installation).


----------------------------------------------------------------------------------------------------

### Optional Installs

* **IBM CPLEX** may be installed using `pip install 'qiskit-optimization[cplex]'` to enable the reading of `LP` files and the usage of
  the `CplexOptimizer`, wrapper for ``cplex.Cplex``. Currently there is no python 3.9 version of CPLEX. In this case, the CPLEX install
  command will have no effect.

* **CVXPY** may be installed using the command `pip install 'qiskit-optimization[cvx]'`.
  CVXPY being installed will enable the usage of the Goemans-Williamson algorithm as an optimizer `GoemansWilliamsonOptimizer`.

* **Matplotlib** may be installed using the command `pip install 'qiskit-optimization[matplotlib]'`.
  Matplotlib being installed will enable the usage of the `draw` method in the graph optimization application classes.

* **Gurobipy** may be installed using the command `pip install 'qiskit-optimization[gurobi]'`.
  Gurobipy being installed will enable the usage of the GurobiOptimizer.

### Creating Your First Optimization Programming Experiment in Qiskit

Now that Qiskit Optimization is installed, it's time to begin working with the optimization module.
Let's try an optimization experiment to compute the solution of a
[Max-Cut](https://en.wikipedia.org/wiki/Maximum_cut). The Max-Cut problem can be formulated as
quadratic program, which can be solved using many several different algorithms in Qiskit.
In this example, the MinimumEigenOptimizer
is employed in combination with the Quantum Approximate Optimization Algorithm (QAOA) as minimum
eigensolver routine.

```python
from docplex.mp.model import Model

from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.translators import from_docplex_mp

from qiskit.utils import algorithm_globals
from qiskit.primitives import Sampler
from qiskit.algorithms.minimum_eigensolvers import QAOA
from qiskit.algorithms.optimizers import SPSA

# Generate a graph of 4 nodes
n = 4
edges = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0), (2, 3, 1.0)]  # (node_i, node_j, weight)

# Formulate the problem as a Docplex model
model = Model()

# Create n binary variables
x = model.binary_var_list(n)

# Define the objective function to be maximized
model.maximize(model.sum(w * x[i] * (1 - x[j]) + w * (1 - x[i]) * x[j] for i, j, w in edges))

# Fix node 0 to be 1 to break the symmetry of the max-cut solution
model.add(x[0] == 1)

# Convert the Docplex model into a `QuadraticProgram` object
problem = from_docplex_mp(model)

# Run quantum algorithm QAOA on qasm simulator
seed = 1234
algorithm_globals.random_seed = seed

spsa = SPSA(maxiter=250)
sampler = Sampler()
qaoa = QAOA(sampler=sampler, optimizer=spsa, reps=5)
algorithm = MinimumEigenOptimizer(qaoa)
result = algorithm.solve(problem)
print(result.prettyprint())  # prints solution, x=[1, 0, 1, 0], the cost, fval=4
```

### Further examples

Learning path notebooks may be found in the
[optimization tutorials](https://qiskit.org/documentation/optimization/tutorials/index.html) section
of the documentation and are a great place to start.

----------------------------------------------------------------------------------------------------

## Contribution Guidelines

If you'd like to contribute to Qiskit, please take a look at our
[contribution guidelines](https://github.com/Qiskit/qiskit-optimization/blob/main/CONTRIBUTING.md).
This project adheres to Qiskit's [code of conduct](https://github.com/Qiskit/qiskit-optimization/blob/main/CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

We use [GitHub issues](https://github.com/Qiskit/qiskit-optimization/issues) for tracking requests and bugs. Please
[join the Qiskit Slack community](https://ibm.co/joinqiskitslack)
and for discussion and simple questions.
For questions that are more suited for a forum, we use the **Qiskit** tag in [Stack Overflow](https://stackoverflow.com/questions/tagged/qiskit).

## Authors and Citation

Optimization was inspired, authored and brought about by the collective work of a team of researchers.
Optimization continues to grow with the help and work of
[many people](https://github.com/Qiskit/qiskit-optimization/graphs/contributors), who contribute
to the project at different levels.
If you use Qiskit, please cite as per the provided
[BibTeX file](https://github.com/Qiskit/qiskit/blob/master/Qiskit.bib).

Please note that if you do not like the way your name is cited in the BibTex file then consult
the information found in the [.mailmap](https://github.com/Qiskit/qiskit-optimization/blob/main/.mailmap)
file.

## License

This project uses the [Apache License 2.0](https://github.com/Qiskit/qiskit-optimization/blob/main/LICENSE.txt).
