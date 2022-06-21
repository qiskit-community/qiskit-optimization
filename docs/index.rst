############################
Qiskit Optimization overview
############################

Overview
==============

**Qiskit Optimization** is an open-source framework that covers the whole range from high-level modeling of optimization
problems, with automatic conversion of problems to different required representations, to a suite
of easy-to-use quantum optimization algorithms that are ready to run on classical simulators,
as well as on real quantum devices via Qiskit.

The Optimization module enables easy, efficient modeling of optimization problems using
`docplex <https://ibmdecisionoptimization.github.io/docplex-doc/>`__.
A uniform interface as well as automatic conversion between different problem representations
allows users to solve problems using a large set of algorithms, from variational quantum algorithms,
such as the Quantum Approximate Optimization Algorithm QAOA, to Grover Adaptive Search using the
GroverOptimizer
leveraging fundamental algorithms provided by Terra. Furthermore, the modular design
of the optimization module allows it to be easily extended and facilitates rapid development and
testing of new algorithms. Compatible classical optimizers are also provided for testing,
validation, and benchmarking.


Next Steps
=================================

`Getting started <getting_started.html>`_

`Tutorials <tutorials/index.html>`_

.. toctree::
    :hidden:

    Overview <self>
    Getting Started <getting_started>
    Tutorials <tutorials/index>
    API Reference <apidocs/qiskit_optimization>
    Release Notes <release_notes>
    GitHub <https://github.com/Qiskit/qiskit-optimization>


.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
   :ref:`search`
