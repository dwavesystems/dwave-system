.. image:: https://img.shields.io/pypi/v/dwave-system.svg
   :target: https://pypi.org/project/dwave-system

.. image:: https://img.shields.io/pypi/pyversions/dwave-system.svg?style=flat
    :target: https://pypi.org/project/dwave-system
    :alt: PyPI - Python Version

.. image:: https://codecov.io/gh/dwavesystems/dwave-system/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/dwavesystems/dwave-system

.. image:: https://circleci.com/gh/dwavesystems/dwave-system.svg?style=shield
   :target: https://circleci.com/gh/dwavesystems/dwave-system


============
dwave-system
============

.. start_system_about

``dwave-system`` is an API for easily incorporating D-Wave quantum computers as
a sampler in the
`Ocean software stack <https://docs.dwavequantum.com/en/latest/ocean/stack.html>`_,
directly or through the `Leap <https://cloud.dwavesys.com/leap/>`_ service's
cloud-based hybrid solvers. It includes ``DWaveSampler``, a dimod sampler that
accepts and passes system parameters such as system identification and
authentication down the stack, ``StrideHybridSolver``, for Leap's Stride™ hybrid
solver, and others. It also includes several useful composites---layers of pre-
and post-processing---that can be used with ``DWaveSampler`` to handle
minor-embedding, optimize chain strength, etc.

.. end_system_about

Installation
============

Installation from PyPI:

.. code-block:: bash

    pip install dwave-system

During development, install from source in editable mode:

.. code-block:: bash

    pip install -e .

License
=======

Released under the Apache License 2.0. See LICENSE file.

Contributing
============

Ocean's `contributing guide <https://docs.dwavequantum.com/en/latest/ocean/contribute.html>`_
has guidelines for contributing to Ocean packages.
