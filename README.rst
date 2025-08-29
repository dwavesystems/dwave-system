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

``dwave-system`` is a basic API for easily incorporating the D-Wave system as a
sampler in the
`Ocean software stack <https://docs.dwavequantum.com/en/latest/ocean/stack.html>`_,
directly or through `Leap <https://cloud.dwavesys.com/leap/>`_ service's
cloud-based hybrid solvers. It includes ``DWaveSampler``, a dimod sampler that
accepts and passes system parameters such as system identification and
authentication down the stack, ``LeapHybridSampler``, for Leap's hybrid solvers,
and others. It also includes several useful composites---layers of pre- and
post-processing---that can be used with ``DWaveSampler`` to handle
minor-embedding, optimize chain strength, etc.

.. end_system_about

Installation
============

**Installation from PyPI:**

.. code-block:: bash

    pip install dwave-system

.. note::
    As of ``dwave-system`` 1.28.0, support for ``dwave-drivers`` is removed (it
    was used for calibration of qubits in chains via ``VirtualGraphComposite``,
    but it's no longer required due to improved calibration of newer QPUs).

**Installation from source:**

.. code-block:: bash

    pip install -r requirements.txt
    python setup.py install

License
=======

Released under the Apache License 2.0. See LICENSE file.

Contributing
============

Ocean's `contributing guide <https://docs.dwavequantum.com/en/latest/ocean/contribute.html>`_
has guidelines for contributing to Ocean packages.

Release Notes
-------------

``dwave-system`` makes use of `reno <https://docs.openstack.org/reno/>`_ to manage
its release notes.

When making a contribution to ``dwave-system`` that will affect users, create
a new release note file by running

.. code-block:: bash

    reno new your-short-descriptor-here

You can then edit the file created under ``releasenotes/notes/``.
Remove any sections not relevant to your changes.
Commit the file along with your changes.

See reno's `user guide <https://docs.openstack.org/reno/latest/user/usage.html>`_
for details.
