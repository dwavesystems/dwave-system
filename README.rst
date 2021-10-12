.. image:: https://img.shields.io/pypi/v/dwave-system.svg
   :target: https://pypi.org/project/dwave-system

.. image:: https://ci.appveyor.com/api/projects/status/959r6vpyertcxkhd?svg=true
   :target: https://ci.appveyor.com/project/dwave-adtt/dwave-system

.. image:: https://codecov.io/gh/dwavesystems/dwave-system/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/dwavesystems/dwave-system

.. image:: https://readthedocs.com/projects/d-wave-systems-dwave-system/badge/?version=latest
   :target: https://docs.ocean.dwavesys.com/projects/system/en/latest/?badge=latest

.. image:: https://circleci.com/gh/dwavesystems/dwave-system.svg?style=svg
   :target: https://circleci.com/gh/dwavesystems/dwave-system

.. index-start-marker

dwave-system
============

`dwave-system` is a basic API for easily incorporating the D-Wave system as a
sampler in the
`D-Wave Ocean software stack <https://docs.ocean.dwavesys.com/en/stable/overview/stack.html>`_,
directly or through `Leap <https://cloud.dwavesys.com/leap/>`_\ 's cloud-based
hybrid solvers. It includes ``DWaveSampler``, a dimod sampler that accepts and
passes system parameters such as system identification and authentication down
the stack, ``LeapHybridSampler``, for Leap's hybrid solvers, and other. It also
includes several useful composites---layers of pre- and post-processing---that
can be used with ``DWaveSampler`` to handle minor-embedding,
optimize chain strength, etc.

.. index-end-marker

Installation
------------

.. installation-start-marker

**Installation from PyPI:**

.. code-block:: bash

    pip install dwave-system

**Installation from PyPI with drivers:**

.. note::
    Prior to v0.3.0, running ``pip install dwave-system`` installed a driver dependency called ``dwave-drivers``
    (previously also called ``dwave-system-tuning``). This dependency has a restricted license and has been made optional
    as of v0.3.0, but is highly recommended. To view the license details:

    .. code-block:: python

        from dwave.drivers import __license__
        print(__license__)

To install with optional dependencies:

.. code-block:: bash

    pip install dwave-system[drivers] --extra-index-url https://pypi.dwavesys.com/simple

**Installation from source:**

.. code-block:: bash

    pip install -r requirements.txt
    python setup.py install

Note that installing from source installs ``dwave-drivers``. To uninstall the proprietary components:

.. code-block:: bash

    pip uninstall dwave-drivers

.. installation-end-marker


License
-------

Released under the Apache License 2.0. See LICENSE file.

Contributing
============

Ocean's `contributing guide <https://docs.ocean.dwavesys.com/en/stable/contributing.html>`_
has guidelines for contributing to Ocean packages.
