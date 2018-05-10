.. image:: https://img.shields.io/pypi/v/dwave-system.svg
    :target: https://pypi.python.org/pypi/dwave-system

.. image:: https://readthedocs.org/projects/dwave-system/badge/?version=latest
    :target: http://dwave-system.readthedocs.io/en/latest/?badge=latest

.. image:: https://coveralls.io/repos/github/dwavesystems/dwave-system/badge.svg?branch=master
    :target: https://coveralls.io/github/dwavesystems/dwave-system?branch=master

.. image:: https://circleci.com/gh/dwavesystems/dwave-system.svg?style=svg
    :target: https://circleci.com/gh/dwavesystems/dwave-system

.. index-start-marker

Note: This is an alpha release of this package.

dwave-system
============

`dwave-system` is a basic API for easily incorporating the D-Wave system as a sampler in
the `D-Wave Ocean <todo>`_ software stack. It includes DWaveSampler, a :class:`dimod.Sampler`
that accepts and passes system parameters such as system identification and authentication
down the stack. It also includes several useful composites---layers of pre- and post-processing---that
can be used with DWaveSampler to handle minor-embedding, optimize chain strength, etc.

.. index-end-marker

Installation
------------

.. installation-start-marker

**Installation from PyPI:**

.. code-block:: bash

    pip install dwave-system

**Installation from PyPI with drivers:**

.. note::
    Prior to v0.3.0, running :code:`pip install dwave-system` installed a driver dependency called :code:`dwave-system-tuning`. This dependency has a restricted license and has been made optional as of v0.3.0, 
    but is highly recommanded. To view the license details:

    .. code-block:: python

        from dwave.system.tuning import __license__
        print(__license__)

To install with optional dependencies:

.. code-block:: bash

    pip install dwave-system[drivers] --extra-index-url https://pypi.dwavesys.com/simple

**Installation from source:**

.. code-block:: bash

    pip install -r requirements.txt
    python setup.py

Note that installing from source installs dwave-system-tuning. To uninstall the proprietary components:

.. code-block:: bash

    pip uninstall dwave-system-tuning

.. installation-end-marker


License
-------

Released under the Apache License 2.0. See LICENSE file.

Contribution
------------

See CONTRIBUTING.rst file.
