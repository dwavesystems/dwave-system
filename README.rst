.. image:: https://travis-ci.org/dwavesystems/dwave-system.svg?branch=master
    :target: https://travis-ci.org/dwavesystems/dwave-system
    :alt: Travis Status

.. image:: https://coveralls.io/repos/github/dwavesystems/dwave-system/badge.svg?branch=master
    :target: https://coveralls.io/github/dwavesystems/dwave-system?branch=master
    :alt: Coveralls Status

.. image:: http://readthedocs.org/projects/dwave-system/badge/?version=latest
    :target: http://dwave-system.readthedocs.io/en/latest/?badge=latest
    :alt: Read the Docs Status

.. index-start-marker

Note: This is an alpha release of this package.

dwave-system
============

dwave-system contains all of the logic needed to use the D-Wave System as part of the
`D-Wave Ocean <todo>`_ software stack.

.. index-end-marker

Installation
------------

.. installation-start-marker

**Installation from PyPI:**

.. code-block:: bash

    pip install dwave-system --extra-index-url https://pypi.dwavesys.com/simple

**Installation from source:**

.. code-block:: bash

    pip install -r requirements.txt --extra-index-url https://pypi.dwavesys.com/simple
    python setup.py

Downloaded with this package is a dependency called dwave-system-tuning that has a restricted license.
To view the license details:

.. code-block:: python

    from dwave.system.tuning import __license__
    print(__license__)

To uninstall the proprietary components:

.. code-block:: bash

    pip uninstall dwave-system-tuning

.. installation-end-marker


License
-------

Released under the Apache License 2.0. See LICENSE file.

Contribution
------------

See CONTRIBUTING.rst file.
