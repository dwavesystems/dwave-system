.. image:: https://travis-ci.org/dwavesystems/dwave_virtual_graph.svg?branch=master
    :target: https://travis-ci.org/dwavesystems/dwave_virtual_graph
    :alt: Travis Status

.. image:: https://coveralls.io/repos/github/dwavesystems/dwave_virtual_graph/badge.svg?branch=master
    :target: https://coveralls.io/github/dwavesystems/dwave_virtual_graph?branch=master
    :alt: Coverage Report

.. image:: https://readthedocs.org/projects/dwave_virtual_graph/badge/?version=latest
    :target: http://dwave_virtual_graph.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. inclusion-marker-do-not-remove

D-Wave Virtual Graph
====================

todo

Rational
--------

todo

Installation
------------

To install:

.. code-block:: bash

    pip install dwave_virtual_graph

To build from souce:

.. code-block:: bash
    
    pip install -r requirments.txt
    python setup.py install

Example Usage
-------------

.. code-block:: python
    
    import dwave_micro_client_dimod as micro
    import dwave_virtual_graph as vg

    # get the D-Wave sampler (see configuration_ for setting up credentials)
    dwave_sampler = micro.DWaveSampler()

    # get the dwave_sampler's structure
    nodelist, edgelist, adj = dwave_sampler.structure

    # create and load an embedding
    embedding = {0: [8, 12], 1: [9, 13], 2: [10, 14], 3: [11, 15]}
    vg.load_embedding(nodelist, edgelist, embedding, 'K4')

    # create virtual graph
    sampler = vg.VirtualGraph(dwave_sampler, 'K4')

License
-------

Released under the Apache License 2.0

.. _configuration: http://dwave-micro-client.readthedocs.io/en/latest/#configuration