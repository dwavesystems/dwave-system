.. _system_warnings:

========
Warnings
========

The ``dwave-system`` package supports various warning classes and provides the
ability to configure warning handling.


Supported Warnings
==================

The following warnings are currently supported.

.. currentmodule:: dwave.system.warnings

.. autoclass:: ChainBreakWarning
.. autoclass:: ChainLengthWarning
.. autoclass:: ChainStrengthWarning
.. autoclass:: EnergyScaleWarning
.. autoclass:: TooFewSamplesWarning

Related Information
-------------------

*   :ref:`qpu_embedding_intro` and :ref:`qpu_embedding_guidance` describe chains
    and how to deal with broken chains.
*   :ref:`qpu_basic_config` and :ref:`qpu_solver_configuration` provide basic
    and advanced information on configuring QPU parameters and best practices.

Configuring Warnings
====================

Settings for raising warnings may be configured by tools such as composites or
the :ref:`index_inspector` tool.

This example configures warnings for an instance of the
:class:`~dwave.system.composites.EmbeddingComposite()` class used on a sampler
structured to represent variable ``a`` with a long chain.

>>> import networkx as nx
>>> import dimod
>>> import dwave.samplers
...
>>> G = nx.Graph()
>>> G.add_edges_from([(n, n + 1) for n in range(10)])
>>> sampler = dimod.StructureComposite(dwave.samplers.SteepestDescentSampler(), G.nodes, G.edges)
>>> sampleset = EmbeddingComposite(sampler).sample_ising({}, {("a", "b"): -1},
...     return_embedding=True,
...     embedding_parameters={"fixed_chains": {"a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}},
...     warnings=dwave.system.warnings.SAVE)
>>> "warnings" in sampleset.info
True

.. currentmodule:: dwave.system.warnings

.. autoclass:: WarningAction
.. autoclass:: WarningHandler

.. autosummary::
    :toctree: generated/

    WarningHandler.chain_break
    WarningHandler.chain_length
    WarningHandler.chain_strength
    WarningHandler.energy_scale
    WarningHandler.too_few_samples
    WarningHandler.issue