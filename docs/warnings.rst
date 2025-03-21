.. _system_warnings:

========
Warnings
========

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

.. autoclass:: ChainBreakWarning
.. autoclass:: ChainLengthWarning
.. autoclass:: ChainStrengthWarning
.. autoclass:: EnergyScaleWarning
.. autoclass:: TooFewSamplesWarning
.. autoclass:: WarningAction
.. autoclass:: WarningHandler
