.. _embedding_system:

=========
Embedding
=========

Provides functions that map :term:`binary quadratic model`\ s and samples between
a :term:`source` :term`graph` and a :term:`target` graph.

For an introduction to :term:`minor-embedding`, see
:std:doc:`Minor-Embedding <oceandocs:concepts/embedding>`.

Generators
==========

Tools for finding embeddings.

Generic
-------

:std:doc:`minorminer <oceandocs:docs_minorminer/source/sdk_index>` is a heuristic tool for minor embedding: given a
minor and target graph, it tries to find a mapping that embeds the minor into the target.

.. autosummary::
   :toctree: generated/

   minorminer.find_embedding

.. currentmodule:: dwave.embedding

Chimera
-------

Minor-embedding in :term:`Chimera`\ -structured target graphs.

.. autosummary::
   :toctree: generated/

   chimera.find_clique_embedding
   chimera.find_biclique_embedding
   chimera.find_grid_embedding

Pegasus
-------

Minor-embedding in :term:`Pegasus`\ -structured target graphs.

.. autosummary::
   :toctree: generated/

   pegasus.find_clique_embedding

Utilities
=========


.. autosummary::
   :toctree: generated/

   embed_bqm
   embed_ising
   embed_qubo
   unembed_sampleset

Diagnostics
===========

.. autosummary::
   :toctree: generated/

   chain_break_frequency
   diagnose_embedding
   is_valid_embedding
   verify_embedding

Chain-Break Resolution
======================

Handling samples with broken chains when unembedding.

Generators
----------

.. autosummary::
   :toctree: generated/

   chain_breaks.discard
   chain_breaks.majority_vote
   chain_breaks.weighted_random

Callable Objects
----------------

.. autosummary::
   :toctree: minimize_energy

   chain_breaks.MinimizeEnergy

Exceptions
==========

.. autosummary::
   :toctree: generated/

   exceptions.EmbeddingError
   exceptions.MissingChainError
   exceptions.ChainOverlapError
   exceptions.DisconnectedChainError
   exceptions.InvalidNodeError
   exceptions.MissingEdgeError
