.. _system_embedding:

=========
Embedding
=========

Provides functions that map :term:`binary quadratic model`\ s and samples between
a :term:`source` :term:`graph` and a :term:`target` graph.

For an introduction to :term:`minor-embedding`, see the
:ref:`qpu_embedding_intro` section.

Generators
==========

Tools for finding embeddings.

Generic
-------

:ref:`minorminer <index_minorminer>` is a heuristic tool for minor embedding: given a
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
   pegasus.find_biclique_embedding

Zephyr
-------

Minor-embedding in :term:`Zephyr`-structured target graphs.

.. autosummary::
   :toctree: generated/

   zephyr.find_clique_embedding
   zephyr.find_biclique_embedding

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

Chain Strength
==============

.. automodule:: dwave.embedding.chain_strength
.. currentmodule:: dwave.embedding

.. autosummary::
   :toctree: generated/

   chain_strength.uniform_torque_compensation
   chain_strength.scaled

Chain-Break Resolution
======================

.. automodule:: dwave.embedding.chain_breaks
.. currentmodule:: dwave.embedding

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
   :toctree: generated/

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

Classes
=======

.. autoclass:: EmbeddedStructure
