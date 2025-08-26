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

The following **modules** provide functions for embedding and chains; for
example, functions such as :func:`~dwave.embedding.transforms.embed_bqm`,
:func:`~dwave.embedding.utils.chain_break_frequency` and
:func:`~dwave.embedding.chain_strength.uniform_torque_compensation`.

.. autosummary::
   :recursive:
   :toctree: generated/
   :template: autosummary_module_functions.rst

   chain_breaks
   chain_strength
   transforms
   utils

Classes
-------

The following classes are provided.

.. autosummary::
    :recursive:
    :nosignatures:
    :toctree: generated
    :template: autosummary_class.rst

    EmbeddedStructure
    ~chain_breaks.MinimizeEnergy

Diagnostic Functions
--------------------

These diagnostics functions are also provided.

.. autosummary::
   :toctree: generated/

   chain_break_frequency
   diagnose_embedding
   is_valid_embedding
   verify_embedding

Exceptions
==========

.. autosummary::
   :toctree: generated/

   exceptions.ChainOverlapError
   exceptions.DisconnectedChainError
   exceptions.EmbeddingError
   exceptions.InvalidNodeError
   exceptions.MissingChainError
   exceptions.MissingEdgeError