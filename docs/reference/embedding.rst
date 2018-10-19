.. _embedding:

=========
Embedding
=========

MinorMiner
==========

:std:doc:`minorminer <minorminer:index>` is a heuristic tool for minor embedding: given a
minor and target graph, it tries to find a mapping that embeds the minor into the target.

.. autosummary::
   :toctree: generated/

   minorminer.find_embedding

Chimera
=======

Functionality for minor-embedding in :term:`Chimera`\ -structured target graphs.

.. currentmodule:: dwave.embedding

.. autosummary::
   :toctree: generated/

   chimera.find_clique_embedding
   chimera.find_biclique_embedding
   chimera.find_grid_embedding
