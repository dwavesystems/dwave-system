.. _included_composites:

==========
Composites
==========

:std:doc:`dimod composites <dimod:introduction>` that provide layers of pre- and
post-processing (e.g., :term:`minor-embedding`) when using the D-Wave system.

.. currentmodule:: dwave.system.composites

.. contents::
    :depth: 3

CutOffs
=======

Prunes the binary quadratic model (BQM) submitted to the child sampler by retaining
only interactions with values commensurate with the sampler’s precision.

CutOffComposite
---------------

.. autoclass:: CutOffComposite


Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   CutOffComposite.child
   CutOffComposite.children
   CutOffComposite.properties
   CutOffComposite.parameters


Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   CutOffComposite.sample
   CutOffComposite.sample_ising
   CutOffComposite.sample_qubo

PolyCutOffComposite
-------------------

Prunes the polynomial submitted to the child sampler by retaining
only interactions with values commensurate with the sampler’s precision.

.. autoclass:: PolyCutOffComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   PolyCutOffComposite.child
   PolyCutOffComposite.children
   PolyCutOffComposite.properties
   PolyCutOffComposite.parameters


Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   PolyCutOffComposite.sample_poly
   PolyCutOffComposite.sample_hising
   PolyCutOffComposite.sample_hubo



Embedding
=========

:term:`Minor-embed` a problem :term:`BQM` into a D-Wave system.

.. automodule:: dwave.system.composites.embedding

.. currentmodule:: dwave.system.composites

AutoEmbeddingComposite
----------------------

.. autoclass:: AutoEmbeddingComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   AutoEmbeddingComposite.child
   AutoEmbeddingComposite.parameters
   AutoEmbeddingComposite.properties

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   AutoEmbeddingComposite.sample
   AutoEmbeddingComposite.sample_ising
   AutoEmbeddingComposite.sample_qubo


EmbeddingComposite
------------------

.. autoclass:: EmbeddingComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   EmbeddingComposite.child
   EmbeddingComposite.parameters
   EmbeddingComposite.properties
   EmbeddingComposite.return_embedding_default
   EmbeddingComposite.warnings_default

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   EmbeddingComposite.sample
   EmbeddingComposite.sample_ising
   EmbeddingComposite.sample_qubo


FixedEmbeddingComposite
-----------------------

.. autoclass:: FixedEmbeddingComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   FixedEmbeddingComposite.properties
   FixedEmbeddingComposite.parameters
   FixedEmbeddingComposite.children
   FixedEmbeddingComposite.child
   FixedEmbeddingComposite.nodelist
   FixedEmbeddingComposite.edgelist
   FixedEmbeddingComposite.adjacency
   FixedEmbeddingComposite.structure

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   FixedEmbeddingComposite.sample
   FixedEmbeddingComposite.sample_ising
   FixedEmbeddingComposite.sample_qubo


LazyFixedEmbeddingComposite
---------------------------

.. autoclass:: LazyFixedEmbeddingComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/


   LazyFixedEmbeddingComposite.parameters
   LazyFixedEmbeddingComposite.properties
   LazyFixedEmbeddingComposite.nodelist
   LazyFixedEmbeddingComposite.edgelist
   LazyFixedEmbeddingComposite.adjacency
   LazyFixedEmbeddingComposite.structure

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   LazyFixedEmbeddingComposite.sample
   LazyFixedEmbeddingComposite.sample_ising
   LazyFixedEmbeddingComposite.sample_qubo

TilingComposite
---------------

.. autoclass:: TilingComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   TilingComposite.properties
   TilingComposite.parameters
   TilingComposite.children
   TilingComposite.child
   TilingComposite.nodelist
   TilingComposite.edgelist
   TilingComposite.adjacency
   TilingComposite.structure

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   TilingComposite.sample
   TilingComposite.sample_ising
   TilingComposite.sample_qubo

VirtualGraphComposite
---------------------

.. autoclass:: VirtualGraphComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   VirtualGraphComposite.properties
   VirtualGraphComposite.parameters
   VirtualGraphComposite.children
   VirtualGraphComposite.child
   VirtualGraphComposite.nodelist
   VirtualGraphComposite.edgelist
   VirtualGraphComposite.adjacency
   VirtualGraphComposite.structure

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   VirtualGraphComposite.sample
   VirtualGraphComposite.sample_ising
   VirtualGraphComposite.sample_qubo

Reverse Anneal
==============

Composites that do batch operations for reverse annealing based on sets of initial
states or anneal schedules.

ReverseBatchStatesComposite
---------------------------

.. autoclass:: ReverseBatchStatesComposite


Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ReverseBatchStatesComposite.child
   ReverseBatchStatesComposite.children
   ReverseBatchStatesComposite.properties
   ReverseBatchStatesComposite.parameters


Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   ReverseBatchStatesComposite.sample
   ReverseBatchStatesComposite.sample_ising
   ReverseBatchStatesComposite.sample_qubo

ReverseAdvanceComposite
-----------------------

.. autoclass:: ReverseAdvanceComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ReverseAdvanceComposite.child
   ReverseAdvanceComposite.children
   ReverseAdvanceComposite.properties
   ReverseAdvanceComposite.parameters


Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   ReverseAdvanceComposite.sample
   ReverseAdvanceComposite.sample_ising
   ReverseAdvanceComposite.sample_qubo
