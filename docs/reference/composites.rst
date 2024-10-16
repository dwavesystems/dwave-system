.. _included_composites:

==========
Composites
==========

:std:doc:`dimod composites <oceandocs:docs_dimod/intro/intro_samplers>` that provide layers of pre- and
post-processing (e.g., :term:`minor-embedding`) when using the D-Wave system:

.. currentmodule:: dwave.system.composites

.. contents::
    :local:
    :depth: 2

Other Ocean packages provide additional composites; for example,
:std:doc:`dimod <oceandocs:docs_dimod/sdk_index>` provides composites that operate
on the problem (e.g., scaling values), track inputs and outputs for debugging,
and other useful functionality relevant to generic samplers.

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
   CutOffComposite.parameters
   CutOffComposite.properties

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
   PolyCutOffComposite.parameters
   PolyCutOffComposite.properties

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
   :show-inheritance:

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   FixedEmbeddingComposite.adjacency
   FixedEmbeddingComposite.child
   FixedEmbeddingComposite.children
   FixedEmbeddingComposite.edgelist
   FixedEmbeddingComposite.nodelist
   FixedEmbeddingComposite.parameters
   FixedEmbeddingComposite.properties
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


   LazyFixedEmbeddingComposite.adjacency
   LazyFixedEmbeddingComposite.edgelist
   LazyFixedEmbeddingComposite.nodelist
   LazyFixedEmbeddingComposite.parameters
   LazyFixedEmbeddingComposite.properties
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

   TilingComposite.adjacency
   TilingComposite.child
   TilingComposite.children
   TilingComposite.edgelist
   TilingComposite.embeddings
   TilingComposite.nodelist
   TilingComposite.num_tiles
   TilingComposite.parameters
   TilingComposite.properties
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

   VirtualGraphComposite.adjacency
   VirtualGraphComposite.child
   VirtualGraphComposite.children
   VirtualGraphComposite.edgelist
   VirtualGraphComposite.nodelist
   VirtualGraphComposite.parameters
   VirtualGraphComposite.properties
   VirtualGraphComposite.structure

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   VirtualGraphComposite.sample
   VirtualGraphComposite.sample_ising
   VirtualGraphComposite.sample_qubo



Linear Bias
===========

Composite for using auxiliary qubits to bias problem qubits.


LinearAncillaComposite
-----------------------

.. autoclass:: LinearAncillaComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   LinearAncillaComposite.child
   LinearAncillaComposite.children
   LinearAncillaComposite.parameters
   LinearAncillaComposite.properties

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   LinearAncillaComposite.sample
   LinearAncillaComposite.sample_ising
   LinearAncillaComposite.sample_qubo


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
   ReverseBatchStatesComposite.parameters
   ReverseBatchStatesComposite.properties

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
   ReverseAdvanceComposite.parameters
   ReverseAdvanceComposite.properties

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   ReverseAdvanceComposite.sample
   ReverseAdvanceComposite.sample_ising
   ReverseAdvanceComposite.sample_qubo

