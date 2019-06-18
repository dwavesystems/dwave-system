.. _included_composites:

==========
Composites
==========

`dwave~system` provides :std:doc:`dimod composites <dimod:reference/samplers>`
for using the D~Wave system.

.. currentmodule:: dwave.system.composites

.. contents::
    :depth: 3

CutOffs
=======

CutOffComposite
---------------

Class
~~~~~
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

Class
~~~~~

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

AutoEmbeddingComposite
----------------------

Class
~~~~~

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

Class
~~~~~

.. autoclass:: EmbeddingComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   EmbeddingComposite.child
   EmbeddingComposite.parameters
   EmbeddingComposite.properties

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   EmbeddingComposite.sample
   EmbeddingComposite.sample_ising
   EmbeddingComposite.sample_qubo


FixedEmbeddingComposite
-----------------------

Class
~~~~~

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

Class
~~~~~

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

Class
~~~~~

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

Class
~~~~~

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
