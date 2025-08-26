.. _system_composites:

==========
Composites
==========

:ref:`dimod composites <concept_samplers_composites>` that provide layers of pre- and
post-processing (e.g., :term:`minor-embedding`) when using the D-Wave system:

.. currentmodule:: dwave.system.composites

Other Ocean packages provide additional composites; for example,
:ref:`dimod <index_dimod>` provides composites that operate
on the problem (e.g., scaling values), track inputs and outputs for debugging,
and other useful functionality relevant to generic samplers.


CutOffs
=======

Prunes the binary quadratic model (BQM) submitted to the child sampler by retaining
only interactions with values commensurate with the sampler’s precision.

.. autosummary::
    :recursive:
    :nosignatures:
    :toctree: generated
    :template: autosummary_class.rst

    CutOffComposite
    PolyCutOffComposite


Embedding
=========

:term:`Minor-embed` a problem :term:`BQM` into a D-Wave system.

.. automodule:: dwave.system.composites.embedding

.. currentmodule:: dwave.system.composites

.. autosummary::
    :recursive:
    :nosignatures:
    :toctree: generated
    :template: autosummary_class.rst

    AutoEmbeddingComposite
    EmbeddingComposite
    FixedEmbeddingComposite
    LazyFixedEmbeddingComposite
    ParallelEmbeddingComposite
    TilingComposite
    VirtualGraphComposite


Linear Bias
===========

.. currentmodule:: dwave.system.composites

Composite for using auxiliary qubits to bias problem qubits.

.. autosummary::
    :recursive:
    :nosignatures:
    :toctree: generated
    :template: autosummary_class.rst

    LinearAncillaComposite


Reverse Anneal
==============

Composites that do batch operations for reverse annealing based on sets of initial
states or anneal schedules.

.. autosummary::
    :recursive:
    :nosignatures:
    :toctree: generated
    :template: autosummary_class.rst

    ReverseBatchStatesComposite
    ReverseAdvanceComposite
