.. _included_composites:

==========
Composites
==========

Samplers can be composed. The `composite pattern <https://en.wikipedia.org/wiki/Composite_pattern>`_
allows layers of pre- and post-processing to be applied to binary quadratic programs without needing
to change the underlying sampler implementation.

We refer to these layers as `composites`. Each composed sampler must
include at least one sampler, and possibly many composites.

dwave-system provides
`dimod composites <http://dimod.readthedocs.io/en/latest/reference/samplers.html#samplers-and-composites>`_
that can use the D-Wave system.

For example, the D-Wave system is Chimera-structured (a particular architecture of sparsely
connected qubits) and so requires a mapping, called *minor embedding*, to any arbitrarily posed
binary quadratic problem. This preprocessing can be accomplished by a composed sampler consisting of
the DWaveSampler and a composite that performs minor-embedding.  

   :Release: |release|
   :Date: |today|

.. toctree::
   :maxdepth: 2

   embedding
   tiling
   virtual_graph
