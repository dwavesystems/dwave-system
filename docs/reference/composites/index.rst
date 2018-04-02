.. _included_composites:

==========
Composites
==========

Samplers can be composed. The `composite pattern <https://en.wikipedia.org/wiki/Composite_pattern>`_
allows layers of pre- and post-processing to be applied to binary quadratic programs without needing
to change the underlying sampler implementation.

We refer to these layers as `composites`. A composed sampler includes at least one sampler
and possibly many composites.

`dwave-system` provides
`dimod composites <http://dimod.readthedocs.io/en/latest/reference/samplers.html#samplers-and-composites>`_
for using the D-Wave system.

For example, the D-Wave system is Chimera-structured (a particular architecture of sparsely
connected qubits) and so any arbitrarily posed binary quadratic problem requires mapping,
called *minor embedding*, to a Chimera graph that represents the system's quantum processing unit.
This preprocessing can be done by a composed sampler consisting of the DWaveSampler
and a composite that performs minor-embedding.

   :Release: |release|
   :Date: |today|

.. toctree::
   :maxdepth: 2

   embedding
   tiling
   virtual_graph
