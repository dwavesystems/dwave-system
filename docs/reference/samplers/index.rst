.. _included_samplers:

========
Samplers
========

*Samplers* are processes that sample from low energy states of a problem’s objective function.
A binary quadratic model (BQM) sampler samples from low energy states in models such as those
defined by an Ising equation or a Quadratic Unconstrained Binary Optimization (QUBO) problem
and returns an iterable of samples, in order of increasing energy. A
`dimod sampler <http://dimod.readthedocs.io/en/latest/reference/samplers.html#samplers-and-composites>`_
provides ‘sample_qubo’ and ‘sample_ising’ methods as well as the generic BQM sampler method.

`dwave-system` provides dimod samplers for using the D-Wave system.

   :Release: |release|
   :Date: |today|

.. toctree::
   :maxdepth: 2

   dwave_sampler
