.. _included_samplers:

========
Samplers
========

A :term:`sampler` is the component used to find variable values that minimize the binary
quadratic model (BQM) representation of a problem.

.. currentmodule:: dwave.system.samplers

DWaveSampler
============

.. autoclass:: DWaveSampler

Properties
----------

For parameters and properties of D-Wave systems, see
:std:doc:`D-Wave System Documentation <sysdocs_gettingstarted:doc_solver_ref>`.

.. autosummary::
   :toctree: generated/

   DWaveSampler.properties
   DWaveSampler.parameters
   DWaveSampler.nodelist
   DWaveSampler.edgelist
   DWaveSampler.adjacency
   DWaveSampler.structure

Methods
-------

.. autosummary::
   :toctree: generated/

   DWaveSampler.sample
   DWaveSampler.sample_ising
   DWaveSampler.sample_qubo
   DWaveSampler.validate_anneal_schedule
