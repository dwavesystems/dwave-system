.. _included_samplers:

========
Samplers
========

A :term:`sampler` accepts a binary quadratic model (BQM) and returns variable assignments.
Samplers generally try to find minimizing values but can also sample from distributions
defined by the BQM.

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

LeapHybridSampler
=================

.. autoclass:: LeapHybridSampler

Properties
----------

To do: write some stuff here

.. autosummary::
   :toctree: generated/

   LeapHybridSampler.properties
   LeapHybridSampler.parameters


Methods
-------

.. autosummary::
   :toctree: generated/

   LeapHybridSampler.sample
   LeapHybridSampler.sample_ising
   LeapHybridSampler.sample_qubo
