.. _dwave_sampler:

==============
D-Wave Sampler
==============

.. automodule:: dwave.system.samplers.dwave_sampler

.. currentmodule:: dwave.system.samplers

Class
=====

.. autoclass:: DWaveSampler

Sampler Properties
==================

:std:doc:`D-Wave System Documentation <sysdocs_gettingstarted:doc_solver_ref>` lists and describes
the parameters and properties of D-Wave systems.

.. autosummary::
   :toctree: generated/

   DWaveSampler.properties
   DWaveSampler.parameters
   DWaveSampler.nodelist
   DWaveSampler.edgelist
   DWaveSampler.adjacency
   DWaveSampler.structure

Sample Methods
==============

.. autosummary::
   :toctree: generated/

   DWaveSampler.sample
   DWaveSampler.sample_ising
   DWaveSampler.sample_qubo

Methods
=======

.. autosummary::
   :toctree: generated/

   DWaveSampler.validate_anneal_schedule
