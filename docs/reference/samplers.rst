.. _included_samplers:

========
Samplers
========

.. currentmodule:: dwave.system.samplers

.. contents::
    :depth: 3

QPU
===

DWaveSampler
------------

Class
~~~~~

.. autoclass:: DWaveSampler

Properties
~~~~~~~~~~

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

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   DWaveSampler.sample
   DWaveSampler.sample_ising
   DWaveSampler.sample_qubo
   DWaveSampler.validate_anneal_schedule
