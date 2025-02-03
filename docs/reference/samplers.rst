.. _included_samplers:

========
Samplers
========

A :term:`sampler` accepts a problem in :term:`quadratic model` (e.g., BQM, CQM) or
:term:`nonlinear model` format and returns variable assignments.
Samplers generally try to find minimizing values but can also sample from
distributions defined by the problem.

.. currentmodule:: dwave.system.samplers

.. contents::
    :local:
    :depth: 1

These samplers are non-blocking: the returned :class:`~dimod.SampleSet` is constructed
from a :class:`~concurrent.futures.Future`-like object that is resolved on the first
read of any of its properties; for example, by printing the results. Your code can
query its status with the :meth:`~dimod.SampleSet.done` method or ensure resolution
with the :meth:`~dimod.SampleSet.resolve` method.

Other Ocean packages provide additional samplers; for example,
:std:doc:`dimod <oceandocs:docs_dimod/sdk_index>` provides samplers for testing
your code.

DWaveSampler
============

.. autoclass:: DWaveSampler
   :show-inheritance: 

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
   DWaveSampler.to_networkx_graph
   DWaveSampler.close

DWaveCliqueSampler
==================

.. autoclass:: DWaveCliqueSampler

Properties
----------

.. autosummary::
   :toctree: generated/

   DWaveCliqueSampler.largest_clique_size
   DWaveCliqueSampler.qpu_linear_range
   DWaveCliqueSampler.qpu_quadratic_range
   DWaveCliqueSampler.properties
   DWaveCliqueSampler.parameters
   DWaveCliqueSampler.target_graph


Methods
-------

.. autosummary::
   :toctree: generated/

   DWaveCliqueSampler.largest_clique
   DWaveCliqueSampler.sample
   DWaveCliqueSampler.sample_ising
   DWaveCliqueSampler.sample_qubo
   DWaveCliqueSampler.close


LeapHybridSampler
=================

.. autoclass:: LeapHybridSampler

Properties
----------

.. autosummary::
   :toctree: generated/

   LeapHybridSampler.properties
   LeapHybridSampler.parameters
   LeapHybridSampler.default_solver


Methods
-------

.. autosummary::
   :toctree: generated/

   LeapHybridSampler.sample
   LeapHybridSampler.sample_ising
   LeapHybridSampler.sample_qubo
   LeapHybridSampler.min_time_limit
   LeapHybridSampler.close

LeapHybridCQMSampler
====================

.. autoclass:: LeapHybridCQMSampler

Properties
----------

.. autosummary::
   :toctree: generated/

   LeapHybridCQMSampler.properties
   LeapHybridCQMSampler.parameters


Methods
-------

.. autosummary::
   :toctree: generated/

   LeapHybridCQMSampler.sample_cqm
   LeapHybridCQMSampler.min_time_limit
   LeapHybridCQMSampler.close

LeapHybridNLSampler
====================

.. autoclass:: LeapHybridNLSampler

Properties
----------

.. autosummary::
   :toctree: generated/

   LeapHybridNLSampler.properties
   LeapHybridNLSampler.parameters
   LeapHybridNLSampler.default_solver

Methods
-------

.. autosummary::
   :toctree: generated/

   LeapHybridNLSampler.sample
   LeapHybridNLSampler.estimated_min_time_limit
   LeapHybridNLSampler.close

LeapHybridDQMSampler
====================

.. autoclass:: LeapHybridDQMSampler

Properties
----------

.. autosummary::
   :toctree: generated/

   LeapHybridDQMSampler.properties
   LeapHybridDQMSampler.parameters
   LeapHybridDQMSampler.default_solver


Methods
-------

.. autosummary::
   :toctree: generated/

   LeapHybridDQMSampler.sample_dqm
   LeapHybridDQMSampler.min_time_limit
   LeapHybridDQMSampler.close
