.. _system_samplers:

========
Samplers
========

A :term:`sampler` accepts a problem in :term:`quadratic model` (e.g., BQM, CQM) or
:term:`nonlinear model` format and returns variable assignments.
Samplers generally try to find minimizing values but can also sample from
distributions defined by the problem.

.. currentmodule:: dwave.system.samplers

These samplers are non-blocking: the returned :class:`~dimod.SampleSet` is constructed
from a :class:`~concurrent.futures.Future`-like object that is resolved on the first
read of any of its properties; for example, by printing the results. Your code can
query its status with the :meth:`~dimod.SampleSet.done` method or ensure resolution
with the :meth:`~dimod.SampleSet.resolve` method.

Other Ocean packages provide additional samplers; for example,
:ref:`dimod <index_dimod>` provides samplers for testing
your code.

DWaveSampler
============

.. autoclass:: DWaveSampler
    :show-inheritance:

Properties
----------

For parameters and properties of D-Wave systems, see the
:ref:`qpu_index_solver_properties` and :ref:`qpu_solver_parameters` sections.

.. autosummary::
    :toctree: generated/

    ~DWaveSampler.adjacency
    ~DWaveSampler.edgelist
    ~DWaveSampler.nodelist
    ~DWaveSampler.parameters
    ~DWaveSampler.properties
    ~DWaveSampler.structure
    ~DWaveSampler.warnings_default

Methods
-------

.. autosummary::
    :toctree: generated/

    ~DWaveSampler.close
    ~DWaveSampler.remove_unknown_kwargs
    ~DWaveSampler.sample
    ~DWaveSampler.sample_ising
    ~DWaveSampler.sample_qubo
    ~DWaveSampler.to_networkx_graph
    ~DWaveSampler.trigger_failover
    ~DWaveSampler.valid_bqm_graph
    ~DWaveSampler.validate_anneal_schedule


DWaveCliqueSampler
==================

.. autoclass:: DWaveCliqueSampler

Properties
----------

.. autosummary::
    :toctree: generated/

    ~DWaveCliqueSampler.largest_clique_size
    ~DWaveCliqueSampler.parameters
    ~DWaveCliqueSampler.properties
    ~DWaveCliqueSampler.qpu_linear_range
    ~DWaveCliqueSampler.qpu_quadratic_range
    ~DWaveCliqueSampler.target_graph

Methods
-------

.. autosummary::
    :toctree: generated/

    ~DWaveCliqueSampler.clique
    ~DWaveCliqueSampler.close
    ~DWaveCliqueSampler.largest_clique
    ~DWaveCliqueSampler.remove_unknown_kwargs
    ~DWaveCliqueSampler.sample
    ~DWaveCliqueSampler.sample_ising
    ~DWaveCliqueSampler.sample_qubo
    ~DWaveCliqueSampler.trigger_failover


LeapHybridSampler
=================

.. autoclass:: LeapHybridSampler

Properties
----------

.. autosummary::
    :toctree: generated/

    ~LeapHybridSampler.default_solver
    ~LeapHybridSampler.parameters
    ~LeapHybridSampler.properties

Methods
-------

.. autosummary::
    :toctree: generated/

    ~LeapHybridSampler.close
    ~LeapHybridSampler.min_time_limit
    ~LeapHybridSampler.remove_unknown_kwargs
    ~LeapHybridSampler.sample
    ~LeapHybridSampler.sample_ising
    ~LeapHybridSampler.sample_qubo


LeapHybridCQMSampler
====================

.. autoclass:: LeapHybridCQMSampler

Properties
----------

.. autosummary::
    :toctree: generated/

    ~LeapHybridCQMSampler.default_solver
    ~LeapHybridCQMSampler.properties
    ~LeapHybridCQMSampler.parameters

Methods
-------

.. autosummary::
    :toctree: generated/

    ~LeapHybridCQMSampler.close
    ~LeapHybridCQMSampler.min_time_limit
    ~LeapHybridCQMSampler.sample_cqm


LeapHybridNLSampler
====================

.. autoclass:: LeapHybridNLSampler

Properties
----------

.. autosummary::
    :toctree: generated/

    ~LeapHybridNLSampler.default_solver
    ~LeapHybridNLSampler.parameters
    ~LeapHybridNLSampler.properties

Methods
-------

.. autosummary::
    :toctree: generated/

    ~LeapHybridNLSampler.close
    ~LeapHybridNLSampler.estimated_min_time_limit
    ~LeapHybridNLSampler.sample


LeapHybridDQMSampler
====================

.. autoclass:: LeapHybridDQMSampler

Properties
----------

.. autosummary::
    :toctree: generated/

    ~LeapHybridDQMSampler.default_solver
    ~LeapHybridDQMSampler.parameters
    ~LeapHybridDQMSampler.properties

Methods
-------

.. autosummary::
    :toctree: generated/

    ~LeapHybridDQMSampler.close
    ~LeapHybridDQMSampler.min_time_limit
    ~LeapHybridDQMSampler.sample_dqm
