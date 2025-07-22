.. _system_samplers:

========
Samplers
========

A :term:`sampler` accepts a problem in :term:`quadratic model` (e.g., BQM, CQM)
or :term:`nonlinear model` format and returns variable assignments. Samplers
generally try to find minimizing values but can also sample from distributions
defined by the problem.

These samplers are non-blocking: the returned :class:`~dimod.SampleSet` is
constructed from a :class:`~concurrent.futures.Future`-like object that is
resolved on the first read of any of its properties; for example, by printing
the results. Your code can query its status with the
:meth:`~dimod.SampleSet.done` method or ensure resolution with the
:meth:`~dimod.SampleSet.resolve` method.

The :ref:`Ocean <index_ocean_sdk>` SDK provides samplers for various uses.

*   Submitting problems directly to a quantum computer (the ``dwave-system``
    package); for example, the :class:`.DWaveSampler` class
*   Submitting problems the the `Leap <https://cloud.dwavesys.com/leap/>`_
    service's :term:`hybrid` solvers (the ``dwave-system`` package); for
    example, the :class:`.LeapHybridNLSampler` class
*   Testing your code (the :ref:`dimod <index_dimod>` package)
*   Solving problems classically (the :ref:`dwave-samplers <index_samplers>`
    package)

QPU Samplers
============

.. currentmodule:: dwave.system.samplers

.. autosummary::
    :recursive:
    :nosignatures:
    :toctree: generated
    :template: autosummary_class.rst

    DWaveSampler
    DWaveCliqueSampler

Hybrid Solvers
==============

.. autosummary::
    :recursive:
    :nosignatures:
    :toctree: generated
    :template: autosummary_class.rst

    LeapHybridNLSampler
    LeapHybridCQMSampler
    LeapHybridSampler
    LeapHybridDQMSampler