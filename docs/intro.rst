.. _intro_system:

============
Introduction
============

*dwave-system* enables easy incorporation of the D-Wave system as a :term:`sampler`
in either a hybrid quantum-classical solution, using
:class:`~dwave.system.samplers.LeapHybridSampler()` or
:std:doc:`dwave-hybrid <oceandocs:docs_hybrid/sdk_index>` samplers such as
:class:`~hybrid.reference.kerberos.KerberosSampler`, or directly using
:class:`~dwave.system.samplers.DWaveSampler()`.

.. note:: For applications that require detailed control on communication with the remote
    compute resource (a D-Wave QPU or Leap's hybrid solvers), see
    :std:doc:`dwave-cloud-client <oceandocs:docs_cloud/sdk_index>`.

:std:doc:`D-Wave System Documentation <sysdocs_gettingstarted:index>` describes
D-Wave quantum computers and `Leap <https://cloud.dwavesys.com/leap/>`_ hybrid solvers,
including features, parameters, and properties. It also provides guidance
on programming the D-Wave system, including how to formulate problems and configure parameters.

Example
=======
This example solves a small example of a known graph problem, minimum
`vertex cover <https://en.wikipedia.org/wiki/Vertex_cover>`_\ . It uses the NetworkX
graphic package to create the problem, Ocean's :std:doc:`dwave_networkx <oceandocs:docs_dnx/sdk_index>`
to formulate the graph problem as a :term:`BQM`, and dwave-system's
:class:`~dwave.system.samplers.DWaveSampler()` to use a D-Wave system as the sampler.
dwave-system's :class:`~dwave.system.composites.EmbeddingComposite()` handles mapping
between the problem graph to the D-Wave system's numerically indexed qubits,
a mapping known as :term:`minor-embedding`.

>>> import networkx as nx
>>> import dwave_networkx as dnx
>>> from dwave.system import DWaveSampler, EmbeddingComposite
...
>>> s5 = nx.star_graph(4)  # a star graph where node 0 is hub to four other nodes
>>> sampler = EmbeddingComposite(DWaveSampler())
>>> print(dnx.min_vertex_cover(s5, sampler))
[0]
