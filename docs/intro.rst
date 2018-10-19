.. _intro:

============
Introduction
============

*dwave-system* enables easy incorporation of the D-Wave system as a
:term:`sampler`---the component used to find variable values that minimize the binary
quadratic model (BQM) representing a problem---in the typical Ocean
:std:doc:`problem-solving procedure <oceandocs:overview/solving_problems>`\ :

1. Formulate the problem as a BQM.
2. Solve the BQM with a sampler.

Example
=======
This example solves a small example of a known graph problem, minimum
`vertex cover <https://en.wikipedia.org/wiki/Vertex_cover>`_\ . It uses the NetworkX
graphic package to create the problem, Ocean's :std:doc:`dwave_networkx <networkx:index>`
to formulate the graph problem as a BQM, and dwave-system's
:class:`~dwave.system.samplers.DWaveSampler()` to use a D-Wave system as the sampler.
(Access to a D-Wave system has been :std:doc:`set up <oceandocs:overview/dwavesys>` in
a configuration file that is used implicitly.) dwave-system's
:class:`~dwave.system.composites.EmbeddingComposite()` handles mapping between the problem graph
to the D-Wave system's numerically indexed qubits, a mapping known as :term:`minor-embedding`.

>>> import networkx as nx
>>> import dwave_networkx as dnx
>>> from dwave.system.samplers import DWaveSampler
>>> from dwave.system.composites import EmbeddingComposite
...
>>> s5 = nx.star_graph(4)  # a star graph where node 0 is hub to four other nodes
>>> sampler = EmbeddingComposite(DWaveSampler())
>>> print(dnx.min_vertex_cover(s5, sampler))
[0]

.. _samplers:

Samplers
========

*Samplers* are processes that sample from low energy states of a problem’s :term:`objective function`.
A BQM sampler samples from low energy states in models such as those
defined by an Ising equation or a Quadratic Unconstrained Binary Optimization (QUBO) problem
and returns an iterable of samples, in order of increasing energy.

Ocean software provides a variety of :std:doc:`dimod samplers <dimod:reference/samplers>`, which
all support ‘sample_qubo’ and ‘sample_ising’ methods as well as the generic BQM sampler method.
In addition to :class:`~dwave.system.samplers.DWaveSampler()`, classical solvers, which run on CPU or GPU, are available and
useful for developing code or on a small versions of a problem to verify code.

.. _composites:

Composites
==========

Samplers can be composed. The `composite pattern <https://en.wikipedia.org/wiki/Composite_pattern>`_
allows layers of pre- and post-processing to be applied to binary quadratic programs without needing
to change the underlying sampler implementation. We refer to these layers as `composites`.
A composed sampler includes at least one sampler and possibly many composites.

Examples of composites are :class:`~dwave.system.composites.EmbeddingComposite()`,
used in the example above, and :class:`~dwave.system.composites.VirtualGraphComposite()`,
both of which handle the mapping known as :term:`minor-embedding`.

Using the D-Wave System as a Sampler
====================================

The :std:doc:`dimod <dimod:index>` API makes it possible to easily interchange samplers
in your code. For example, you might develop code using :std:doc:`dwave_neal <neal:index>`,
Ocean's classical simulated annealing sampler, and then swap in a D-Wave system
composed sampler.

:std:doc:`Using a D-Wave System <oceandocs:overview/dwavesys>` explains how you set up
access to a D-Wave system.

:std:doc:`D-Wave System Documentation <sysdocs_gettingstarted:index>` describes the
D-Wave system, its features, parameters, and properties. The documentation provides guidance
on programming the D-Wave system, including how to formulate problems and configure parameters.

Below one example attribute of the D-Wave system is described. For others and further
information, see the :std:doc:`D-Wave System Documentation <sysdocs_gettingstarted:index>`.

.. _minorEmbedding:

Minor-Embedding
---------------

The D-Wave system is Chimera-structured.

The Chimera architecture comprises sets of connected unit cells, each with four
horizontal qubits connected to four vertical qubits via couplers (bipartite
connectivity). Unit cells are tiled vertically and horizontally with adjacent
qubits connected, creating a lattice of sparsely connected qubits. A unit cell
is typically rendered as either a cross or a column.

.. figure:: _static/ChimeraUnitCell.png
	:align: center
	:name: ChimeraUnitCell
	:scale: 40 %
	:alt: Chimera unit cell.

	Chimera unit cell.

.. figure:: _static/chimera.png
  :name: chimera
  :scale: 70 %
  :alt: Chimera graph.  qubits are arranged in unit cells that form bipartite connections.

  A :math:`3 {\rm x} 3`  Chimera graph, denoted C3. Qubits are arranged in 9 unit cells.

To solve an arbitrarily posed binary quadratic problem on a D-Wave system requires mapping,
called *minor embedding*, to a Chimera graph that represents the system's quantum processing unit.
This preprocessing can be done by a composed sampler consisting of the
:class:`~dwave.system.samplers.DWaveSampler()` and a composite that performs minor-embedding.
