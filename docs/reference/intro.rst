.. _intro:

============
Introduction
============

.. _samplers:

Samplers
========

*Samplers* are processes that sample from low energy states of a problem’s objective function.
A binary quadratic model (BQM) sampler samples from low energy states in models such as those
defined by an Ising equation or a Quadratic Unconstrained Binary Optimization (QUBO) problem
and returns an iterable of samples, in order of increasing energy. A
:std:doc:`dimod sampler <dimod:reference/samplers>` provides ‘sample_qubo’ and
‘sample_ising’ methods as well as the generic BQM sampler method.

.. _composites:

Composites
==========

Samplers can be composed. The `composite pattern <https://en.wikipedia.org/wiki/Composite_pattern>`_
allows layers of pre- and post-processing to be applied to binary quadratic programs without needing
to change the underlying sampler implementation.

We refer to these layers as `composites`. A composed sampler includes at least one sampler
and possibly many composites.

.. _chimera:

D-Wave System Architecture: Chimera
===================================

The D-Wave system is Chimera-structured.

The Chimera architecture comprises sets of connected unit cells, each with four
horizontal qubits connected to four vertical qubits via couplers (bipartite
connectivity). Unit cells are tiled vertically and horizontally with adjacent
qubits connected, creating a lattice of sparsely connected qubits. A unit cell
is typically rendered as either a cross or a column.

.. figure:: ../_static/ChimeraUnitCell.png
	:align: center
	:name: ChimeraUnitCell
	:scale: 40 %
	:alt: Chimera unit cell.

	Chimera unit cell.

.. figure:: ../_static/chimera.png
  :name: chimera
  :scale: 70 %
  :alt: Chimera graph.  qubits are arranged in unit cells that form bipartite connections.

  A :math:`3 {\rm x} 3`  Chimera graph, denoted C3. Qubits are arranged in 9 unit cells.

.. _minorEmbedding:

Minor-Embedding
===============

To solve an arbitrarily posed binary quadratic problem on a D-Wave system requires mapping,
called *minor embedding*, to a Chimera graph that represents the system's quantum processing unit.
This preprocessing can be done by a composed sampler consisting of the DWaveSampler
and a composite that performs minor-embedding.
