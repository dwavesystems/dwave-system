.. 
    Copyright 2016 D-Wave Systems Inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.


Documentation for DWave Chimera Embedding Algorithms
====================================================

This is the documentation for the :py:mod:`chimera_embedding` module.  The module
contains a Python implementation of the polynomial-time algorithm described in [BKR]_.
Additionally, we provide Python examples for how to use the :py:mod:`chimera_embedding`
module to generate embeddings as well as Python, C and Matlab examples for how to use
those embeddings to solve problems.

The algorithm described in [BKR]_ is useful for generating embeddings for 
completely-connected Ising spin glass problems (in graph theoretical terms, the algorithm
finds clique minors in subgraphs of Chimera graphs).  A simpler problem is to find complete bipartite
minors, and that is also implemented in the same module.

The embeddings produced in this module are called *native* embeddings, in part
because the Chimera graph structure was designed to support a specific clique
embedding, described in [C]_.  The embeddings we call *native* have all chains
of equal length, and minimize that chain length parameter for a given clique 
size.  The :py:mod:`chimera_embedding` module is particularly effective where
some qubits have been removed from a particular Chimera graph, but runtime can
suffer when certain couplers are removed.  Details can be found in the 
:ref:`polynomialembedder` documentation.

Contents
========

.. toctree::
   :maxdepth: 2

   polynomialembedder
   embed_clique
   solve_clique
   embed_biclique
   solve_biclique



References
==========


.. [BKR] Tomas Boothby, Andrew D. King, Aidan Roy.  Fast clique minor 
    generation in Chimera qubit connectivity graphs. Quantum Information
    Processing, (2015).  http://arxiv.org/abs/1507.04774

.. [C] Vicky Choi.  Minor-embedding in adiabatic quantum computation: II.
    Minor-universal graph design.  Quantum Information Processing 10(3),
    343-353 (2011).
