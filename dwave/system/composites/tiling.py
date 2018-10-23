# Copyright 2018 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# ================================================================================================
"""
A :std:doc:`dimod composite <dimod:reference/samplers>` that tiles small problems
multiple times to a Chimera-structured sampler.

The :class:`.TilingComposite` takes a problem that can fit on a small
:term:`Chimera` graph and replicates it across a larger
Chimera graph to obtain samples from multiple areas of the solver in one call.
For example, a 2x2 Chimera lattice could be tiled 64 times (8x8) on a fully-yielded
D-Wave 2000Q system (16x16).

See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_ for explanations
of technical terms in descriptions of Ocean tools.

"""
from __future__ import division
from math import sqrt, ceil

import dimod
import dwave_networkx as dnx

import numpy as np

__all__ = ['TilingComposite']

class TilingComposite(dimod.Sampler, dimod.Composite, dimod.Structured):
    """Composite to tile a small problem across a Chimera-structured sampler.

    Enables parallel sampling for small problems (problems that are minor-embeddable in
    a small part of a D-Wave solver's :term:`Chimera` graph).

    Notation *CN* refers to a Chimera graph consisting of an NxN grid of unit cells, where
    each unit cell is a bipartite graph with shores of size t. The D-Wave 2000Q QPU
    supports a C16 Chimera graph: its 2048 qubits are logically mapped into a 16x16 matrix of
    unit cell of 8 qubits (t=4).

    A problem that can be minor-embedded in a single unit cell, for example, can therefore
    be tiled across the unit cells of a D-Wave 2000Q as 16x16 duplicates. This enables
    sampling 256 solutions in a single call.

    Args:
       sampler (:class:`dimod.Sampler`): Structured dimod sampler such as a :class:`~dwave.system.samplers.DWaveSampler()`.
       sub_m (int): Number of rows of Chimera unit cells for minor-embedding the problem once.
       sub_n (int): Number of columns of Chimera unit cells for minor-embedding the problem once.
       t (int, optional, default=4): Size of the shore within each Chimera unit cell.

    Examples:
       This example submits a two-variable QUBO problem representing a logical NOT gate
       to a D-Wave system selected by the user's default
       :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`.
       The QUBO---two nodes with biases of -1 that are coupled with strength 2---needs
       only any two coupled qubits and so is easily minor-embedded in a single unit cell.
       Composite :class:`.TilingComposite` tiles it multiple times for parallel solution:
       the two nodes should typically have opposite values.

       >>> from dwave.system.samplers import DWaveSampler
       >>> from dwave.system.composites import EmbeddingComposite
       >>> from dwave.system.composites import TilingComposite
       ...
       >>> sampler = EmbeddingComposite(TilingComposite(DWaveSampler(), 1, 1, 4))
       >>> Q = {(1, 1): -1, (1, 2): 2, (2, 1): 0, (2, 2): -1}
       >>> response = sampler.sample_qubo(Q)
       >>> response.first    # doctest: +SKIP
       Sample(sample={1: 0, 2: 1}, energy=-1.0, num_occurrences=1, chain_break_fraction=0.0)

    See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_
    for explanations of technical terms in descriptions of Ocean tools.

    """
    nodelist = None
    """list: List of active qubits for the structured solver.

    Examples:
       >>> sampler_tile = TilingComposite(DWaveSampler(), 2, 1, 4)
       >>> sampler_tile.nodelist   # doctest: +SKIP
       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    """

    edgelist = None
    """list: List of active couplers for the D-Wave solver.

    Examples:
       >>> sampler_tile = TilingComposite(DWaveSampler(), 1, 2, 4)
       >>> len(sampler_tile.edgelist)
       36
    """

    parameters = None
    """dict[str, list]: Parameters in the form of a dict.

    See :obj:`.EmbeddingComposite.parameters` for detailed information.
    """

    properties = None
    """dict: Properties in the form of a dict.

    See :obj:`.EmbeddingComposite.properties` for detailed information.
    """

    children = None
    """list: The single wrapped structured sampler.

    See :obj:`.EmbeddingComposite.children` for detailed information.
    """

    def __init__(self, sampler, sub_m, sub_n, t=4):

        self.parameters = sampler.parameters.copy()
        self.properties = properties = {'child_properties': sampler.properties}

        tile = dnx.chimera_graph(sub_m, sub_n, t)
        self.nodelist = sorted(tile.nodes)
        self.edgelist = sorted(sorted(edge) for edge in tile.edges)
        # dimod.Structured abstract base class automatically populates adjacency and structure as
        # mixins based on nodelist and edgelist

        if not isinstance(sampler, dimod.Structured):
            # we could also just tile onto the unstructured sampler but in that case we would need
            # to know how many tiles to use
            raise ValueError("given child sampler should be structured")
        self.children = [sampler]

        nodes_per_cell = t * 2
        edges_per_cell = t * t
        m = n = int(ceil(sqrt(ceil(len(sampler.structure.nodelist) / nodes_per_cell))))  # assume square lattice shape
        system = dnx.chimera_graph(m, n, t, node_list=sampler.structure.nodelist, edge_list=sampler.structure.edgelist)
        c2i = {chimera_index: linear_index for (linear_index, chimera_index) in system.nodes(data='chimera_index')}
        sub_c2i = {chimera_index: linear_index for (linear_index, chimera_index) in tile.nodes(data='chimera_index')}

        # Count the connections between these qubits
        def _between(qubits1, qubits2):
            edges = [edge for edge in system.edges if edge[0] in qubits1 and edge[1] in qubits2]
            return len(edges)

        # Get the list of qubits in a cell
        def _cell_qubits(i, j):
            return [c2i[(i, j, u, k)] for u in range(2) for k in range(t) if (i, j, u, k) in c2i]

        # get a mask of complete cells
        cells = [[False for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                qubits = _cell_qubits(i, j)
                cells[i][j] = len(qubits) == nodes_per_cell and _between(qubits, qubits) == edges_per_cell

        # List of 'embeddings'
        self.embeddings = properties['embeddings'] = embeddings = []

        # For each possible chimera cell check if the next few cells are complete
        for i in range(m + 1 - sub_m):
            for j in range(n + 1 - sub_n):

                # Check if the sub cells are matched
                match = all(cells[i + sub_i][j + sub_j] for sub_i in range(sub_m) for sub_j in range(sub_n))

                # Check if there are connections between the cells.
                for sub_i in range(sub_m):
                    for sub_j in range(sub_n):
                        if sub_m > 1 and sub_i < sub_m - 1:
                            match &= _between(_cell_qubits(i + sub_i, j + sub_j),
                                              _cell_qubits(i + sub_i + 1, j + sub_j)) == t
                        if sub_n > 1 and sub_j < sub_n - 1:
                            match &= _between(_cell_qubits(i + sub_i, j + sub_j),
                                              _cell_qubits(i + sub_i, j + sub_j + 1)) == t

                if match:
                    # Pull those cells out into an embedding.
                    embedding = {}
                    for sub_i in range(sub_m):
                        for sub_j in range(sub_n):
                            cells[i + sub_i][j + sub_j] = False  # Mark cell as matched
                            for u in range(2):
                                for k in range(t):
                                    embedding[sub_c2i[sub_i, sub_j, u, k]] = {c2i[(i + sub_i, j + sub_j, u, k)]}

                    embeddings.append(embedding)

        if len(embeddings) == 0:
            raise ValueError("no tile embeddings found; is the sampler Chimera structured?")

    @dimod.bqm_structured
    def sample(self, bqm, **kwargs):
        """Sample from the specified binary quadratic model.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            **kwargs:
                Optional keyword arguments for the sampling method, specified per solver.

        Returns:
            :class:`dimod.Response`: A `dimod` :obj:`~dimod.Response` object.

        Examples:
            This example submits a simple Ising problem of just two variables on a
            D-Wave system selected by the user's default
            :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`.
            Because the problem fits in a single :term:`Chimera` unit cell, it is tiled
            across the solver's entire Chimera graph, resulting in multiple samples
            (the exact number depends on the working Chimera graph of the D-Wave system).

            >>> from dwave.system.samplers import DWaveSampler
            >>> from dwave.system.composites import EmbeddingComposite
            >>> from dwave.system.composites import EmbeddingComposite, TilingComposite
            ...
            >>> sampler = EmbeddingComposite(TilingComposite(DWaveSampler(), 1, 1, 4))
            >>> response = sampler.sample_ising({},{('a', 'b'): 1})
            >>> len(response)    # doctest: +SKIP
            246

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_
        for explanations of technical terms in descriptions of Ocean tools.

        """

        # apply the embeddings to the given problem to tile it across the child sampler
        embedded_bqm = dimod.BinaryQuadraticModel.empty(bqm.vartype)
        __, __, target_adjacency = self.child.structure
        for embedding in self.embeddings:
            embedded_bqm.update(dimod.embed_bqm(bqm, embedding, target_adjacency))

        # solve the problem on the child system
        tiled_response = self.child.sample(embedded_bqm, **kwargs)

        responses = []

        for embedding in self.embeddings:
            embedding = {v: chain for v, chain in embedding.items() if v in bqm.linear}

            responses.append(dimod.unembed_response(tiled_response, embedding, bqm))

        # stack the records
        record = np.rec.array(np.hstack((resp.record for resp in responses)))

        vartypes = set(resp.vartype for resp in responses)
        if len(vartypes) > 1:
            raise RuntimeError("inconsistent vartypes returned")
        vartype = vartypes.pop()

        info = {}
        for resp in responses:
            info.update(resp.info)

        labels = responses[0].variable_labels

        return dimod.Response(record, labels, info, vartype)

    @property
    def num_tiles(self):
        return len(self.embeddings)
