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

"""
A :std:doc:dimod composite <oceandocs:docs_dimod/reference/samplers> that tiles 
small problems multiple times to a structured sampler.

The :class:.TilingComposite class takes a problem that can fit on a small
:term:Chimera graph and replicates it across a larger Pegasus or
Chimera graph to obtain samples from multiple areas of the solver in one call.
For example, a 2x2 Chimera lattice could be tiled 64 times (8x8) on a 
fully-yielded D-Wave 2000Q system (16x16).

See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/stable/concepts/index.html>`_
for explanations of technical terms in descriptions of Ocean tools.

"""

from math import sqrt, ceil

import dimod
import dwave_networkx as dnx
import numpy as np

import dwave.embedding

__all__ = ['TilingComposite']


class TilingComposite(dimod.Sampler, dimod.Composite, dimod.Structured):
    """Composite to tile a small problem across a structured sampler.



    Enables parallel sampling on Chimera or Pegasus structured samplers of 
    small problems. The small problem should be defined on a :term:`Chimera` 
    graph of dimensions ``sub_m``, ``sub_n``, ``t``, or minor-embeddable to 
    such a graph.

    Notation *CN* refers to a Chimera graph consisting of an NxN grid of unit 
    cells, where each unit cell is a bipartite graph with shores of size t. 
    The D-Wave 2000Q QPU supports a C16 Chimera graph: its 2048 qubits are 
    logically mapped into a 16x16 matrix of unit cells of 8 qubits (t=4). 
    See also :func:dwave_networkx.chimera_graph 

    Notation *PN* referes to a Pegasus graph consisting of a 3x(N-1)x(N-1) grid 
    of cells, where each unit cell is a bipartite graph with shore of size t, 
    supplemented with odd couplers (see nice_coordinate definition). The 
    Advantage QPU supports a P16 Pegasus graph: its qubits may be mapped to a 
    3x15x15 matrix of unit cells, each of 8 qubits. This code supports tiling of
    Chimera-structured problems, with an option of additional odd-couplers,
    onto Pegasus. See also :func:dwave_networkx.pegasus_graph .

    A problem that can be minor-embedded in a single chimera unit cell, for 
    example, can therefore be tiled across the unit cells of a D-Wave 2000Q as 
    16x16 duplicates (or Advantage as 3x15x15 duplicates), subject to solver
    yield. This enables up to 256 (625) parallel samples per read.

    Args:
       sampler (:class:`dimod.Sampler`): Structured dimod sampler such as a 
           :class:`~dwave.system.samplers.DWaveSampler()`.
       sub_m (int): Minimum number of Chimera unit cell rows required for
           minor-embedding a single instance of the problem.
       sub_n (int): Minimum number of Chimera unit cell columns required for 
           minor-embedding a single instance of the problem.
       t (int, optional, default=4): Size of the shore within each Chimera unit 
           cell.

    Examples:
       This example submits a two-variable QUBO problem representing a logical
       NOT gate to a D-Wave system. The QUBO---two nodes with biases of -1 that
       are coupled with strength 2---needs only any two coupled qubits and so is
       easily minor-embedded in a single unit cell.
       Composite :class:`.TilingComposite` tiles it multiple times for parallel solution:
       the two nodes should typically have opposite values.

       >>> from dwave.system import DWaveSampler, EmbeddingComposite
       >>> from dwave.system import TilingComposite
       ...
       >>> qpu_2000q = DWaveSampler(solver={'topology__type': 'chimera'})
       >>> sampler = EmbeddingComposite(TilingComposite(qpu_2000q, 1, 1, 4))
       >>> Q = {(1, 1): -1, (1, 2): 2, (2, 1): 0, (2, 2): -1}
       >>> sampleset = sampler.sample_qubo(Q)
       >>> len(sampleset)> 1
       True

    See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/stable/concepts/index.html>`_
    for explanations of technical terms in descriptions of Ocean tools.

    """
    nodelist = None
    """list: List of active qubits for the structured solver."""

    edgelist = None
    """list: List of active couplers for the D-Wave solver."""

    parameters = None
    """dict[str, list]: Parameters in the form of a dict."""

    properties = None
    """dict: Properties in the form of a dict."""

    children = None
    """list: The single wrapped structured sampler."""

    def __init__(self, sampler, sub_m, sub_n, t=4):

        self.parameters = sampler.parameters.copy()
        self.properties = properties = {'child_properties': sampler.properties}

        tile = dnx.chimera_graph(sub_m, sub_n, t)
        self.nodelist = sorted(tile.nodes)
        self.edgelist = sorted(sorted(edge) for edge in tile.edges)
        # dimod.Structured abstract base class automatically populates adjacency
        # and structure as mixins based on nodelist and edgelist

        if not isinstance(sampler, dimod.Structured):
            # we could also just tile onto the unstructured sampler but in that
            # case we would need to know how many tiles to use
            raise ValueError("given child sampler should be structured")
        self.children = [sampler]
        # Chimera values (unless pegasus specified)
        num_sublattices=1
        nodes_per_cell = t * 2
        edges_per_cell = t * t
        if not ('topology' in sampler.properties
            and 'type' in sampler.properties['topology']
            and 'shape' in sampler.properties['topology']):
            raise ValueError('To use this composite it is necessary for the'
                             'structured sampler to have an explicit topology'
                             '(sampler.properties[\'topology\']). Necessary'
                             'fields are \'type\' and \'shape\'. ')
        if sampler.properties['topology']['type'] == 'chimera':
            if len(sampler.properties['topology']['shape']) != 3: 
                raise ValueError('topology shape is not of length 3 '
                                 '(not compatible with chimera)')
            if sampler.properties['topology']['shape'][2] != t: 
                raise ValueError('Tiling methodology requires that solver'
                                 'and subproblem have identical shore size')
            m = sampler.properties['topology']['shape'][0]
            n = sampler.properties['topology']['shape'][1]
        else:
            if len(sampler.properties['topology']['shape']) != 1: 
                raise ValueError('topology shape is not of length 1 '
                                 '(not compatible with pegasus)')
            # Full yield in odd-couplers also required.
            # Generalizes chimera subgraph requirement and leads to some 
            # simplification of expressions, but at with a cost in cell-yield
            edges_per_cell += t
            # Square solvers only by pegasus lattice definition PN yields
            # 3 by N-1 by N-1 cells:
            num_sublattices=3
            m = n = sampler.properties['topology']['shape'][0] - 1
            if t!=4:
                raise ValueError(
                    't=4 for all pegasus processors, value is not typically'
                    'stored in solver properties and is difficult to infer.'
                    'Therefore only the value t=4 is supported.')
             
        
        if num_sublattices==1:
            # Chimera defaults. Appended coordinates (treat as first and only sublattice)
            system = dnx.chimera_graph(m, n, t,
                                       node_list=sampler.structure.nodelist,
                                       edge_list=sampler.structure.edgelist)
            
            c2i = {(0, *chimera_index) : linear_index
                   for (linear_index, chimera_index)
                   in system.nodes(data='chimera_index')}
        else:
            system = dnx.pegasus_graph(m,
                                       node_list=sampler.structure.nodelist,
                                       edge_list=sampler.structure.edgelist)
            # Vector specification in terms of nice coordinates:
            c2i = {dnx.pegasus_coordinates(m+1).linear_to_nice(linear_index):
                   linear_index for linear_index in system.nodes()}
        
        sub_c2i = {chimera_index: linear_index for (linear_index, chimera_index)
                   in tile.nodes(data='chimera_index')}
        
        # Count the connections between these qubits
        def _between(qubits1, qubits2):
            edges = [edge for edge in system.edges if edge[0] in qubits1
                     and edge[1] in qubits2]
            return len(edges)
        
        # Get the list of qubits in a cell
        def _cell_qubits(s, i, j):
            return [c2i[(s, i, j, u, k)] for u in range(2) for k in range(t)
                    if (s, i, j, u, k) in c2i]

        # get a mask of complete cells
        cells = [[[False for _ in range(n)] for _ in range(m)]
                 for _ in range(num_sublattices)]
        
        for s in range(num_sublattices):
            for i in range(m):
                for j in range(n):
                    qubits = _cell_qubits(s, i, j)
                    cells[s][i][j] = (
                        len(qubits) == nodes_per_cell
                        and _between(qubits, qubits) == edges_per_cell)
                    
        # List of 'embeddings'
        self.embeddings = properties['embeddings'] = embeddings = []

        # For each possible chimera cell check if the next few cells are complete
        for s in range(num_sublattices):
            for i in range(m + 1 - sub_m):
                for j in range(n + 1 - sub_n):
                    
                    # Check if the sub cells are matched
                    match = all(cells[s][i + sub_i][j + sub_j]
                                for sub_i in range(sub_m)
                                for sub_j in range(sub_n))
                    
                    # Check if there are connections between the cells.
                    # Both Pegasus and Chimera have t vertical and t horizontal between cells:
                    for sub_i in range(sub_m):
                        for sub_j in range(sub_n):
                            if sub_m > 1 and sub_i < sub_m - 1:
                                match &= _between(
                                    _cell_qubits(s, i + sub_i, j + sub_j),
                                    _cell_qubits(s, i + sub_i + 1, j + sub_j)) == t
                            if sub_n > 1 and sub_j < sub_n - 1:
                                match &= _between(
                                    _cell_qubits(s, i + sub_i, j + sub_j),
                                    _cell_qubits(s, i + sub_i, j + sub_j + 1)) == t
                    
                    if match:
                        # Pull those cells out into an embedding.
                        embedding = {}
                        for sub_i in range(sub_m):
                            for sub_j in range(sub_n):
                                cells[s][i + sub_i][j + sub_j] = False  # Mark cell as matched
                                for u in range(2):
                                    for k in range(t):
                                        embedding[sub_c2i[sub_i, sub_j, u, k]] = {
                                            c2i[(s,i + sub_i, j + sub_j, u, k)]}

                        embeddings.append(embedding)

        if len(embeddings) == 0:
            raise ValueError("no tile embeddings found; "
                             "is the sampler Pegasus or Chimera structured?")
        
    @dimod.bqm_structured
    def sample(self, bqm, **kwargs):
        """Sample from the specified binary quadratic model.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            **kwargs:
                Optional keyword arguments for the sampling method, specified per solver.

        Returns:
            :class:`dimod.SampleSet`

        Examples:
            This example submits a simple Ising problem of just two variables on a
            D-Wave system.
            Because the problem fits in a single :term:`Chimera` unit cell, it is tiled
            across the solver's entire Chimera graph, resulting in multiple samples
            (the exact number depends on the working Chimera graph of the D-Wave system).

            >>> from dwave.system import DWaveSampler, EmbeddingComposite
            >>> from dwave.system import TilingComposite
            ...
            >>> qpu_2000q = DWaveSampler(solver={'topology__type': 'chimera'})
            >>> sampler = EmbeddingComposite(TilingComposite(qpu_2000q, 1, 1, 4))
            >>> response = sampler.sample_ising({},{('a', 'b'): 1})
            >>> len(response)    # doctest: +SKIP
            246

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/stable/concepts/index.html>`_
        for explanations of technical terms in descriptions of Ocean tools.

        """

        # apply the embeddings to the given problem to tile it across the child sampler
        embedded_bqm = dimod.BinaryQuadraticModel.empty(bqm.vartype)
        __, __, target_adjacency = self.child.structure
        for embedding in self.embeddings:
            embedded_bqm.update(dwave.embedding.embed_bqm(bqm, embedding, target_adjacency))

        # solve the problem on the child system
        tiled_response = self.child.sample(embedded_bqm, **kwargs)

        responses = []

        for embedding in self.embeddings:
            embedding = {v: chain for v, chain in embedding.items() if v in bqm.variables}

            responses.append(dwave.embedding.unembed_sampleset(tiled_response, embedding, bqm))

        answer = dimod.concatenate(responses)
        answer.info.update(tiled_response.info)

        return answer

    @property
    def num_tiles(self):
        return len(self.embeddings)
