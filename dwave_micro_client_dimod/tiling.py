"""
TilingComposite
==================
"""
from __future__ import division
#import itertools
from math import sqrt, ceil

import dimod
import dwave_networkx as dnx
import dwave_embedding_utilities as embutil


class TilingComposite(dimod.TemplateComposite):
    """Composite to tile a small problem across a structured sampler.

    Args:
        sampler (:class:`dimod.TemplateSampler`): A structured dimod sampler to be wrapped.
        sub_m (int): The number of rows in the sub Chimera lattice.
        sub_n (int): The number of columns in the sub Chimera lattice.
        t (int): The size of the shore within each Chimera cell.

    """

    def __init__(self, sampler, sub_m, sub_n, t):
        # The composite __init__ adds the sampler into self.children
        dimod.TemplateComposite.__init__(self, sampler)
        self._child = sampler  # faster access than self.children[0]
        tile = dnx.chimera_graph(sub_m, sub_n, t)
        self.structure = (sorted(tile.nodes), sorted(tile.edges), sorted(tile.adjacency()))

        nodes_per_cell = t * 2
        edges_per_cell = t * t
        # CAN SQUARE SHAPE BE ASSUMED?
        size_m = size_n = int(ceil(sqrt(ceil(len(sampler.structure[0]) / nodes_per_cell))))
        system = dnx.chimera_graph(size_m, size_n, t, sampler.structure[0], sampler.structure[1])
        c2i = {chimera_index: linear_index for (linear_index, chimera_index) in system.nodes(data='chimera_index')}
        sub_c2i = {chimera_index: linear_index for (linear_index, chimera_index) in tile.nodes(data='chimera_index')}

        # Count the connections between these qubits
        def _between(qubits1, qubits2):
            edges = [edge for edge in system.edges if edge[0] in qubits1 and edge[1] in qubits2]
            return len(edges)

        # Get the list of qubits in a cell
        def _cell_qubits(i, j):
            return [c2i[(i, j, u, k)] for u in range(2) for k in range(t)]

        # get a mask of complete cells
        cells = [[False for _ in range(size_n)] for _ in range(size_m)]
        for i in range(size_m):
            for j in range(size_n):
                qubits = _cell_qubits(i, j)
                active_in_cell = sum(q in system.nodes for q in qubits)
                cells[i][j] = active_in_cell == nodes_per_cell and _between(qubits, qubits) == edges_per_cell

        # List of 'embeddings'
        self.embeddings = []

        # For each possible chimera cell check if the next few cells are complete
        for i in range(size_m + 1 - sub_m):
            for j in range(size_n + 1 - sub_n):

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

                    self.embeddings.append(embedding)

    @dimod.decorators.ising(1, 2)
    def sample_ising(self, h, J, **kwargs):
        __, __, adjacency = self.structure
        if not all(v in adjacency for v in h):
            raise ValueError("nodes in linear bias do not map to the structure")
        if not all(u in adjacency[v] for u, v in J):
            raise ValueError("edges in quadratic bias do not map to the structure")

        # apply the embeddings to the given problem to tile it across the child sampler
        h_embs = {}
        J_embs = {}
        for embedding in self.embeddings:
            __, __, target_adjacency = self._child.structure
            h_emb, J_emb, J_chain = embutil.embed_ising(h, J, embedding, target_adjacency)
            assert(not J_chain)
            h_embs.update(h_emb)
            J_embs.update(J_emb)

        # solve the problem on the child system
        response = self._child.sample_ising(h_embs, J_embs, **kwargs)

        # unembed the tiled problem and combine results into one response object
        source_response = dimod.SpinResponse()
        for embedding in self.embeddings:
            samples = embutil.unembed_samples(response, embedding,
                                              chain_break_method=embutil.minimize_energy,
                                              linear=h, quadratic=J)  # needed by minimize_energy
            source_response.add_samples_from(samples,
                                             sample_data=(data for __, data in response.samples(data=True)),
                                             h=h, J=J)

        return source_response
