"""
Tiles many smaller problems across a larger Chimera-structured sampler.
"""
from __future__ import division
from math import sqrt, ceil

import dimod
import dwave_networkx as dnx
import dwave_embedding_utilities as embutil

__all__ = ['TilingComposite']


class TilingComposite(dimod.Sampler, dimod.Composite, dimod.Structured):
    """ Composite to tile a small problem across a Chimera-structured sampler. A problem that can fit on a small Chimera
    graph can be replicated across a larger Chimera graph to get samples from multiple areas of the system in one call.
    For example, a 2x2 Chimera lattice could be tiled 64 times (8x8) on a fully-yielded D-Wave 2000Q system (16x16).

    Args:
        sampler (:class:`dimod.Sampler`): A structured dimod sampler to be wrapped.
        sub_m (int): The number of rows in the sub-Chimera lattice.
        sub_n (int): The number of columns in the sub-Chimera lattice.
        t (int): The size of the shore within each Chimera cell.

    """
    nodelist = None
    """list: The nodes available to the sampler."""

    edgelist = None
    """list: The edges available to the sampler."""

    parameters = None
    """dict[str, list]: The keys are the keyword parameters accepted by the child sampler."""

    properties = None
    """dict: Contains one key :code:`'child_properties'` which has a copy of the child sampler's properties."""

    children = None
    """list: Contains the single wrapped structured sampler."""

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

    def sample_ising(self, h, J, **kwargs):
        """Sample from the sub-Chimera lattice.

        Args:
            h (list/dict): Linear terms of the model.
            J (dict of (int, int):float): Quadratic terms of the model.
            **kwargs: Parameters for the sampling method, specified per solver.

        Returns:
            :class:`dimod.Response`

        """
        __, __, adjacency = self.structure
        if not all(v in adjacency for v in h):
            raise ValueError("nodes in linear bias do not map to the structure")
        if not all(u in adjacency[v] for u, v in J):
            raise ValueError("edges in quadratic bias do not map to the structure")

        # apply the embeddings to the given problem to tile it across the child sampler
        h_embs = {}
        J_embs = {}
        for embedding in self.embeddings:
            __, __, target_adjacency = self.child.structure
            h_emb, J_emb, J_chain = embutil.embed_ising(h, J, embedding, target_adjacency)
            assert(not J_chain)
            h_embs.update(h_emb)
            J_embs.update(J_emb)

        # solve the problem on the child system
        response = self.child.sample_ising(h_embs, J_embs, **kwargs)

        data_vectors = response.data_vectors.copy()

        source_response = None

        for embedding in self.embeddings:
            samples = embutil.unembed_samples(response, embedding,
                                              chain_break_method=embutil.minimize_energy,
                                              linear=h, quadratic=J)  # needed by minimize_energy

            # override the energy because it might have changed
            data_vectors['energy'] = [dimod.ising_energy(sample, h, J) for sample in samples]

            tile_response = dimod.Response.from_dicts(samples, data_vectors)

            if source_response is None:
                source_response = tile_response
                source_response.info.update(response.info)  # overwrite the info
            else:
                source_response.update(tile_response)

        return source_response

    @property
    def num_tiles(self):
        return len(self.embeddings)


def draw_tiling(sampler, t=4):
    """Draw Chimera graph of sampler with colored tiles.

    Args:
        sampler (:class:`dwave_micro_client_dimod.TilingComposite`): A tiled dimod sampler to be drawn.
        t (int): The size of the shore within each Chimera cell.

    Uses `dwave_networkx.draw_chimera` (see draw_chimera_).
    Linear biases are overloaded to color the graph according to which tile each Chimera cell belongs to.

    .. _draw_chimera: http://dwave-networkx.readthedocs.io/en/latest/reference/generated/dwave_networkx.drawing.chimera_layout.draw_chimera.html

    """

    child = sampler.child
    nodes_per_cell = t * 2
    m = n = int(ceil(sqrt(ceil(len(child.structure.nodelist) / nodes_per_cell))))  # assume square lattice shape
    system = dnx.chimera_graph(m, n, t, node_list=child.structure.nodelist, edge_list=child.structure.edgelist)

    labels = {node: -len(sampler.embeddings) for node in system.nodes}  # unused cells are blue
    labels.update({node: i for i, embedding in enumerate(sampler.embeddings) for s in embedding.values() for node in s})
    dnx.draw_chimera(system, linear_biases=labels)
