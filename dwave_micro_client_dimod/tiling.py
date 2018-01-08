"""
TilingComposite
==================
"""
#import itertools

import dimod
import dwave_networkx as dnx
import dwave_embedding_utilities as embutil


class TilingComposite(dimod.TemplateComposite):
    """Composite to tile a small problem across a structured sampler.

    Args:
        sampler (:class:`dimod.TemplateSampler`):
            A structured dimod sampler to be wrapped.

    """

    def __init__(self, sampler):
        # The composite __init__ adds the sampler into self.children
        dimod.TemplateComposite.__init__(self, sampler)
        self._child = sampler  # faster access than self.children[0]
        self.structure = (sorted(tile.nodes), sorted(tile.edges), sorted(tile.adjacency()))

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









##
#THIS SHOULD ALL HAPPEN IN __init__
##
import math


def simple_find_subchimera(config, sub_size_m, sub_size_n):
    """
    Find a bunch of complete sub chimera graphs on the target solver.

    :param config: Configuration file data.
    :param sub_size_m: How big of a complete chimera graph do we need.
    :param sub_size_n: How big of a complete chimera graph do we need.
    """

    # Load the adjacency/dwave_sapi components for each version
    import dwave_sapi2.remote
    import dwave_sapi2.util
    connection = dwave_sapi2.remote.RemoteConnection(config['url'], config['token'])
    solver = connection.get_solver(config['solver'])
    adjacency = dwave_sapi2.util.get_hardware_adjacency(solver) # BRAD FIND IN _child.structure
    c2i = dwave_sapi2.util.chimera_to_linear_index # BRAD USE DNX (NEEDS TO BE ADDED)
    size = int(math.sqrt(solver.properties['num_qubits'] / 8)) # BRAD NO HARDCODED 8, CAN SQUARE SHAPE BE ASSUMED?

    # Get active qubits
    active_qubits = set() # BRAD FIND IN _child.structure
    for qubit1, qubit2 in adjacency:
        active_qubits.add(qubit1)
        active_qubits.add(qubit2)

    # Count the connections between these qubits
    def _between(qubits1, qubits2):
        edges = [edge for edge in adjacency if edge[0] in qubits1 and edge[1] in qubits2]
        return len(edges)

    # Get the list of qubits in a cell
    def _cell_qubits(ii, jj):
        return [c2i([[ii, jj, uu, kk]], size, size, 4)[0] for uu in range(2) for kk in range(4)] # BRAD NO HARDCODED 4

    # get a mask of complete cells
    cells = [[False for _ in range(size)] for _ in range(size)]
    for ii in range(size):
        for jj in range(size):
            qubits = _cell_qubits(ii, jj)
            active_in_cell = sum(q in active_qubits for q in qubits)
            cells[ii][jj] = active_in_cell == 8 and _between(qubits, qubits) == 32 # BRAD NO HARDCODED 8

    # List of 'embeddings'
    embeddings = []

    # For each possible chimera cell check if the next few cells are complete
    for ii in range(size + 1 - sub_size_m):
        for jj in range(size + 1 - sub_size_n):

            # Check if the sub cells are matched
            match = all(cells[ii + _i][jj + _j] for _i in range(sub_size_m) for _j in range(sub_size_n))

            # Check if there are connections between the cells.
            for _i in range(sub_size_m):
                for _j in range(sub_size_n):
                    if sub_size_m > 1 and _i < sub_size_m - 1:
                         # BRAD NO HARDCODED 4
                        match &= 4 == _between(_cell_qubits(ii + _i, jj + _j), _cell_qubits(ii + _i + 1, jj + _j))
                    if sub_size_n > 1 and _j < sub_size_n - 1:
                         # BRAD NO HARDCODED 4
                        match &= 4 == _between(_cell_qubits(ii + _i, jj + _j), _cell_qubits(ii + _i, jj + _j + 1))

            if match:
                # Pull those cells out into an embedding.
                embedding = [] # BRAD REFORMAT AS PER ADTT CONVENTION
                for _i in range(sub_size_m):
                    for _j in range(sub_size_n):
                        cells[ii + _i][jj + _j] = False
                        for uu in range(2):
                            for kk in range(4):
                                 # BRAD FLATTEN THIS OUT, NO HARDCODED 4
                                embedding.append(c2i([[ii + _i, jj + _j, uu, kk]], size, size, 4))

                embeddings.append(embedding)

    return embeddings
