# Copyright 2020 D-Wave Systems Inc.
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
# =============================================================================
from networkx import grid_graph
from dwave_networkx import pegasus_graph, chimera_graph
from dwave.embedding.exceptions import *
from dwave.embedding.diagnostic import diagnose_embedding, verify_embedding
from dimod.generators import doped

def _pegasus_cubic(x, y, z):
    return [(z // 4, y, x, 0, z % 4), (z // 4, y, x, 1, z % 4)]


def _chimera_cubic_z12(x, y, z):
    return [(y * 3 + i, x * 3 + z // 4, 0, z % 4) for i in range(3)] + [(y * 3 + z // 4, x * 3 + i, 1, z % 4) for i in
                                                                        range(3)]


def _chimera_cubic_z8_v1(x, y, z):
    k, d = divmod(z, 2)
    return [(2 * y, 2 * x + d, 0, k), (2 * y + 1, 2 * x + d, 0, k), (2 * y + d, 2 * x, 1, k),
            (2 * y + d, 2 * x + 1, 1, k)]


def _chimera_cubic_z8_v2(x, y, z):
    d, k = divmod(z, 4)
    return [(2 * y, 2 * x + d, 0, k), (2 * y + 1, 2 * x + d, 0, k), (2 * y + d, 2 * x, 1, k),
            (2 * y + d, 2 * x + 1, 1, k)]


def _cubic_lattice_pegasus(size, grid=(12, 5, 5)):
    """ given the target size of a pegasus graph and the grid, will return the
        cubic lattice embedding of the given grid on a complete pegasus of size. """

    pegasus = pegasus_graph(size, nice_coordinates = True)
    g = grid_graph(list(grid))

    emb = {v: _pegasus_cubic(*v) for v in g}
    relabeled_emb = {k: [pegasus.nodes[vi]['linear_index'] for vi in v] for k, v in emb.items()}
    return relabeled_emb


def _cubic_lattice_chimera(size, version=1, grid=(8, 8, 8)):
    """ given the target size of a chimera graph, can only be 16, embedding version and the grid, will return the
        cubic lattice embedding of the given grid on a complete C16. """

    if size < 16:
        raise ValueError("cannot generate cubic lattice for chimera size < 16")
    chimera = chimera_graph(size, coordinates=True)

    g = grid_graph(list(grid))

    if grid == (8, 8, 8):
        if version == 1:
            emb = {v: _chimera_cubic_z8_v1(*v) for v in g}
        elif version == 2:
            emb = {v: _chimera_cubic_z8_v2(*v) for v in g}
    else:
        emb = {v: _chimera_cubic_z12(*v) for v in g}

    relabeled_emb = {k: [chimera.nodes[vi]['linear_index'] for vi in v] for k, v in emb.items()}
    return relabeled_emb


def _clean_embedding(embedding, source, target):
    """ given an embedding for a complete chimera/pegasus, and a target graph of a chimera/pegasus with missing
        nodes/edges the function will clean the source graph and the embedding """
    to_remove = set(())

    for x in diagnose_embedding(embedding, source, target):

        if issubclass(x[0], InvalidNodeError) or isinstance(x[0], DisconnectedChainError):
            to_remove.add(x[1])
        elif issubclass(x[0], MissingEdgeError):
            to_remove.add(x[1])
            to_remove.add(x[2])
        elif issubclass(x[0], DisconnectedChainError):
            to_remove.add(x[1])
        else:
            raise TypeError("Do not know how to handle {}".format(x[0]))

    source.remove_nodes_from(to_remove)
    [embedding.pop(key) for key in to_remove]

    if verify_embedding(embedding, source, target):
        return embedding, source
    else:
        raise EmbeddingError("Couldn't find a cubic lattice embedding")


def cubic_lattice_embedding(source, target, version=1):
    """

    Args:
        source (:obj:`~networkx.Graph`): A grid graph of dimensions (8,8,8) or (12,5,5). It will be modified in
            place based on the nodes of the target graph.
        target (:obj:`~networkx.Graph`): A pegasus or a chimera graph.
        version (int): only applicable for chimera (8,8,8) grid. can be 1 or 2. toggles between different modes
            of cubic lattice generation.

    Returns (dict): cubic lattice embedding

    """
    try:
        topology = target.graph['family']
        size = target.graph['rows']
        grid = tuple([x + 1 for x in list(source.nodes)[-1]])

    except:
        raise TypeError('the graph family should be either chimera or pegasus, received nothing')

    if topology == 'pegasus':
        embedding = _cubic_lattice_pegasus(size, grid=grid)

    elif topology == 'chimera':
        embedding = _cubic_lattice_chimera(size, version=version, grid=grid)

    else:
        raise TypeError("Don't know how to generate cubic lattice for topology {}".format(topology))

    embedding, source = _clean_embedding(embedding, source, target)
    return embedding, source


def cubic_lattice(target, grid, version=1, doping=0, seed=None):
    source = grid_graph(list(grid))
    embedding, source = cubic_lattice_embedding(source, target, version)
    bqm = doped(doping, source, seed=seed)

    return bqm, embedding


