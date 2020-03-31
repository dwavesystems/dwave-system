# Copyright 2019 D-Wave Systems Inc.
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

from dwave_networkx.generators.chimera import chimera_graph
from dwave_networkx.generators.pegasus import (get_tuple_defragmentation_fn, fragmented_edges,
    pegasus_coordinates, pegasus_graph)
from dwave.embedding.polynomialembedder import processor
import networkx as nx


def _pegasus_fragment_helper(m=None, target_graph=None):
    # This is a function that takes m or a target_graph and produces a
    # `processor` object for the corresponding Pegasus graph, and a function
    # that translates embeddings produced by that object back to the original
    # pegasus graph.  Consumed by `find_clique_embedding` and
    # `find_biclique_embedding`.

    # Organize parameter values
    if target_graph is None:
        if m is None:
            raise TypeError("m and target_graph cannot both be None.")
        target_graph = pegasus_graph(m)

    m = target_graph.graph['rows']     # We only support square Pegasus graphs

    # Deal with differences in ints vs coordinate target_graphs
    if target_graph.graph['labels'] == 'nice':
        back_converter = pegasus_coordinates.pegasus_to_nice
        back_translate = lambda embedding: {key: [back_converter(p) for p in chain]
                                      for key, chain in embedding.items()}
    elif target_graph.graph['labels'] == 'int':
        # Convert nodes in terms of Pegasus coordinates
        coord_converter = pegasus_coordinates(m)

        # A function to convert our final coordinate embedding to an ints embedding
        back_translate = lambda embedding: {key: list(coord_converter.iter_pegasus_to_linear(chain))
                                      for key, chain in embedding.items()}
    else:
        back_translate = lambda embedding: embedding

    # collect edges of the graph produced by splitting each Pegasus qubit into six pieces
    fragment_edges = list(fragmented_edges(target_graph))

    # Find clique embedding in K2,2 Chimera graph
    embedding_processor = processor(fragment_edges, M=m*6, N=m*6, L=2, linear=False)

    # Convert chimera fragment embedding in terms of Pegasus coordinates
    defragment_tuple = get_tuple_defragmentation_fn(target_graph)
    def embedding_to_pegasus(nodes, emb):
        emb = map(defragment_tuple, emb)
        emb = dict(zip(nodes, emb))
        emb = back_translate(emb)
        return emb

    return embedding_processor, embedding_to_pegasus

@nx.utils.decorators.nodes_or_number(0)
def find_clique_embedding(k, m=None, target_graph=None):
    """Find an embedding for a clique in a Pegasus graph.

    Given a clique (fully connected graph) and target Pegasus graph, attempts
    to find an embedding by transforming the Pegasus graph into a :math:`K_{2,2}`
    Chimera graph and then applying a Chimera clique-finding algorithm. Results
    are converted back to Pegasus coordinates.

    Args:
        k (int/iterable/:obj:`networkx.Graph`): A complete graph to embed,
            formatted as a number of nodes, node labels, or a NetworkX graph.
        m (int): Number of tiles in a row of a square Pegasus graph. Required to
            generate an m-by-m Pegasus graph when `target_graph` is None.
        target_graph (:obj:`networkx.Graph`): A Pegasus graph. Required when `m`
            is None.

    Returns:
        dict: An embedding as a dict, where keys represent the clique's nodes and
        values, formatted as lists, represent chains of pegasus coordinates.

    Examples:
        This example finds an embedding for a :math:`K_3` complete graph in a
        2-by-2 Pegaus graph.

        >>> from dwave.embedding.pegasus import find_clique_embedding
        ...
        >>> print(find_clique_embedding(3, 2))    # doctest: +SKIP
        {0: [10, 34], 1: [35, 11], 2: [32, 12]}

    """
    _, nodes = k

    embedding_processor, embedding_to_pegasus = _pegasus_fragment_helper(m, target_graph)
    chimera_clique_embedding = embedding_processor.tightestNativeClique(len(nodes))
    pegasus_clique_embedding = embedding_to_pegasus(nodes, chimera_clique_embedding)

    if len(pegasus_clique_embedding) != len(nodes):
        raise ValueError("No clique embedding found")

    return pegasus_clique_embedding


@nx.utils.decorators.nodes_or_number(0)
@nx.utils.decorators.nodes_or_number(1)
def find_biclique_embedding(a, b, m=None, target_graph=None):
    """Find an embedding for a biclique in a Pegasus graph.

    Given a biclique (a bipartite graph where every vertex in a set in connected
    to all vertices in the other set) and a target :term:`Pegasus` graph size or
    edges, attempts to find an embedding.

    Args:
        a (int/iterable):
            Left shore of the biclique to embed. If a is an integer, generates
            an embedding for a biclique with the left shore of size a labelled
            [0,a-1]. If a is an iterable of nodes, generates an embedding for a
            biclique with the left shore of size len(a) labelled for the given
            nodes.

        b (int/iterable):
            Right shore of the biclique to embed.If b is an integer, generates
            an embedding for a biclique with the right shore of size b labelled
            [0,b-1]. If b is an iterable of nodes, generates an embedding for a
            biclique with the right shore of size len(b) labelled for the given
            nodes.

        m (int): Number of tiles in a row of a square Pegasus graph. Required to
            generate an m-by-m Pegasus graph when `target_graph` is None.

        target_graph (:obj:`networkx.Graph`): A Pegasus graph. Required when `m`
            is None.

    Returns:
        tuple: A 2-tuple containing:

            dict: An embedding mapping the left shore of the biclique to
            the Pegasus lattice.

            dict: An embedding mapping the right shore of the biclique to
            the Pegasus lattice.

    Examples:
        This example finds an embedding for an alphanumerically labeled biclique in a small
        Pegasus graph

        >>> from dwave.embedding.pegasus import find_biclique_embedding
        ...
        >>> left, right = find_biclique_embedding(['a', 'b', 'c'], ['d', 'e'], 2)
        >>> print(left, right)  # doctest: +SKIP
        {'a': [40], 'b': [41], 'c': [42]} {'d': [4], 'e': [5]}

    """
    _, anodes = a
    _, bnodes = b

    embedding_processor, embedding_to_pegasus = _pegasus_fragment_helper(m, target_graph)
    embedding = embedding_processor.tightestNativeBiClique(len(anodes), len(bnodes))

    if not embedding:
        raise ValueError("cannot find a K{},{} embedding for given Pegasus lattice".format(a, b))

    left = embedding_to_pegasus(anodes, embedding[0])
    right = embedding_to_pegasus(bnodes, embedding[1])
    return left, right
