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
    # Organize parameter values
    if target_graph is None:
        if m is None:
            raise TypeError("m and target_graph cannot both be None.")
        target_graph = pegasus_graph(m)

    m = target_graph.graph['rows']     # We only support square Pegasus graphs
    _, nodes = k

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
    chimera_clique_embedding = embedding_processor.tightestNativeClique(len(nodes))

    # Convert chimera fragment embedding in terms of Pegasus coordinates
    defragment_tuple = get_tuple_defragmentation_fn(target_graph)
    pegasus_clique_embedding = map(defragment_tuple, chimera_clique_embedding)
    pegasus_clique_embedding = dict(zip(nodes, pegasus_clique_embedding))
    pegasus_clique_embedding = back_translate(pegasus_clique_embedding)

    if len(pegasus_clique_embedding) != len(nodes):
        raise ValueError("No clique embedding found")

    return pegasus_clique_embedding
