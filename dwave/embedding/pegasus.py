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
from dwave_networkx.generators.pegasus import (get_tuple_defragmentation_fn, get_tuple_fragmentation_fn,
    pegasus_coordinates, pegasus_graph, get_nice_to_pegasus_fn, get_pegasus_to_nice_fn)
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
        fwd_converter = get_nice_to_pegasus_fn()
        back_converter = get_pegasus_to_nice_fn()
        pegasus_coords = [fwd_converter(*p) for p in target_graph.nodes]
        back_translate = lambda embedding: {key: [back_converter(*p) for p in chain]
                                      for key, chain in embedding.items()}
    elif target_graph.graph['labels'] == 'int':
        # Convert nodes in terms of Pegasus coordinates
        coord_converter = pegasus_coordinates(m)
        pegasus_coords = map(coord_converter.tuple, target_graph.nodes)

        # A function to convert our final coordinate embedding to an ints embedding
        back_translate = lambda embedding: {key: list(coord_converter.ints(chain))
                                      for key, chain in embedding.items()}
    else:
        pegasus_coords = target_graph.nodes
        back_translate = lambda embedding: embedding

    # Break each Pegasus qubits into six Chimera fragments
    # Note: By breaking the graph in this way, you end up with a K2,2 Chimera graph
    fragment_tuple = get_tuple_fragmentation_fn(target_graph)
    fragments = fragment_tuple(pegasus_coords)

    # Create a K2,2 Chimera graph
    # Note: 6 * m because Pegasus qubits split into six pieces, so the number of rows and columns
    #   get multiplied by six
    chim_m = 6 * m
    chim_graph = chimera_graph(chim_m, t=2, coordinates=True)

    # Determine valid fragment couplers in a K2,2 Chimera graph
    edges = chim_graph.subgraph(fragments).edges()

    # Find clique embedding in K2,2 Chimera graph
    embedding_processor = processor(edges, M=chim_m, N=chim_m, L=2, linear=False)
    chimera_clique_embedding = embedding_processor.tightestNativeClique(len(nodes))

    # Convert chimera fragment embedding in terms of Pegasus coordinates
    defragment_tuple = get_tuple_defragmentation_fn(target_graph)
    pegasus_clique_embedding = map(defragment_tuple, chimera_clique_embedding)
    pegasus_clique_embedding = dict(zip(nodes, pegasus_clique_embedding))
    pegasus_clique_embedding = back_translate(pegasus_clique_embedding)

    if len(pegasus_clique_embedding) != len(nodes):
        raise ValueError("No clique embedding found")

    return pegasus_clique_embedding
