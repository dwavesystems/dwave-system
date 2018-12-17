from dwave_networkx.generators.chimera import chimera_graph
from dwave_networkx.generators.pegasus import (get_tuple_defragmentation_fn, get_tuple_fragmentation_fn,
    pegasus_coordinates)
from dwave.embedding.polynomialembedder import processor
import networkx as nx


#TODO: should I be catching the case when user does not provide sufficient offsets?
#TODO: perhaps just note that if the offset isn't needed, put in None



#TODO: change function interface to more closely resemble chimera
@nx.utils.decorators.nodes_or_number(0)
def find_clique_embedding(k, G):
    """Find an embedding of a k-sized clique on a Pegasus graph.

    This clique is found by transforming the Pegasus graph into a K2,2 Chimera graph and then
    applying a Chimera clique finding algorithm. The results are then converted back in terms of
    Pegasus coordinates.

    Args:
         G: a Pegasus graph

    Returns:
        A dictionary representing G's clique embedding. Each dictionary key represents a node
        in said clique. Each corresponding dictionary value is a list of pegasus coordinates
        that should be chained together to represent said node.
    """
    n_nodes, nodes = k
    m = G.graph['rows']     # We only support square Pegasus graphs
    v_offsets = G.graph['vertical_offsets']
    h_offsets = G.graph['horizontal_offsets']

    # Break each Pegasus qubits into six Chimera fragments
    # Note: By breaking the graph in this way, you end up with a K2,2 Chimera graph
    coord_converter = pegasus_coordinates(m)
    pegasus_coords = map(coord_converter.tuple, G.nodes)
    fragment_tuple = get_tuple_fragmentation_fn(G)
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
    chimera_clique_embedding = embedding_processor.tightestNativeClique(n_nodes)

    # Convert chimera fragment embedding in terms of Pegasus coordinates
    defragment_tuple = get_tuple_defragmentation_fn(G)
    pegasus_clique_embedding = map(defragment_tuple, chimera_clique_embedding)

    #TODO: raise error for no embedding
    return dict(zip(nodes, pegasus_clique_embedding))
