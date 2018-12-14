from dwave_networkx.generators.chimera import chimera_graph
from dwave_networkx.generators.pegasus import pegasus_graph, pegasus_coordinates
from dwave.embedding.polynomialembedder import processor
import networkx as nx


#TODO: should I be catching the case when user does not provide sufficient offsets?
#TODO: perhaps just note that if the offset isn't needed, put in None
def get_chimera_fragments(pegasus_coords, vertical_offsets, horizontal_offsets):
    """Takes the Pegasus qubit coordinates and returns their corresponding K2,2 Chimera fragment
    coordinates.

    Specifically, each Pegasus qubit is split into six fragments. If edges are drawn between
    adjacent fragments and drawn between fragments that are connected by an existing Pegasus
    coupler, we can see that a K2,2 Chimera graph is formed.

    The K2,2 Chimera graph uses a coordinate system with an origin at the upper left corner of the
    graph.
        y: number of vertical fragments from the top-most row
        x: number of horizontal fragments from the left-most column
        u: 1 if it belongs to a horizontal qubit, 0 otherwise
        r: fragment index on the K2,2 shore

    Args:
        pegasus_coords: List of 4-tuple ints
        vertical_offsets: List of ints. List of offsets for vertical pegasus qubits
        horizontal_offsets: List of ints. List of offsets for horizontal Pegasus qubits

    """
    fragments = []
    for u, w, k, z in pegasus_coords:
        # Determine offset
        offset = horizontal_offsets if u else vertical_offsets
        offset = offset[k]

        # Find the base (i.e. zeroth) Chimera fragment of this pegasus coordinate
        x0 = (z * 12 + offset) // 2
        y = (w * 12 + k) // 2
        r = k % 2
        base = [0, 0, u, r]

        # Generate the six fragments associated with this pegasus coordinate
        for x in range(x0, x0 + 6):
            base[u] = x
            base[1 - u] = y
            fragments.append(tuple(base))

    return fragments


def get_pegasus_coordinates(chimera_coords, pegasus_vertical_offsets, pegasus_horizontal_offsets):
    """Given a list of K2,2 Chimera coordinates, return the corresponding set of pegasus
    coordinates.

    Args:
        chimera_coords: List of 4-tuple ints
        pegasus_vertical_offsets: List of ints
        pegasus_horizontal_offsets: List of ints

    Return:
        A set of pegasus coordinates
    """
    pegasus_coords = []
    for y, x, u, r in chimera_coords:
        # Set up shifts and offsets
        shifts = [x, y]
        offsets = pegasus_horizontal_offsets if u else pegasus_vertical_offsets

        # Determine number of tiles and track number
        w, k = divmod(2 * shifts[u] + r, 12)

        # Determine qubit index on track
        x0 = shifts[1-u] * 2 - offsets[k]
        z = x0 // 12

        pegasus_coords.append((u, w, k, z))

    # Several chimera coordinates may map to the same pegasus coordinate, hence, apply set(..)
    return set(pegasus_coords)


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
    fragments = get_chimera_fragments(pegasus_coords, v_offsets, h_offsets)

    # Create a K2,2 Chimera graph
    # Note: 6 * m because Pegasus qubits split into six pieces, so the number of rows and columns
    #   get multiplied by six
    chim_m = 6 * m
    chim_graph = chimera_graph(6*m, t=2, coordinates=True)

    # Determine valid fragment couplers in a K2,2 Chimera graph
    edges = chim_graph.subgraph(fragments).edges()

    # Find clique embedding in K2,2 Chimera graph
    embedding_processor = processor(edges, M=chim_m, N=chim_m, L=2, linear=False)
    chimera_clique_embedding = embedding_processor.tightestNativeClique(n_nodes)

    # Convert chimera fragment embedding in terms of Pegasus coordinates
    pegasus_clique_embedding = map(lambda x: get_pegasus_coordinates(x, v_offsets, h_offsets),
                                   chimera_clique_embedding)

    #TODO: raise error for no embedding
    return dict(zip(nodes, pegasus_clique_embedding))
