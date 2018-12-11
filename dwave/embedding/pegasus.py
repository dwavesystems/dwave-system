from dwave_networkx.generators.chimera import chimera_graph
from dwave_networkx.generators.pegasus import pegasus_graph, pegasus_coordinates
from dwave.embedding.polynomialembedder import processor


#TODO: ask Kelly if largest clique in pegasus == largest native clique in pegasus
#TODO: double check that Pegasus generator topology is the one we want
#TODO: check that // is okay

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
        ck: fragment index on the K2,2 shore
    """
    fragments = []
    for u, w, k, z in pegasus_coords:
        # Determine offset
        offset = vertical_offsets if u else horizontal_offsets
        offset = offset[k]

        # Find the base (i.e. zeroth) Chimera fragment of this pegasus coordinate
        x0 = (z * 12 + offset) // 2
        y = (w * 12 + k) // 2
        ck = k % 2
        base = [0, 0, u, ck]

        # Generate the 6 fragments associated with this pegasus coordinate
        for x in range(x0, x0 + 6):
            base[u] = x
            base[1 - u] = y
            fragments.append(tuple(base))

    return fragments


def get_pegasus_coordinates(chimera_coords, pegasus_vertical_offsets, pegasus_horizontal_offsets):
    """Given a list of K2,2 Chimera coordinates, return the corresponding set of pegasus
    coordinates.

    Args:
        chimera_coords: List of K2,2 Chimera coordinates
        pegasus_vertical_offsets: List of vertical offsets of the Pegasus graph
        pegasus_horizontal_offsets: List of horizontal offsets of the Pegasus graph

    Return:
        A set of pegasus coordinates
    """
    pegasus_coords = []
    for q in chimera_coords:
        u = q[2]
        x = q[u]
        y = q[1 - u]
        w, k = divmod(2 * y + q[3], 12)
        offset = pegasus_horizontal_offsets if u else pegasus_vertical_offsets
        x0 = x * 2 - offset[k]
        z = x0 // 12
        pegasus_coords.append((u, w, k, z))

    return set(pegasus_coords)


def find_largest_native_clique(G):
    """Find the largest native clique in a Pegasus graph.

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
    # Break each Pegasus qubits into six Chimera fragments
    # Note: By breaking the graph in this way, you end up with a K2,2 Chimera graph
    v_offsets = G.graph['vertical_offsets']
    h_offsets = G.graph['horizontal_offsets']
    coord_converter = pegasus_coordinates(G.graph['rows'])   #TODO: double check n_cols is not needed
    pegasus_coords = map(coord_converter.tuple, G.nodes)
    fragments = get_chimera_fragments(pegasus_coords, v_offsets, h_offsets)

    # Create a K2,2 Chimera graph
    n_fragments = 6
    n_rows = G.graph['rows'] * n_fragments
    n_cols = G.graph['columns'] * n_fragments
    chim_graph = chimera_graph(n_rows, n=n_cols, t=2, coordinates=True)

    # Determine valid fragment couplers in a K2,2 Chimera graph
    edges = chim_graph.subgraph(fragments).edges()

    # Find clique embedding in K2,2 Chimera graph
    embedding_processor = processor(edges, M=n_rows, N=n_cols, L=2, linear=False)
    chimera_clique_embedding = embedding_processor.largestNativeClique()

    # Convert chimera fragment embedding in terms of Pegasus coordinates
    pegasus_clique_embedding = map(lambda x: get_pegasus_coordinates(x, v_offsets, h_offsets),
                                   chimera_clique_embedding)
    pegasus_clique_embedding = {i: x for i, x in enumerate(pegasus_clique_embedding)}
    return pegasus_clique_embedding

def k_example():
    G = pegasus_graph(6)
    print(find_largest_native_clique(G))


k_example()
