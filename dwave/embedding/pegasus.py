from dwave_networkx.generators.chimera import chimera_graph
from dwave_networkx.generators.pegasus import pegasus_graph, pegasus_coordinates
from dwave.embedding.polynomialembedder import processor


#TODO: ask Kelly if largest clique in pegasus == largest native clique in pegasus
#TODO: double check that Pegasus generator topology is the one we want
#TODO: check that // is okay

#TODO: should I be catching the case when user does not provide sufficient offsets?
#TODO: perhaps just note that if the offset isn't needed, put in None
def get_chimera_fragments(pegasus_coords, vertical_offsets, horizontal_offsets):
    """Takes the Pegasus qubits from G and returns their corresponding Chimera fragments"""
    fragments = []

    for u, w, k, z in pegasus_coords:
        # Determine offset
        offset = vertical_offsets if u else horizontal_offsets
        offset = offset[k]

        # Find base Chimera fragment
        x0 = (z * 12 + offset) // 2
        y = (w * 12 + k) // 2
        ck = k % 2
        base = [0, 0, u, ck]

        # Generate the 6 fragments associated with node
        for x in range(x0, x0 + 6):
            base[u] = x
            base[1 - u] = y
            fragments.append(tuple(base))

    return set(fragments)


def get_pegasus_coordinates(chimera_coords, vertical_offsets, horizontal_offsets):
    pegasus_coords = []
    for q in chimera_coords:
        u = q[2]
        x = q[u]
        y = q[1 - u]
        w, k = divmod(2 * y + q[3], 12)
        offset = horizontal_offsets if u else vertical_offsets
        x0 = x * 2 - offset[k]
        z = x0 // 12
        pegasus_coords.append((u, w, k, z))

    return set(pegasus_coords)


def find_largest_clique(G):
    # Break Pegasus qubits into chimera fragments
    # Note: When you break Pegasus qubits into 6 pieces, you end up with a K2,2 Chimera graph
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
    print(find_largest_clique(G))


k_example()
