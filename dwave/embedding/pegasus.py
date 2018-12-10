from dwave_networkx.generators.chimera import chimera_graph
from dwave_networkx.generators.pegasus import pegasus_graph, pegasus_coordinates
from dwave.embedding.polynomialembedder import processor


#TODO: ask Kelly if largest clique in pegasus == largest native clique in pegasus
#TODO: ask Kelly if this is the only code that needs fragment to coord conversion.
#TODO: remove hardcoded 12 and 2

def _fragmentize(G):
    """Takes the Pegasus qubits from G and returns their corresponding Chimera fragments"""
    fragments = []
    coord_converter = pegasus_coordinates(G.graph['rows'])   #TODO: double check n_cols is not needed

    for node in G.nodes:
        u, w, k, z = coord_converter.tuple(node)

        # Find base Chimera fragmenta
        #TODO: check that things are ints
        offset = G.graph['horizontal_offsets'] if u else G.graph['vertical_offsets']
        offset = offset[k]
        x0 = int((z * 12 + offset) / 2)
        y = int((w * 12 + k) / 2)
        ck = int(k % 2)
        base = [0, 0, u, ck]

        # Generate the 6 fragments associated with node
        for x in range(x0, x0 + 6):
            base[u] = x
            base[1 - u] = y
            fragments.append(tuple(base))

    return set(fragments)


def _defragmentize(embedding, G):
    pegasus_embedding = []
    for chain_of_fragments in embedding:

        pegasus_chain = []
        for q in chain_of_fragments:
            u = q[2]
            x = q[u]
            y = q[1 - u]
            w, k = divmod(2 * y + q[3], 12)
            offset = G.graph['horizontal_offsets'] if u else G.graph['vertical_offsets']
            x0 = x * 2 - offset[k]
            z = x0 / 12
            pegasus_chain.append((u, w, k, z))

        pegasus_embedding.append(set(pegasus_chain))

    return pegasus_embedding

def find_largest_clique(G):
    # Set up
    n_rows = G.graph['rows']    # Rows of unit cells
    n_cols = G.graph['columns']    # Columns of unit cells
    n_fragments = 6             # Number of fragments a qubit breaks into

    # Break pegasus qubits into chimera fragments
    fragments = _fragmentize(G)

    # Find clique embedding in terms of chimera fragments
    n_chim_rows = n_rows * n_fragments
    n_chim_cols = n_cols * n_fragments
    chim_graph = chimera_graph(n_chim_rows, n=n_chim_cols, t=2, coordinates=True)
    chim_edges = chim_graph.subgraph(fragments).edges()
    embedding_processor = processor(chim_edges, M=n_chim_rows, N=n_chim_cols, L=2, linear=False)

    chim_clique_embedding = embedding_processor.largestNativeClique()

    # Convert chimera fragment embedding in terms of Pegasus coordinates
    #TODO: differentiate between list and dict clique embedding
    #TODO: need to convert calculations in terms of int
    clique_embedding = {i: x for i, x in enumerate(_defragmentize(chim_clique_embedding, G))}
    return clique_embedding

def k_example():
    G = pegasus_graph(6)
    print(find_largest_clique(G))


k_example()