from dwave_networkx.generators.pegasus import pegasus_graph, pegasus_coordinates
from dwave.embedding.polynomialembedder import processor


#TODO: ask Kelly if largest clique in pegasus == largest native clique in pegasus
def _pegasus_coord_to_chimera_fragments(q):
    u, w, k, z = q
    x0 = (z * 12 + __pegasus_offsets[u][k]) / 2
    y = (w * 12 + k) / 2
    ck = k % 2
    base = [0, 0, u, ck]
    for x in range(x0, x0 + 6):
        base[u] = x
        base[1 - u] = y
        yield tuple(base)


def _chimera_fragments_to_pegasus_coord(q):
    u = q[2]
    x = q[u]
    y = q[1 - u]
    w, k = divmod(2 * y + q[3], 12)
    offset = G.graph['horizontal_offsets'] if u else G.graph['vertical_offsets']
    x0 = x * 2 - offset[k]
    z = x0 / 12
    return u, w, k, z


def find_largest_clique(G):
    # Set up
    n_rows = G.graph['rows']    # Rows of unit cells
    n_cols = G.graph['cols']    # Columns of unit cells
    n_fragments = 6             # Number of fragments a qubit breaks into
    coord_converter = pegasus_coordinates(n_rows)   #TODO: double check n_cols is not needed

    # Break pegasus qubits into chimera fragments

    # Find clique embedding in terms of chimera fragments
    n_chimera_rows = n_rows * n_fragments
    n_chimera_cols = n_cols * n_fragments
    embedding_processor = processor(edges, M=n_chimera_rows, N=n_chimera_cols, L=2, linear=False)

    clique_embedding = embedding_processor.largestNativeClique()

    # Convert chimera fragment embedding in terms of Pegasus coordinates
    clique_embedding = {i: c for i, c in enumerate(DefragChimeraEmbedding(emb))}



def k_example():
    G = pegasus_graph(6)
    find_largest_clique(G)


k_example()
