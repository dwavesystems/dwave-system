import minorminer

import dwave_embedding_utilities as embutil

from dwave.system.cache.database_manager \
    import select_embedding_from_source, select_embedding_from_tag, insert_embedding, cache_connect


def get_embedding(source_nodes, source_edges, target_nodelist, target_edgelist, presorted=False):
    embedding = select_embedding_from_source(cur, source_nodes, source_edges, target_nodelist, target_edgelist)
    return embedding


def get_embedding_from_tag(embedding_tag, target_nodelist, target_edgelist, presorted=False):
    with cache_connect() as cur:
        embedding = select_embedding_from_tag(cur, embedding_tag, target_nodelist, target_edgelist)
    return embedding


def load_embedding(target_nodelist, target_edgelist, embedding, embedding_tag):

    target_adjacency = {v: set() for v in target_nodelist}
    for u, v in target_edgelist:
        target_adjacency[u].add(v)
        target_adjacency[v].add(u)

    source_adjacency = embutil.target_to_source(target_adjacency, embedding)
    source_nodelist = sorted(source_adjacency)
    source_edgelist = sorted(sorted(edge) for edge in _adjacency_to_edges(source_adjacency))

    with cache_connect() as cur:
        insert_embedding(cur, source_nodelist, source_edgelist, target_nodelist, target_edgelist,
                         embedding, embedding_tag)


def _adjacency_to_edges(adjacency):
    """determine from an adjacency the list of edges
    if (u, v) in edges, then (v, u) should not be"""
    edges = set()
    for u in adjacency:
        for v in adjacency[u]:
            try:
                edge = (u, v) if u <= v else (v, u)
            except TypeError:
                # Py3 does not allow sorting of unlike types
                if (v, u) in edges:
                    continue
                edge = (u, v)

            edges.add(edge)
    return edges
