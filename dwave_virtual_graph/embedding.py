import minorminer

from dwave_virtual_graph.exceptions import MissingEmbedding
# from dwave_virtual_graph.database_manager import select_embedding


def get_embedding(source_edges, target_edges):
    """Retrieve the an embedding mapping source to target.

    Args:
        source_edges (iterable[(hashable, hashable)]): The source graph
            as an iterable of edges as 2-tuples of nodes.
        target_edges (iterable[(hashable, hashable)]): The target graph
            as an iterable of edges as 2-tuples of nodes.

    Returns:
        dict[hashable, iterable]: An embedding that maps source to
        target.

    """

    embedding = None
    # embedding = select_embedding(source_edges, target_edges)

    if embedding is None:
        # for now let's just always generate on the fly
        embedding = minorminer.find_embedding(source_edges, target_edges)

        # this should change in later versions
        if isinstance(embedding, list):
            embedding = dict(enumerate(embedding))

    return embedding


def get_embedding_from_tag(embedding_tag):
    """Retrieve the embedding associated with the given tag.

    Args:
        embedding_tag (str): The tag associated with a previously
            cached embedding.

    Returns:
        dict[hashable, iterable]: An embedding.

    Raises:
        :exception:`.MissingEmbedding`: If there is no embedding in the
            cached with the given tag.


    """
    raise MissingEmbedding("retrieval from tag is not yet implemented")
