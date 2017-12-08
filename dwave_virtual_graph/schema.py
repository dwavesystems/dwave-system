__all__ = ['schema']

graph = \
    """
    CREATE TABLE IF NOT EXISTS graph
        (
            num_nodes INTEGER NOT NULL,
            num_edges INTEGER NOT NULL,
            edges TEXT NOT NULL,
            id INTEGER PRIMARY KEY
        );
    """

# graph_ix_num_nodes_num_edges_edges_id = """CREATE INDEX IF NOT EXISTS idx_graph ON graphs(
#                     num_nodes,
#                     num_edges,
#                     edges,
#                     id);
#               """

embedding = \
    """
    CREATE TABLE IF NOT EXISTS embedding
        (
            embedding TEXT NOT NULL,
            id INTEGER PRIMARY KEY
        );
    """

mapping = \
    """
    CREATE TABLE IF NOT EXISTS mapping
        (
            source_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            embedding_id, INTEGER NOT NULL,
            tag TEXT,
            id INTEGER PRIMARY KEY
        );
    """

embedding_view = \
    """
    CREATE VIEW IF NOT EXISTS embedding_view AS
    SELECT
        embedding
    FROM
        embedding, mapping, graph
    WHERE
        embedding.id = mapping.embedding_id AND
        source
    """

schema = [graph, embedding, mapping]
