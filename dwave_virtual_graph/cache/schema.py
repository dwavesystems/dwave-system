"""The schema used by the sqlite database for storing the penalty models."""

schema = \
    """
    CREATE TABLE IF NOT EXISTS chain(
        nodes TEXT UNIQUE,  -- json list of nodes in the chain
        chain_length INTEGER NOT NULL,  -- number of nodes in the chain
        id INTEGER PRIMARY KEY);

    CREATE TABLE IF NOT EXISTS system(
        system_name TEXT UNIQUE,  -- the name of the system
        id INTEGER PRIMARY KEY);

    CREATE TABLE IF NOT EXISTS flux_bias(
        chain_id INTEGER NOT NULL,
        system_id INTEGER NOT NULL,
        insert_time DATE NOT NULL,  -- flux biases are only valid for relatively short time
        flux_bias BLOB NOT NULL,
        chain_strength BLOB NOT NULL,  -- magnitude of the chain strength
        FOREIGN KEY (chain_id) REFERENCES chain(id) ON DELETE CASCADE,
        FOREIGN KEY (system_id) REFERENCES system(id) ON DELETE CASCADE,
        PRIMARY KEY(chain_id, system_id, chain_strength));

    CREATE VIEW IF NOT EXISTS flux_bias_view AS
        SELECT
            chain.chain_length,
            chain.nodes,
            system.system_name,
            flux_bias.flux_bias,
            flux_bias.insert_time,
            flux_bias.chain_strength
        FROM flux_bias, chain, system WHERE
            flux_bias.system_id = system.id AND
            flux_bias.chain_id = chain.id;

    -- CREATE TABLE IF NOT EXISTS graph(
    --     num_nodes INTEGER NOT NULL,  -- for integer-labeled graphs, num_nodes encodes all of the nodes
    --     num_edges INTEGER NOT NULL,  -- redundant, allows for faster selects
    --     edges TEXT UNIQUE,  -- json list of lists, should be sorted (with each edge sorted)
    --     id INTEGER PRIMARY KEY);

    -- CREATE TABLE IF NOT EXISTS embedding(
    --     source_id INTEGER REFERENCES graph(id),
    --     target_id INTEGER REFERENCES graph(id),
    --     tag TEXT,  -- user specified
    --     id INTEGER PRIMARY KEY,
    --     CONSTRAINT embedding_constraint UNIQUE (source_id, target_id, tag));

    -- CREATE TABLE IF NOT EXISTS embedding_chains(
    --     source_node INTEGER NOT NULL,
    --     chain_id INTEGER NOT NULL,
    --     embedding_id INTEGER NOT NULL,
    --     FOREIGN KEY (chain_id) REFERENCES chain(id),
    --     FOREIGN KEY (embedding_id) REFERENCES embedding(id),
    --     PRIMARY KEY(embedding_id, source_node, chain_id));

    -- CREATE VIEW IF NOT EXISTS embedding_view AS
    --     SELECT
    --         source_graph.id 'source_id',
    --         source_graph.num_nodes 'source_num_nodes',
    --         source_graph.num_edges 'source_num_edges',
    --         source_graph.edges 'source_edges',
    --         target_graph.id 'target_id',
    --         target_graph.num_nodes 'target_num_nodes',
    --         target_graph.num_edges 'target_num_edges',
    --         target_graph.edges 'target_edges',
    --         source_node,
    --         chain_length,
    --         chain.nodes 'chain',
    --         embedding.id 'embedding_id',
    --         embedding.tag 'embedding_tag'
    --     FROM embedding_chains, graph 'source_graph', graph 'target_graph', chain, embedding
    --     WHERE
    --         embedding.source_id = source_graph.id AND
    --         embedding.target_id = target_graph.id AND
    --         embedding_chains.chain_id = chain.id AND
    --         embedding.id = embedding_chains.embedding_id;
    """
