# Copyright 2018 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# ================================================================================================
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

    CREATE TABLE IF NOT EXISTS graph(
        num_nodes INTEGER NOT NULL,  -- for integer-labeled graphs, num_nodes encodes all of the nodes
        num_edges INTEGER NOT NULL,  -- redundant, allows for faster selects
        edges TEXT NOT NULL,  -- json list of lists, should be sorted (with each edge sorted)
        id INTEGER PRIMARY KEY,
        CONSTRAINT graph UNIQUE (
            num_nodes,
            edges));

    CREATE TABLE IF NOT EXISTS embedding(
        source_id INTEGER NOT NULL,
        target_id INTEGER NOT NULL,
        tag TEXT,  -- user specified
        id INTEGER PRIMARY KEY,
        protected INTEGER DEFAULT 0,
        CONSTRAINT embedding_constraint UNIQUE (target_id, tag),
        CONSTRAINT embedding_constraint UNIQUE (target_id, source_id),
        FOREIGN KEY (source_id) REFERENCES graph(id) ON DELETE CASCADE,
        FOREIGN KEY (target_id) REFERENCES graph(id) ON DELETE CASCADE);

    CREATE TABLE IF NOT EXISTS embedding_component(
        source_node INTEGER NOT NULL,
        chain_id INTEGER NOT NULL,
        embedding_id INTEGER NOT NULL,
        FOREIGN KEY (chain_id) REFERENCES chain(id) ON DELETE CASCADE,
        FOREIGN KEY (embedding_id) REFERENCES embedding(id) ON DELETE CASCADE,
        PRIMARY KEY(embedding_id, source_node));

    CREATE VIEW IF NOT EXISTS embedding_component_view AS
        SELECT
            source_graph.num_nodes 'source_num_nodes',
            source_graph.num_edges 'source_num_edges',
            source_graph.edges 'source_edges',

            target_graph.num_nodes 'target_num_nodes',
            target_graph.num_edges 'target_num_edges',
            target_graph.edges 'target_edges',

            embedding_component.source_node,

            chain.chain_length,
            chain.nodes 'chain',

            embedding.tag 'embedding_tag'
        FROM
            graph 'source_graph',
            graph 'target_graph',
            embedding,
            embedding_component,
            chain

        WHERE
            embedding.source_id = source_graph.id AND
            embedding.target_id = target_graph.id AND
            embedding_component.chain_id = chain.id AND
            embedding.id = embedding_component.embedding_id;
    """
