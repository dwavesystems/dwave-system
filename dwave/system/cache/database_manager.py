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
"""Utilities for accessing the sqlite cache.

Note:
    All methods assume that nodes are integer-labeled.

"""
import sqlite3
import json
import os
import struct
import base64
import datetime

from six import iteritems
from six.moves import range

from dwave.system.cache.cache_manager import cache_file
from dwave.system.cache.schema import schema
from dwave.system.exceptions import MissingFluxBias


def cache_connect(database=None):
    """Returns a connection object to a sqlite database.

    Args:
        database (str, optional): The path to the database the user wishes
            to connect to. If not specified, a default is chosen using
            :func:`.cache_file`. If the special database name ':memory:'
            is given, then a temporary database is created in memory.

    Returns:
        :class:`sqlite3.Connection`

    """
    if database is None:
        database = cache_file()

    if os.path.isfile(database):
        # just connect to the database as-is
        conn = sqlite3.connect(database)
    else:
        # we need to populate the database
        conn = sqlite3.connect(database)
        conn.executescript(schema)

    with conn as cur:
        # turn on foreign keys, allows deletes to cascade.
        cur.execute("PRAGMA foreign_keys = ON;")

    conn.row_factory = sqlite3.Row

    return conn


def insert_chain(cur, chain, encoded_data=None):
    """Insert a chain into the cache.

    Args:
        cur (:class:`sqlite3.Cursor`):
            An sqlite3 cursor. This function is meant to be run within a :obj:`with` statement.

        chain (iterable):
            A collection of nodes. Chains in embedding act as one node.

        encoded_data (dict, optional):
            If a dictionary is provided, it will be populated with the serialized data. This is
            useful for preventing encoding the same information many times.

    Notes:
        This function assumes that the nodes in chain are index-labeled.

    """
    if encoded_data is None:
        encoded_data = {}

    if 'nodes' not in encoded_data:
        encoded_data['nodes'] = json.dumps(sorted(chain), separators=(',', ':'))
    if 'chain_length' not in encoded_data:
        encoded_data['chain_length'] = len(chain)

    insert = "INSERT OR IGNORE INTO chain(chain_length, nodes) VALUES (:chain_length, :nodes);"

    cur.execute(insert, encoded_data)


def iter_chain(cur):
    """Iterate over all of the chains in the database.

    Args:
        cur (:class:`sqlite3.Cursor`):
            An sqlite3 cursor. This function is meant to be run within a :obj:`with` statement.

    Yields:
        list: The chain.

    """
    select = "SELECT nodes FROM chain"
    for nodes, in cur.execute(select):
        yield json.loads(nodes)


def insert_system(cur, system_name, encoded_data=None):
    """Insert a system name into the cache.

    Args:
        cur (:class:`sqlite3.Cursor`):
            An sqlite3 cursor. This function is meant to be run within a :obj:`with` statement.

        system_name (str):
            The unique name of a system

        encoded_data (dict, optional):
            If a dictionary is provided, it will be populated with the serialized data. This is
            useful for preventing encoding the same information many times.

    """
    if encoded_data is None:
        encoded_data = {}

    if 'system_name' not in encoded_data:
        encoded_data['system_name'] = system_name

    insert = "INSERT OR IGNORE INTO system(system_name) VALUES (:system_name);"
    cur.execute(insert, encoded_data)


def iter_system(cur):
    """Iterate over all system names in the cache.

    Args:
        cur (:class:`sqlite3.Cursor`):
            An sqlite3 cursor. This function is meant to be run within a :obj:`with` statement.

    Yields:
        str: system name

    """
    select = "SELECT system_name FROM system"
    for system_name, in cur.execute(select):
        yield system_name


def insert_flux_bias(cur, chain, system, flux_bias, chain_strength, encoded_data=None):
    """Insert a flux bias offset into the cache.

    Args:
        cur (:class:`sqlite3.Cursor`):
            An sqlite3 cursor. This function is meant to be run within a :obj:`with` statement.

        chain (iterable):
            A collection of nodes. Chains in embedding act as one node.

        system (str):
            The unique name of a system.

        flux_bias (float):
            The flux bias offset associated with the given chain.

        chain_strength (float):
            The magnitude of the negative quadratic bias that induces the given chain in an Ising
            problem.

        encoded_data (dict, optional):
            If a dictionary is provided, it will be populated with the serialized data. This is
            useful for preventing encoding the same information many times.

    """
    if encoded_data is None:
        encoded_data = {}

    insert_chain(cur, chain, encoded_data)
    insert_system(cur, system, encoded_data)

    if 'flux_bias' not in encoded_data:
        encoded_data['flux_bias'] = _encode_real(flux_bias)
    if 'chain_strength' not in encoded_data:
        encoded_data['chain_strength'] = _encode_real(chain_strength)
    if 'insert_time' not in encoded_data:
        encoded_data['insert_time'] = datetime.datetime.now()

    insert = \
        """
        INSERT OR REPLACE INTO flux_bias(chain_id, system_id, insert_time, flux_bias, chain_strength)
        SELECT
            chain.id,
            system.id,
            :insert_time,
            :flux_bias,
            :chain_strength
        FROM chain, system
        WHERE
            chain.chain_length = :chain_length AND
            chain.nodes = :nodes AND
            system.system_name = :system_name;
        """

    cur.execute(insert, encoded_data)


def iter_flux_bias(cur):
    """Iterate over all flux biases in the cache.

    Args:
        cur (:class:`sqlite3.Cursor`):
            An sqlite3 cursor. This function is meant to be run within a :obj:`with` statement.

    Yields:
        tuple: A 4-tuple:

            list: The chain.

            str: The system name.

            float: The flux bias associated with the chain.

            float: The chain strength associated with the chain.

    """
    select = \
        """
        SELECT nodes, system_name, flux_bias, chain_strength FROM flux_bias_view;
        """

    for nodes, system, flux_bias, chain_strength in cur.execute(select):
        yield json.loads(nodes), system, _decode_real(flux_bias), _decode_real(chain_strength)


def _encode_real(v):
    """Encode real numbers as base 64 encoded little endian 8 byte floats."""
    bytes_ = struct.pack('<d', v)
    return base64.b64encode(bytes_)


def _decode_real(blob):
    """Inverse of _encode_real."""
    bytes_ = base64.b64decode(blob)
    return struct.unpack('<d', bytes_)[0]


def get_flux_biases_from_cache(cur, chains, system_name, chain_strength, max_age=3600):
    """Determine the flux biases for all of the the given chains, system and chain strength.

    Args:
        cur (:class:`sqlite3.Cursor`):
            An sqlite3 cursor. This function is meant to be run within a :obj:`with` statement.

        chains (iterable):
            An iterable of chains. Each chain is a collection of nodes. Chains in embedding act as
            one node.

        system_name (str):
            The unique name of a system.

        chain_strength (float):
            The magnitude of the negative quadratic bias that induces the given chain in an Ising
            problem.

        max_age (int, optional, default=3600):
            The maximum age (in seconds) for the flux_bias offsets.

    Returns:
        dict: A dict where the keys are the nodes in the chains and the values are the flux biases.

    """

    select = \
        """
        SELECT
            flux_bias
        FROM flux_bias_view WHERE
            chain_length = :chain_length AND
            nodes = :nodes AND
            chain_strength = :chain_strength AND
            system_name = :system_name AND
            insert_time >= :time_limit;
        """

    encoded_data = {'chain_strength': _encode_real(chain_strength),
                    'system_name': system_name,
                    'time_limit': datetime.datetime.now() + datetime.timedelta(seconds=-max_age)}

    flux_biases = {}
    for chain in chains:
        encoded_data['chain_length'] = len(chain)
        encoded_data['nodes'] = json.dumps(sorted(chain), separators=(',', ':'))

        row = cur.execute(select, encoded_data).fetchone()
        if row is None:
            raise MissingFluxBias
        flux_bias = _decode_real(*row)

        if flux_bias == 0:
            continue

        flux_biases.update({v: flux_bias for v in chain})

    return flux_biases


def insert_graph(cur, nodelist, edgelist, encoded_data=None):
    """Insert a graph into the cache.

    A graph is stored by number of nodes, number of edges and a
    json-encoded list of edges.

    Args:
        cur (:class:`sqlite3.Cursor`): An sqlite3 cursor. This function
            is meant to be run within a :obj:`with` statement.
        nodelist (list): The nodes in the graph.
        edgelist (list): The edges in the graph.
        encoded_data (dict, optional): If a dictionary is provided, it
            will be populated with the serialized data. This is useful for
            preventing encoding the same information many times.

    Notes:
        This function assumes that the nodes are index-labeled and range
        from 0 to num_nodes - 1.

        In order to minimize the total size of the cache, it is a good
        idea to sort the nodelist and edgelist before inserting.

    Examples:
        >>> nodelist = [0, 1, 2]
        >>> edgelist = [(0, 1), (1, 2)]
        >>> with pmc.cache_connect(':memory:') as cur:
        ...     pmc.insert_graph(cur, nodelist, edgelist)

        >>> nodelist = [0, 1, 2]
        >>> edgelist = [(0, 1), (1, 2)]
        >>> encoded_data = {}
        >>> with pmc.cache_connect(':memory:') as cur:
        ...     pmc.insert_graph(cur, nodelist, edgelist, encoded_data)
        >>> encoded_data['num_nodes']
        3
        >>> encoded_data['num_edges']
        2
        >>> encoded_data['edges']
        '[[0,1],[1,2]]'

    """
    if encoded_data is None:
        encoded_data = {}

    if 'num_nodes' not in encoded_data:
        encoded_data['num_nodes'] = len(nodelist)
    if 'num_edges' not in encoded_data:
        encoded_data['num_edges'] = len(edgelist)
    if 'edges' not in encoded_data:
        encoded_data['edges'] = json.dumps(edgelist, separators=(',', ':'))

    insert = \
        """
        INSERT OR IGNORE INTO graph(num_nodes, num_edges, edges)
        VALUES (:num_nodes, :num_edges, :edges);
        """

    cur.execute(insert, encoded_data)


def iter_graph(cur):
    """Iterate over all graphs in the cache.

    Args:
        cur (:class:`sqlite3.Cursor`): An sqlite3 cursor. This function
            is meant to be run within a :obj:`with` statement.

    Yields:
        tuple: A 2-tuple containing:

            list: The nodelist for a graph in the cache.

            list: the edgelist for a graph in the cache.

    Examples:
        >>> nodelist = [0, 1, 2]
        >>> edgelist = [(0, 1), (1, 2)]
        >>> with pmc.cache_connect(':memory:') as cur:
        ...     pmc.insert_graph(cur, nodelist, edgelist)
        ...     list(pmc.iter_graph(cur))
        [([0, 1, 2], [[0, 1], [1, 2]])]

    """
    select = """SELECT num_nodes, num_edges, edges from graph;"""
    for num_nodes, num_edges, edges in cur.execute(select):
        yield list(range(num_nodes)), json.loads(edges)


def insert_embedding(cur, source_nodelist, source_edgelist, target_nodelist, target_edgelist,
                     embedding, embedding_tag):
    """Insert an embedding into the cache.

    Args:
        cur (:class:`sqlite3.Cursor`):
            An sqlite3 cursor. This function is meant to be run within a :obj:`with` statement.

        source_nodelist (list):
            The nodes in the source graph. Should be integer valued.

        source_edgelist (list):
            The edges in the source graph.

        target_nodelist (list):
            The nodes in the target graph. Should be integer valued.

        target_edgelist (list):
            The edges in the target graph.

        embedding (dict):
            The mapping from the source graph to the target graph.
            Should be of the form {v: {s, ...}, ...} where v is a variable in the
            source model and s is a variable in the target model.

        embedding_tag (str):
            A string tag to associate with the embedding.

    """
    encoded_data = {}

    # first we need to encode the graphs and create the embedding id

    source_data = {}
    insert_graph(cur, source_nodelist, source_edgelist, source_data)
    encoded_data['source_edges'] = source_data['edges']
    encoded_data['source_num_nodes'] = source_data['num_nodes']
    encoded_data['source_num_edges'] = source_data['num_edges']

    target_data = {}
    insert_graph(cur, target_nodelist, target_edgelist, target_data)
    encoded_data['target_edges'] = target_data['edges']
    encoded_data['target_num_nodes'] = target_data['num_nodes']
    encoded_data['target_num_edges'] = target_data['num_edges']

    encoded_data['tag'] = embedding_tag

    insert_embedding = \
        """
        INSERT OR REPLACE INTO embedding(
            source_id,
            target_id,
            tag)
        SELECT
            source_graph.id,
            target_graph.id,
            :tag
        FROM
            graph 'source_graph',
            graph 'target_graph'
        WHERE
            source_graph.edges = :source_edges AND
            source_graph.num_nodes = :source_num_nodes AND
            source_graph.num_edges = :source_num_edges AND
            target_graph.edges = :target_edges AND
            target_graph.num_nodes = :target_num_nodes AND
            target_graph.num_edges = :target_num_edges
        """

    cur.execute(insert_embedding, encoded_data)

    # now each chain needs to be inserted

    insert_embedding_component = \
        """
        INSERT OR REPLACE INTO embedding_component(
            source_node,
            chain_id,
            embedding_id)
        SELECT
            :source_node,
            chain.id,
            embedding.id
        FROM
            graph 'source_graph',
            graph 'target_graph',
            chain,
            embedding
        WHERE
            source_graph.edges = :source_edges AND
            source_graph.num_nodes = :source_num_nodes AND
            target_graph.edges = :target_edges AND
            target_graph.num_nodes = :target_num_nodes AND
            embedding.source_id = source_graph.id AND
            embedding.target_id = target_graph.id AND
            embedding.tag = :tag AND
            chain.nodes = :nodes AND
            chain.chain_length = :chain_length
        """

    for v, chain in iteritems(embedding):
        chain_data = {'source_node': v}
        insert_chain(cur, chain, chain_data)

        encoded_data.update(chain_data)

        cur.execute(insert_embedding_component, encoded_data)


def select_embedding_from_tag(cur, embedding_tag, target_nodelist, target_edgelist):
    """Select an embedding from the given tag and target graph.

    Args:
        cur (:class:`sqlite3.Cursor`):
            An sqlite3 cursor. This function is meant to be run within a :obj:`with` statement.

        source_nodelist (list):
            The nodes in the source graph. Should be integer valued.

        source_edgelist (list):
            The edges in the source graph.

        target_nodelist (list):
            The nodes in the target graph. Should be integer valued.

        target_edgelist (list):
            The edges in the target graph.

    Returns:
        dict: The mapping from the source graph to the target graph.
        In the form {v: {s, ...}, ...} where v is a variable in the
        source model and s is a variable in the target model.

    """
    encoded_data = {'num_nodes': len(target_nodelist),
                    'num_edges': len(target_edgelist),
                    'edges': json.dumps(target_edgelist, separators=(',', ':')),
                    'tag': embedding_tag}

    select = \
        """
        SELECT
            source_node,
            chain
        FROM
            embedding_component_view
        WHERE
            embedding_tag = :tag AND
            target_edges = :edges AND
            target_num_nodes = :num_nodes AND
            target_num_edges = :num_edges
        """

    embedding = {v: json.loads(chain) for v, chain in cur.execute(select, encoded_data)}
    return embedding


def select_embedding_from_source(cur, source_nodelist, source_edgelist,
                                 target_nodelist, target_edgelist):
    """Select an embedding from the source graph and target graph.

    Args:
        cur (:class:`sqlite3.Cursor`):
            An sqlite3 cursor. This function is meant to be run within a :obj:`with` statement.

        target_nodelist (list):
            The nodes in the target graph. Should be integer valued.

        target_edgelist (list):
            The edges in the target graph.

        embedding_tag (str):
            A string tag to associate with the embedding.

    Returns:
        dict: The mapping from the source graph to the target graph.
        In the form {v: {s, ...}, ...} where v is a variable in the
        source model and s is a variable in the target model.

    """
    encoded_data = {'target_num_nodes': len(target_nodelist),
                    'target_num_edges': len(target_edgelist),
                    'target_edges': json.dumps(target_edgelist, separators=(',', ':')),
                    'source_num_nodes': len(source_nodelist),
                    'source_num_edges': len(source_edgelist),
                    'source_edges': json.dumps(source_edgelist, separators=(',', ':'))}

    select = \
        """
        SELECT
            source_node,
            chain
        FROM
            embedding_component_view
        WHERE
            source_num_edges = :source_num_edges AND
            source_edges = :source_edges AND
            source_num_nodes = :source_num_nodes AND

            target_num_edges = :target_num_edges AND
            target_edges = :target_edges AND
            target_num_nodes = :target_num_nodes
        """

    embedding = {v: json.loads(chain) for v, chain in cur.execute(select, encoded_data)}
    return embedding
