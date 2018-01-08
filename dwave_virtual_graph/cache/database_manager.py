"""todo

Note:
    All methods assume that nodes are integer-labeled.
"""
import sqlite3
import json
import os
import struct
import base64
import datetime

from dwave_virtual_graph.cache.cache_manager import cache_file
from dwave_virtual_graph.cache.schema import schema
from dwave_virtual_graph.exceptions import MissingFluxBias


def iteritems(d):
    return d.items()


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

        nodelist (list): The nodes in the graph.

        edgelist (list): The edges in the graph.

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
        tuple: A 3-tuple containing:

            int: The chain length.

            list: The chain.

            int: The id assigned by sql to the chain.

    """
    select = "SELECT nodes FROM chain"
    for nodes, in cur.execute(select):
        yield json.loads(nodes)


def insert_system(cur, system, encoded_data=None):
    """Insert a system name into the cache.

    Args:
        cur (:class:`sqlite3.Cursor`):
            An sqlite3 cursor. This function is meant to be run within a :obj:`with` statement.

        system (str):
            The unique name of a system

        encoded_data (dict, optional):
            If a dictionary is provided, it will be populated with the serialized data. This is
            useful for preventing encoding the same information many times.

    """
    if encoded_data is None:
        encoded_data = {}

    if 'system_name' not in encoded_data:
        encoded_data['system_name'] = system

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
    """todo"""
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
    """"""
    select = \
        """
        SELECT nodes, system_name, flux_bias, chain_strength FROM flux_bias_view;
        """

    for nodes, system, flux_bias, chain_strength in cur.execute(select):
        yield json.loads(nodes), system, _decode_real(flux_bias), _decode_real(chain_strength)


def _encode_real(v):
    bytes_ = struct.pack('<d', v)
    return base64.b64encode(bytes_)


def _decode_real(blob):
    bytes_ = base64.b64decode(blob)
    return struct.unpack('<d', bytes_)[0]


def get_flux_biases_from_cache(cur, chains, system_name, chain_strength, max_age=3600):

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

    flux_biases = []
    for chain in chains:
        encoded_data['chain_length'] = len(chain)
        encoded_data['nodes'] = json.dumps(sorted(chain), separators=(',', ':'))

        row = cur.execute(select, encoded_data).fetchone()
        if row is None:
            raise MissingFluxBias
        flux_bias = _decode_real(*row)

        flux_biases.extend([v, flux_bias] for v in chain)

    return flux_biases


# def insert_graph(cur, edgelist, num_nodes=None):
#     """todo"""
#     assert isinstance(edgelist, list)
#     # assert sorted

#     if num_nodes is None:
#         num_nodes = len(set().union(*edgelist))
#     num_edges = len(edgelist)
#     edges = json.dumps(edgelist, separators=(',', ':'))

#     insert = "INSERT OR IGNORE INTO graph(num_nodes, num_edges, edges) VALUES (?, ?, ?);"
#     cur.execute(insert, (num_nodes, num_edges, edges))


# def iter_graph(cur):
#     select = """SELECT num_nodes, num_edges, edges from graph;"""
#     for num_nodes, num_edges, edges in cur.execute(select):
#         yield num_nodes, num_edges, json.loads(edges)


# def insert_embedding_tag(cur, source_edgelist, target_edgelist, embedding_tag=None):
#     insert = \
#         """
#         INSERT INTO embedding(
#             tag,
#             source_id,
#             target_id)
#         SELECT
#             ?,
#             source_graph.id,
#             target_graph.id
#         FROM graph 'source_graph', graph 'target_graph'
#         WHERE
#             source_graph.edges = ? AND
#             target_graph.edges = ?;
#         """

#     source_edges = json.dumps(source_edgelist, separators=(',', ':'))
#     target_edges = json.dumps(target_edgelist, separators=(',', ':'))

#     cur.execute(insert, (embedding_tag, source_edges, target_edges))


# def insert_embedding(cur, source_edgelist, target_edgelist, embedding):
#     """todo"""
#     insert_graph(cur, source_edgelist)
#     insert_graph(cur, target_edgelist)
#     insert_embedding_tag(cur, source_edgelist, target_edgelist)

#     insert = \
#         """
#         INSERT INTO embedding_chains(
#             source_node,
#             chain_id,
#             embedding_id)
#         SELECT
#             ?,
#             chain.id,
#             embedding.id
#         FROM graph 'source_graph', graph 'target_graph', chain, embedding
#         WHERE
#             source_graph.edges = ? AND
#             target_graph.edges = ? AND
#             chain.nodes = ? AND
#             embedding.source_id = source_graph.id AND
#             embedding.target_id = target_graph.id;
#         """

#     source_edges = json.dumps(source_edgelist, separators=(',', ':'))
#     target_edges = json.dumps(target_edgelist, separators=(',', ':'))
#     for v, chain in iteritems(embedding):
#         insert_chain(cur, chain)

#         chain = json.dumps(chain, separators=(',', ':'))
#         cur.execute(insert, (v, source_edges, target_edges, chain))


# def iter_embedding(cur):
#     select = \
#         """
#         SELECT
#             source_id,
#             source_num_nodes,
#             source_edges,
#             target_id,
#             target_num_nodes,
#             target_edges,
#             source_node,
#             chain
#         FROM embedding_view
#         ORDER BY source_id, target_id;
#         """

#     graphs = {}
#     embedding = None
#     idx = (-1, -1)  # set to default
#     for sid, source_num_nodes, source_edges, tid, target_num_nodes, target_edges, v, chain, in cur.execute(select):
#         if idx == (sid, tid):
#             # in this case we're still working on the same embedding
#             embedding[v] = json.loads(chain)
#         else:
#             if embedding is not None:
#                 yield json.loads(source_edges), json.loads(target_edges), embedding

#             # starting a new embedding
#             embedding = {v: json.loads(chain)}
#             idx = (sid, tid)

#     yield json.loads(source_edges), json.loads(target_edges), embedding


# # def select_embedding
