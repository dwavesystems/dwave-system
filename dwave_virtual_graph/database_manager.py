"""todo

Note:
    All methods assume that nodes are integer-labeled.
"""
import sqlite3
import json
import os

from dwave_virtual_graph.exceptions import UniqueEmbeddingTagError
from dwave_virtual_graph.cache_manager import cache_file
from dwave_virtual_graph.schema import schema


def iteritems(d):
    return d.items()


def cache_connect(database=cache_file()):
    """Returns a connection object to a sqlite database.

    Args:
        database (str, optional): The path to the database the user wishes
            to connect to. If not specified, a default is chosen.

    Returns:
        :class:`sqlite3.Connection`

    """
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

    return conn


def insert_chain(cur, chain):
    """Insert a chain into the database.

    todo
    """
    assert isinstance(chain, list)
    assert all(isinstance(v, int) for v in chain)

    insert = \
        """
        INSERT OR IGNORE INTO chain(chain_length, nodes) VALUES (?, ?);
        """
    cur.execute(insert, (len(chain), json.dumps(chain, separators=(',', ':'))))


def iter_chain(cur):
    """Iterate over all of the chains in the database.

    Args:
        cur (:class:`sqlite3.Cursor`): A cursor connecting
            to the database.

    Yields:
        tuple: A 3-tuple containing:

            int: The chain length.
            list: The chain.
            int: The id assigned by sql to the chain.

    """
    select = "SELECT chain_length, nodes, id FROM chain"
    for chain_length, nodes, id_ in cur.execute(select):
        yield chain_length, json.loads(nodes), id_


def insert_system(cur, system):
    """todo"""
    assert isinstance(system, str)

    insert = "INSERT OR IGNORE INTO system(name) VALUES (?);"
    cur.execute(insert, (system,))


def iter_system(cur):
    """todo"""
    select = "SELECT name id FROM system"
    for name, id_ in cur.execute(select):
        yield name, id_


def insert_flux_bias(cur, chain, system, flux_bias):
    """todo"""
    insert_chain(cur, chain)
    insert_system(cur, system)

    insert = \
        """
        INSERT OR REPLACE INTO flux_bias(chain_id, system_id, flux_bias)
        SELECT chain.id, system.id, ? FROM chain, system
        WHERE chain.nodes = ? AND system.name = ?;
        """

    cur.execute(insert, (flux_bias, json.dumps(chain, separators=(',', ':')), system))


def iter_flux_bias(cur, age=3600):
    """age in seconds"""
    select = \
        """
        SELECT nodes, system_name, flux_bias FROM flux_bias_view
        WHERE insert_time >= datetime('now', ?);
        """

    # NB: I am not doing a string insert into the select statement, I am constructing
    # the value that is then inserted using the '?'.
    time_modifier = '-{} seconds'.format(age)
    for nodes, system, bias in cur.execute(select, (time_modifier,)):
        yield json.loads(nodes), system, bias


def insert_graph(cur, edgelist, num_nodes=None):
    """todo"""
    assert isinstance(edgelist, list)
    # assert sorted

    if num_nodes is None:
        num_nodes = len(set().union(*edgelist))
    num_edges = len(edgelist)
    edges = json.dumps(edgelist, separators=(',', ':'))

    insert = "INSERT OR IGNORE INTO graph(num_nodes, num_edges, edges) VALUES (?, ?, ?);"
    cur.execute(insert, (num_nodes, num_edges, edges))


def iter_graph(cur):
    select = """SELECT num_nodes, num_edges, edges from graph;"""
    for num_nodes, num_edges, edges in cur.execute(select):
        yield num_nodes, num_edges, json.loads(edges)


def insert_embedding_tag(cur, source_edgelist, target_edgelist, embedding_tag=None):
    insert = \
        """
        INSERT INTO embedding(
            tag,
            source_id,
            target_id)
        SELECT
            ?,
            source_graph.id,
            target_graph.id
        FROM graph 'source_graph', graph 'target_graph'
        WHERE
            source_graph.edges = ? AND
            target_graph.edges = ?;
        """

    source_edges = json.dumps(source_edgelist, separators=(',', ':'))
    target_edges = json.dumps(target_edgelist, separators=(',', ':'))

    cur.execute(insert, (embedding_tag, source_edges, target_edges))


def insert_embedding(cur, source_edgelist, target_edgelist, embedding):
    """todo"""
    insert_graph(cur, source_edgelist)
    insert_graph(cur, target_edgelist)
    insert_embedding_tag(cur, source_edgelist, target_edgelist)

    insert = \
        """
        INSERT INTO embedding_chains(
            source_node,
            chain_id,
            embedding_id)
        SELECT
            ?,
            chain.id,
            embedding.id
        FROM graph 'source_graph', graph 'target_graph', chain, embedding
        WHERE
            source_graph.edges = ? AND
            target_graph.edges = ? AND
            chain.nodes = ? AND
            embedding.source_id = source_graph.id AND
            embedding.target_id = target_graph.id;
        """

    source_edges = json.dumps(source_edgelist, separators=(',', ':'))
    target_edges = json.dumps(target_edgelist, separators=(',', ':'))
    for v, chain in iteritems(embedding):
        insert_chain(cur, chain)

        chain = json.dumps(chain, separators=(',', ':'))
        cur.execute(insert, (v, source_edges, target_edges, chain))


def iter_embedding(cur):
    select = \
        """
        SELECT
            source_id,
            source_num_nodes,
            source_edges,
            target_id,
            target_num_nodes,
            target_edges,
            source_node,
            chain
        FROM embedding_view
        ORDER BY source_id, target_id;
        """

    graphs = {}
    embedding = None
    idx = (-1, -1)  # set to default
    for sid, source_num_nodes, source_edges, tid, target_num_nodes, target_edges, v, chain, in cur.execute(select):
        if idx == (sid, tid):
            # in this case we're still working on the same embedding
            embedding[v] = json.loads(chain)
        else:
            if embedding is not None:
                yield json.loads(source_edges), json.loads(target_edges), embedding

            # starting a new embedding
            embedding = {v: json.loads(chain)}
            idx = (sid, tid)

    yield json.loads(source_edges), json.loads(target_edges), embedding


# def select_embedding
