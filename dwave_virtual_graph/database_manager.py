import sqlite3
import json

from dwave_virtual_graph.cache_manager import cache_file
from dwave_virtual_graph.schema import schema


def cache_connect():
    """Returns a connection object to a sqlite database.

    Returns:
        :class:`sqlite3.Connection`

    """
    conn = sqlite3.connect(cache_file())

    # let us go ahead and populate the database with the tables we need. Each table
    # is only created if it does not already exist
    with conn as cur:
        for statement in schema:
            cur.execute(statement)

        # turn on foreign keys, allows deletes to cascade. That is if the graph
        # associated with a penaltymodel is removed, the penaltymodel will also
        # be removed. Also enforces that the graph exist when inserting.
        cur.execute("PRAGMA foreign_keys = ON;")

    return conn


def select_embedding(source_edges, target_edges):

    encoded_source = _edge_encode(source_edges)
    encoded_target = _edge_encode(target_edges)

    #
    # FOR NOW JUST BUILD EACH TIME
    #
    return None

    # select = \
    #     """SELECT embedding FROM

    #     """

    with cache_connect() as cur:
        sid = _graph_id(cur, encoded_source)
        tid = _graph_id(cur, encoded_target)

    print(sid, tid)


def _edge_encode(edgelist):
    num_edges = len(edgelist)
    num_nodes = len(set().union(*edgelist))
    edges = json.dumps(sorted(sorted(edge) for edge in edgelist), separators=(',', ':'))
    return {'num_nodes': num_nodes, 'num_edges': num_edges, 'edges': edges}


def _graph_id(cur, encoded_graph):
    """Get the unique id associated with each graph in the cache and add it
    to encoded_graph.
    Acts on the cursor so intended use is to be invoked inside a with statement.
    """

    select = \
        """SELECT id from graph WHERE
            num_nodes = :num_nodes
            AND num_edges = :num_edges
            AND edges = :edges;
        """
    insert = \
        """INSERT INTO graph(num_nodes, num_edges, edges) VALUES
            (
                :num_nodes,
                :num_edges,
                :edges
            );
        """

    row = cur.execute(select, encoded_graph).fetchone()

    # if it's not there, insert and re-query
    if row is None:
        cur.execute(insert, encoded_graph)
        row = cur.execute(select, encoded_graph).fetchone()

    # None is not iterable so this is self checking
    encoded_graph['graph_id'], = row
