import unittest

import networkx as nx
import dwave_networkx as dnx

import dwave_virtual_graph as vg


class TestGetEmbedding(unittest.TestCase):
    """Note that these all assume correctness for minorminer's find_embedding function"""
    def test_smoke(self):
        """Check that nothing breaks"""
        target_graph = dnx.chimera_graph(4, 4, 4)
        source_graph = nx.Graph()
        source_graph.add_edges_from([('a', 'b'), ('b', 'c'), ('a', 'c')])

        vg.get_embedding(source_graph.edges, target_graph.edges)
