import unittest

import networkx as nx
import dwave_networkx as dnx

import dwave_virtual_graph as dvg


class TestEmbeddingUtilities(unittest.TestCase):
    def test_self_embedding(self):
        """a 1-to-1 embedding should not change the adjacency"""
        target_adj = dnx.chimera_graph(4)
        embedding = {v: {v} for v in target_adj}

        source_adj = dvg.target_to_source(target_adj, embedding)

        # print(source_adj)

        # test the adjacencies are equal (source_adj is a dict and target_adj is a networkx graph)
        for v in target_adj:
            self.assertIn(v, source_adj)
            for u in target_adj[v]:
                self.assertIn(u, source_adj[v])

        for v in source_adj:
            self.assertIn(v, target_adj)
            for u in source_adj[v]:
                self.assertIn(u, target_adj[v])

    def test_embedding_to_one_node(self):
        """an embedding that maps everything to one node should result in a singleton graph"""
        target_adj = nx.barbell_graph(16, 7)
        embedding = {'a': set(target_adj)}  # all map to 'a'

        source_adj = dvg.target_to_source(target_adj, embedding)
        self.assertEqual(source_adj, {'a': set()})

        embedding = {'a': {0, 1}}  # not every node is assigned to a chain
        source_adj = dvg.target_to_source(target_adj, embedding)
        self.assertEqual(source_adj, {'a': set()})

    def test_embedding_overlap(self):
        target_adj = nx.complete_graph(5)
        embedding = {'a': {0, 1}, 'b': {1, 2}}  # overlap

        with self.assertRaises(ValueError):
            source_adj = dvg.target_to_source(target_adj, embedding)
