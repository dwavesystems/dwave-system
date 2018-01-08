import unittest
import time
import homebase

import networkx as nx

import dwave_virtual_graph as vg
import dwave_virtual_graph.cache as vgcache

tmp_database_name = 'tmp_test_database_manager_{}.db'.format(time.time())
# test_database_path = vgcache.cache_file(filename=tmp_database_name)
# conn = vgcache.cache_connect(test_database_path)


class TestCacheManager(unittest.TestCase):
    def test_same_database(self):
        """multiple calls to get_database_path"""
        db1 = vgcache.cache_file()
        db2 = vgcache.cache_file()

        self.assertEqual(db1, db2)


class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        # new connection for just this test
        self.clean_connection = vgcache.cache_connect(':memory:')

        # test_database_path = vgcache.cache_file(filename=tmp_database_name)
        # test_connection = vgcache.cache_connect(test_database_path)

    def tearDown(self):
        # close the memory connection
        self.clean_connection.close()

    def test_insert_retrieve_chain(self):
        """insert and retrieve some chains"""
        conn = self.clean_connection

        chain = [0, 1, 3]

        # insert the chain twice, should only result in one net insert
        with conn as cur:
            vgcache.insert_chain(cur, chain)
            vgcache.insert_chain(cur, chain)

        with conn as cur:
            all_chains = list(vgcache.iter_chain(cur))
        self.assertEqual(len(all_chains), 1)
        returned_chain, = all_chains
        self.assertEqual(len(returned_chain), len(chain))
        self.assertEqual(returned_chain, chain)

        # insert a new chain
        with conn as cur:
            vgcache.insert_chain(cur, [0, 2, 3])
        with conn as cur:
            all_chains = list(vgcache.iter_chain(cur))
        self.assertEqual(len(all_chains), 2)  # should now be two chains

    def test_insert_flux_bias(self):
        conn = self.clean_connection

        with conn as cur:
            vgcache.insert_flux_bias(cur, [0, 1, 2], 'test_system', .1, 1)

        with conn as cur:
            # try dumping
            flux_biases = list(vgcache.iter_flux_bias(cur))
        self.assertEqual(len(flux_biases), 1)
        self.assertEqual(flux_biases[0], ([0, 1, 2], 'test_system', .1, 1))

        # retrieve by name
        with conn as cur:
            biases = vgcache.get_flux_biases_from_cache(cur, [[0, 1, 2]], 'test_system', 1)

        for v, fbo in biases:
            self.assertIn(v, [0, 1, 2])
            self.assertEqual(fbo, .1)

        # now get something wrong out
        with self.assertRaises(vg.MissingFluxBias):
            with conn as cur:
                biases = vgcache.get_flux_biases_from_cache(cur, [[0, 1, 2]], 'another_system', 1)

    def test_graph_insert_retrieve(self):
        conn = self.clean_connection

        graph = nx.barbell_graph(8, 8)
        nodelist = sorted(graph)
        edgelist = sorted(sorted(edge) for edge in graph.edges)

        with conn as cur:
            vgcache.insert_graph(cur, nodelist, edgelist)

            # should only be one graph
            graphs = list(vgcache.iter_graph(cur))
            self.assertEqual(len(graphs), 1)
            (nodelist_, edgelist_), = graphs
            self.assertEqual(nodelist, nodelist_)
            self.assertEqual(edgelist, edgelist_)

        # trying to reinsert should still result in only one graph
        with conn as cur:
            vgcache.insert_graph(cur, nodelist, edgelist)
            graphs = list(vgcache.iter_graph(cur))
            self.assertEqual(len(graphs), 1)

        # inserting with an empty dict as encoded_data should populate it
        encoded_data = {}
        with conn as cur:
            vgcache.insert_graph(cur, nodelist, edgelist, encoded_data)
        self.assertIn('num_nodes', encoded_data)
        self.assertIn('num_edges', encoded_data)
        self.assertIn('edges', encoded_data)

        # now adding another graph should result in two items
        graph = nx.complete_graph(4)
        nodelist = sorted(graph)
        edgelist = sorted(sorted(edge) for edge in graph.edges)
        with conn as cur:
            vgcache.insert_graph(cur, nodelist, edgelist)
            graphs = list(vgcache.iter_graph(cur))
            self.assertEqual(len(graphs), 2)

#     def test_insert_embedding(self):
#         conn = self.test_connection

#         source_graph = [[0, 1], [0, 2], [1, 2]]
#         target_graph = [[0, 1], [0, 3], [1, 2], [2, 3]]
#         embedding = {0: [0], 1: [1], 2: [2, 3]}

#         with conn as cur:
#             # insert then dump everything
#             vgcache.insert_embedding(cur, source_graph, target_graph, embedding)
#             returned_embeddings = list(vgcache.iter_embedding(cur))

#             # shoud only be one thing and it should be equal to inserted
#             self.assertEqual(len(returned_embeddings), 1)
#             (source, target, emb), = returned_embeddings
#             self.assertEqual(source, source_graph)
#             self.assertEqual(target, target_graph)
#             self.assertEqual(emb, embedding)

#         # now try to reinsert, should raise an error
#         with self.assertRaises(vgcache.UniqueEmbeddingTagError):
#             with conn as cur:
#                 vgcache.insert_embedding(cur, source_graph, target_graph, embedding)

#         # now let's add an embedding to a graph we already know
#         source_graph = [[0, 1]]
#         target_graph = [[0, 1], [0, 2], [1, 2]]
#         embedding = {0: [0], 1: [1, 2]}
#         with conn as cur:
#             vgcache.insert_embedding(cur, source_graph, target_graph, embedding)

#             returned_embeddings = list(vgcache.iter_embedding(cur))
#             self.assertEqual(len(returned_embeddings), 2)

#     def tearDown(self):
#         self.clean_connection.close()
#         self.test_connection.close()
