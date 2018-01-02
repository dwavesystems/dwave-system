import unittest
import time
import homebase

import dwave_virtual_graph as vg

tmp_database_name = 'tmp_test_database_manager_{}.db'.format(time.time())


class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        # new connection for just this test
        self.clean_connection = vg.cache_connect(':memory:')

        # connection that persists for all tests run in this file
        self.test_connection = vg.cache_connect(vg.cache_file(filename=tmp_database_name))

    def test_insert_chain(self):
        """insert and retrieve some chains"""
        conn = self.clean_connection

        chain = [0, 1, 3]

        # insert the chain twice, should only result in one net insert
        with conn as cur:
            vg.insert_chain(cur, chain)
            vg.insert_chain(cur, chain)

        with conn as cur:
            all_chains = list(vg.iter_chain(cur))
        self.assertEqual(len(all_chains), 1)
        (chain_length, returned_chain, id_), = all_chains
        self.assertEqual(chain_length, len(chain))
        self.assertEqual(returned_chain, chain)

        # insert a new chain
        with conn as cur:
            vg.insert_chain(cur, [0, 2, 3])
        with conn as cur:
            all_chains = list(vg.iter_chain(cur))
        self.assertEqual(len(all_chains), 2)  # should now be two chains

    def test_insert_flux_bias(self):
        conn = self.clean_connection

        with conn as cur:
            vg.insert_flux_bias(cur, [0, 1, 2], 'test_system', .1)

            # now try dumping the chain, should just be one
            all_chains = list(vg.iter_chain(cur))
            self.assertEqual(len(all_chains), 1)
            (chain_length, returned_chain, id_), = all_chains
            self.assertEqual(chain_length, len([0, 1, 2]))
            self.assertEqual(returned_chain, [0, 1, 2])

        # now insert one with the same system but a different chain
        with conn as cur:
            vg.insert_flux_bias(cur, [0, 2, 3], 'test_system', .1)
            self.assertEqual(len(list(vg.iter_flux_bias(cur))), 2)

        # now if we re-do an insert with a different flux bias, the newer
        # one should be the only one
        with conn as cur:
            vg.insert_flux_bias(cur, [0, 2, 3], 'test_system', .2)
            all_biases = list(vg.iter_flux_bias(cur))
            self.assertEqual(len(all_biases), 2)

            # now select over stuff inserted 10 seconds into the
            # future, should return nothing
            all_biases = list(vg.iter_flux_bias(cur, age=-10))
            self.assertEqual(len(all_biases), 0)

    def test_insert_graph(self):
        conn = self.clean_connection

        # inserting the same twice should result in only one graph
        with conn as cur:
            vg.insert_graph(cur, [[0, 1], [1, 2], [0, 2]])
            vg.insert_graph(cur, [[0, 1], [1, 2], [0, 2]])
            self.assertEqual(len(list(vg.iter_graph(cur))), 1)

    def test_insert_embedding(self):
        conn = self.test_connection

        source_graph = [[0, 1], [0, 2], [1, 2]]
        target_graph = [[0, 1], [0, 3], [1, 2], [2, 3]]
        embedding = {0: [0], 1: [1], 2: [2, 3]}

        with conn as cur:
            # insert then dump everything
            vg.insert_embedding(cur, source_graph, target_graph, embedding)
            returned_embeddings = list(vg.iter_embedding(cur))

            # shoud only be one thing and it should be equal to inserted
            self.assertEqual(len(returned_embeddings), 1)
            (source, target, emb), = returned_embeddings
            self.assertEqual(source, source_graph)
            self.assertEqual(target, target_graph)
            self.assertEqual(emb, embedding)

        # now try to reinsert, should raise an error
        with self.assertRaises(vg.UniqueEmbeddingTagError):
            with conn as cur:
                vg.insert_embedding(cur, source_graph, target_graph, embedding)

        # now let's add an embedding to a graph we already know
        source_graph = [[0, 1]]
        target_graph = [[0, 1], [0, 2], [1, 2]]
        embedding = {0: [0], 1: [1, 2]}
        with conn as cur:
            vg.insert_embedding(cur, source_graph, target_graph, embedding)

            returned_embeddings = list(vg.iter_embedding(cur))
            self.assertEqual(len(returned_embeddings), 2)

    def tearDown(self):
        self.clean_connection.close()
        self.test_connection.close()
