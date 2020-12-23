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

import unittest
import time
import homebase

import networkx as nx

from dwave.system.exceptions import MissingFluxBias
import dwave.system.cache as cache

tmp_database_name = 'tmp_test_database_manager_{}.db'.format(time.time())
# test_database_path = cache.cache_file(filename=tmp_database_name)
# conn = cache.cache_connect(test_database_path)


class TestCacheManager(unittest.TestCase):
    def test_same_database(self):
        """multiple calls to get_database_path"""
        db1 = cache.cache_file()
        db2 = cache.cache_file()

        self.assertEqual(db1, db2)


class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        # new connection for just this test
        self.clean_connection = cache.cache_connect(':memory:')

        # test_database_path = cache.cache_file(filename=tmp_database_name)
        # test_connection = cache.cache_connect(test_database_path)

    def tearDown(self):
        # close the memory connection
        self.clean_connection.close()

    def test_insert_retrieve_chain(self):
        """insert and retrieve some chains"""
        conn = self.clean_connection

        chain = [0, 1, 3]

        # insert the chain twice, should only result in one net insert
        with conn as cur:
            cache.insert_chain(cur, chain)
            cache.insert_chain(cur, chain)

        with conn as cur:
            all_chains = list(cache.iter_chain(cur))
        self.assertEqual(len(all_chains), 1)
        returned_chain, = all_chains
        self.assertEqual(len(returned_chain), len(chain))
        self.assertEqual(returned_chain, chain)

        # insert a new chain
        with conn as cur:
            cache.insert_chain(cur, [0, 2, 3])
        with conn as cur:
            all_chains = list(cache.iter_chain(cur))
        self.assertEqual(len(all_chains), 2)  # should now be two chains

    def test_insert_flux_bias(self):
        conn = self.clean_connection

        with conn as cur:
            cache.insert_flux_bias(cur, [0, 1, 2], 'test_system', .1, 1)

        with conn as cur:
            # try dumping
            flux_biases = list(cache.iter_flux_bias(cur))
        self.assertEqual(len(flux_biases), 1)
        self.assertEqual(flux_biases[0], ([0, 1, 2], 'test_system', .1, 1))

        # retrieve by name
        with conn as cur:
            biases = cache.get_flux_biases_from_cache(cur, [[0, 1, 2]], 'test_system', 1)

        for v, fbo in biases.items():
            self.assertIn(v, [0, 1, 2])
            self.assertEqual(fbo, .1)

        # now get something wrong out
        with self.assertRaises(MissingFluxBias):
            with conn as cur:
                biases = cache.get_flux_biases_from_cache(cur, [[0, 1, 2]], 'another_system', 1)

    def test_graph_insert_retrieve(self):
        conn = self.clean_connection

        graph = nx.barbell_graph(8, 8)
        nodelist = sorted(graph)
        edgelist = sorted(sorted(edge) for edge in graph.edges)

        with conn as cur:
            cache.insert_graph(cur, nodelist, edgelist)

            # should only be one graph
            graphs = list(cache.iter_graph(cur))
            self.assertEqual(len(graphs), 1)
            (nodelist_, edgelist_), = graphs
            self.assertEqual(nodelist, nodelist_)
            self.assertEqual(edgelist, edgelist_)

        # trying to reinsert should still result in only one graph
        with conn as cur:
            cache.insert_graph(cur, nodelist, edgelist)
            graphs = list(cache.iter_graph(cur))
            self.assertEqual(len(graphs), 1)

        # inserting with an empty dict as encoded_data should populate it
        encoded_data = {}
        with conn as cur:
            cache.insert_graph(cur, nodelist, edgelist, encoded_data)
        self.assertIn('num_nodes', encoded_data)
        self.assertIn('num_edges', encoded_data)
        self.assertIn('edges', encoded_data)

        # now adding another graph should result in two items
        graph = nx.complete_graph(4)
        nodelist = sorted(graph)
        edgelist = sorted(sorted(edge) for edge in graph.edges)
        with conn as cur:
            cache.insert_graph(cur, nodelist, edgelist)
            graphs = list(cache.iter_graph(cur))
            self.assertEqual(len(graphs), 2)

    def test_insert_embedding(self):
        test_database_path = cache.cache_file(filename=tmp_database_name)
        conn = cache.cache_connect(test_database_path)

        source_nodes = [0, 1, 2]
        source_edges = [[0, 1], [0, 2], [1, 2]]
        target_nodes = [0, 1, 2, 3]
        target_edges = [[0, 1], [0, 3], [1, 2], [2, 3]]

        embedding = {0: [0], 1: [1], 2: [2, 3]}

        with conn as cur:
            cache.insert_embedding(cur, source_nodes, source_edges, target_nodes, target_edges,
                                     embedding, 'tag1')

            embedding_ = cache.select_embedding_from_tag(cur, 'tag1', target_nodes, target_edges)

            self.assertEqual(embedding, embedding_)

        # now reinsert but with a different embedding
        embedding = {0: [0, 1], 1: [2], 2: [3]}
        with conn as cur:
            cache.insert_embedding(cur, source_nodes, source_edges, target_nodes, target_edges,
                                     embedding, 'tag1')

            # get it back
            embedding_ = cache.select_embedding_from_tag(cur, 'tag1', target_nodes, target_edges)

            self.assertEqual(embedding, embedding_)

            # get it back from source graph
            embedding_ = cache.select_embedding_from_source(cur, source_nodes, source_edges,
                                                              target_nodes, target_edges)
            self.assertEqual(embedding, embedding_)
