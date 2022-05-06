# Copyright 2022 D-Wave Systems Inc.
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
from random import shuffle

from dwave.embedding.zephyr import find_clique_embedding, find_biclique_embedding
from dwave.embedding.diagnostic import is_valid_embedding
from dwave_networkx.generators.zephyr import zephyr_graph
import networkx as nx
from parameterized import parameterized


class Test_find_clique_embedding(unittest.TestCase):
    def test_k_parameter_int(self):
        k_int = 5
        m = 3

        # Find embedding
        ze = zephyr_graph(m, coordinates=True)
        embedding = find_clique_embedding(k_int, target_graph=ze)

        self.assertTrue(is_valid_embedding(embedding, nx.complete_graph(k_int), ze))

    def test_k_parameter_list(self):
        k_nodes = ['one', 'two', 'three']
        m = 4

        # Find embedding
        ze = zephyr_graph(m, coordinates=True)
        embedding = find_clique_embedding(k_nodes, target_graph=ze)

        self.assertTrue(is_valid_embedding(embedding, nx.complete_graph(k_nodes), ze))

    def test_k_parameter_graph(self):
        k_graph = nx.complete_graph(10)
        m = 4

        # Find embedding
        ze = zephyr_graph(m, coordinates=True)
        embedding = find_clique_embedding(k_graph, target_graph=ze)

        self.assertTrue(is_valid_embedding(embedding, nx.complete_graph(k_graph), ze))

    def test_zero_clique(self):
        k = 0
        m = 3

        # Find embedding
        ze = zephyr_graph(m)
        embedding = find_clique_embedding(k, target_graph=ze)

        self.assertEqual(k, len(embedding))
        self.assertEqual({}, embedding)

    def test_one_clique(self):
        k = 1
        m = 4

        # Find embedding
        ze = zephyr_graph(m, coordinates=True)
        embedding = find_clique_embedding(k, target_graph=ze)

        self.assertTrue(is_valid_embedding(embedding, nx.complete_graph(k), ze))

    def test_valid_clique_ints(self):
        k = nx.complete_graph(15)
        m = 4

        # Find embedding
        ze = zephyr_graph(m)
        embedding = find_clique_embedding(k, target_graph=ze)

        self.assertTrue(is_valid_embedding(embedding, k, ze))

    def test_valid_clique_coord(self):
        k = nx.complete_graph(15)
        m = 4

        # Find embedding
        ze = zephyr_graph(m, coordinates=True)
        embedding = find_clique_embedding(k, target_graph=ze)

        self.assertTrue(is_valid_embedding(embedding, k, ze))

    def test_impossible_clique(self):
        k = 55
        m = 2

        # Find embedding
        ze = zephyr_graph(m)

        with self.assertRaises(ValueError):
            find_clique_embedding(k, target_graph=ze)

    def test_clique_incomplete_graph(self):
        k = 5
        m = 2

        # Nodes in a known K5 embedding
        # Note: {0: (16, 96), 1: (18, 98), 2: (20, 100), 3: (22, 102), 4: (24, 104)}
        known_embedding_nodes = {16, 96, 18, 98, 20, 100, 22, 102, 24, 104}

        # Create graph with missing nodes
        incomplete_ze = zephyr_graph(m)
        removed_nodes = set(incomplete_ze.nodes) - known_embedding_nodes
        incomplete_ze.remove_nodes_from(removed_nodes)

        # See if clique embedding is found
        embedding = find_clique_embedding(k, target_graph=incomplete_ze)
        self.assertTrue(is_valid_embedding(embedding, nx.complete_graph(k), incomplete_ze))

    def test_clique_missing_edges(self):
        k = 9
        m = 2

        ze = zephyr_graph(m)

        # pick a random ordering of the edges
        edges = list(ze.edges())
        shuffle(edges)

        K = nx.complete_graph(k)

        # now delete edges, one at a time, until we can no longer embed K
        with self.assertRaises(ValueError):
            while 1:
                (u, v) = edges.pop()
                ze.remove_edge(u, v)

                # See if clique embedding is found
                embedding = find_clique_embedding(k, target_graph=ze)
                self.assertTrue(is_valid_embedding(embedding, K, ze))


class Test_find_biclique_embedding(unittest.TestCase):
    ABC_DE = (['a', 'b', 'c'], ['d', 'e'], 2)
    ABC_P = (['a', 'b', 'c'], 7, 2)
    N_DE = (4, ['d', 'e'], 2)

    @parameterized.expand(((6, 6, 2), (16, 16, 3), (4, 5, 1), ABC_DE, ABC_P, N_DE))
    def test_success(self, a, b, m):
        left, right = find_biclique_embedding(a, b, m)

        # check that labels match args
        for arg, map_ in ((a, left), (b, right)):
            if isinstance(arg, int):
                self.assertEqual(len(map_), arg)
            else:
                self.assertEqual(set(map_), set(arg))

        # check that labels are disjoint
        self.assertEqual(set(left) | set(right), set(left) ^ set(right))

        # check that the embedding is valid
        ze = zephyr_graph(m)
        biclique = nx.complete_bipartite_graph(left, right)
        self.assertTrue(is_valid_embedding({**left, **right}, biclique, ze))

    @parameterized.expand((((1, 2), (2, 3)), ((1, 2), 2), (3, (2, 3)), (('a', 'b'), ('b', 'c'))))
    def test_overlapping_labels(self, a, b):
        m = 2

        # Find embedding
        ze = zephyr_graph(m)

        with self.assertRaises(ValueError):
            find_biclique_embedding(a, b, target_graph=ze)

    def test_impossible_biclique(self):
        a = 25
        b = 30
        m = 2

        # Find embedding
        ze = zephyr_graph(m)

        with self.assertRaises(ValueError):
            find_biclique_embedding(a, b, target_graph=ze)


if __name__ == "__main__":
    unittest.main()
