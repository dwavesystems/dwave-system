# Copyright 2016 D-Wave Systems Inc.
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

import unittest

import networkx as nx
import dwave_networkx as dnx
import dimod

import dwave.embedding

from dwave.embedding.chimera import find_clique_embedding, find_biclique_embedding, find_grid_embedding


class Test_find_clique_embedding(unittest.TestCase):
    def test_full_yield_one_tile_k3(self):
        emb = find_clique_embedding(3, 1)

        target = dnx.chimera_graph(1)

        source = dwave.embedding.target_to_source(target, emb)

        self.assertEqual(source, {0: {1, 2}, 1: {0, 2}, 2: {0, 1}})

    def test_full_yield_one_tile_k2(self):
        emb = find_clique_embedding(2, 1)

        target = dnx.chimera_graph(1)

        source = dwave.embedding.target_to_source(target, emb)

        self.assertEqual(source, {0: {1}, 1: {0}})

    def test_str_labels(self):
        emb = find_clique_embedding(['a', 'b'], 1)

        self.assertEqual(len(emb), 2)
        self.assertIn('a', emb)
        self.assertIn('b', emb)


class Test_find_biclique_embedding(unittest.TestCase):
    def test_full_yield_one_tile_k44(self):
        left, right = find_biclique_embedding(4, 4, 1)
        # smoke test for now


class TestFindGridEmbedding(unittest.TestCase):
    def test_3d_2x2x2_on_c2(self):
        embedding = find_grid_embedding([2, 2, 2], 2, 2, t=4)

        # should be 4 grids
        self.assertEqual(len(embedding), 2*2*2)

        target_adj = dwave.embedding.target_to_source(dnx.chimera_graph(2), embedding)

        G = nx.grid_graph(dim=[2, 2, 2])
        for u in G.adj:
            for v in G.adj[u]:
                self.assertIn(u, target_adj)
                self.assertIn(v, target_adj[u])

        for u in target_adj:
            for v in target_adj[u]:
                self.assertIn(u, G.adj)
                self.assertIn(v, G.adj[u])

    def test_1d_3_on_c16(self):
        embedding = find_grid_embedding([3], 16)

        self.assertEqual(len(embedding), 3)

        target_adj = dwave.embedding.target_to_source(dnx.chimera_graph(16), embedding)

        G = nx.path_graph(3)
        for u in G.adj:
            for v in G.adj[u]:
                self.assertIn(u, target_adj)
                self.assertIn(v, target_adj[u])

        for u in target_adj:
            for v in target_adj[u]:
                self.assertIn(u, G.adj)
                self.assertIn(v, G.adj[u])

    def test_2d_6x4_on_c6(self):
        dims = [3, 2]
        chimera = (3,)

        embedding = find_grid_embedding(dims, *chimera)

        self.assertEqual(len(embedding), self.prod(dims))

        target_adj = dwave.embedding.target_to_source(dnx.chimera_graph(*chimera), embedding)

        G = nx.grid_graph(list(reversed(dims)))
        for u in G.adj:
            for v in G.adj[u]:
                self.assertIn(u, target_adj)
                self.assertIn(v, target_adj[u], "{} is not adjacent to {}".format(v, u))

        for u in target_adj:
            for v in target_adj[u]:
                self.assertIn(u, G.adj)
                self.assertIn(v, G.adj[u], "{} is not adjacent to {}".format(v, u))

    def test_3d_6x3x4_on_c4(self):
        dims = [6, 3, 4]
        chimera = (4,)

        with self.assertRaises(ValueError):
            find_grid_embedding(dims, *chimera)

    def test_3d_4x4x4_on_c16(self):
        dims = [4, 4, 4]
        chimera = (16,)

        embedding = find_grid_embedding(dims, *chimera)

        self.assertEqual(len(embedding), self.prod(dims))

        target_adj = dwave.embedding.target_to_source(dnx.chimera_graph(*chimera), embedding)

        G = nx.grid_graph(dims)
        for u in G.adj:
            for v in G.adj[u]:
                self.assertIn(u, target_adj)
                self.assertIn(v, target_adj[u], "{} is not adjacent to {}".format(v, u))

        self.assertEqual(set(G.nodes), set(target_adj))

    @staticmethod
    def prod(iterable):
        import operator
        import functools
        return functools.reduce(operator.mul, iterable, 1)
