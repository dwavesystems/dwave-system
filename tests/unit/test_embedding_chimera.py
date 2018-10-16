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

import dwave_networkx as dnx
import dimod

from dwave.embedding.chimera import find_clique_embedding, find_biclique_embedding


class Test_find_clique_embedding(unittest.TestCase):
    def test_full_yield_one_tile_k3(self):
        emb = find_clique_embedding(3, 1)

        target = dnx.chimera_graph(1)

        source = dimod.embedding.target_to_source(target, emb)

        self.assertEqual(source, {0: {1, 2}, 1: {0, 2}, 2: {0, 1}})

    def test_full_yield_one_tile_k2(self):
        emb = find_clique_embedding(2, 1)

        target = dnx.chimera_graph(1)

        source = dimod.embedding.target_to_source(target, emb)

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
