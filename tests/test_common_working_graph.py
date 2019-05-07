# Copyright 2019 D-Wave Systems Inc.
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
# =============================================================================
from __future__ import division

import unittest

import networkx as nx
import dwave_networkx as dnx

class TestCommonGraph(unittest.TestCase):
    def test_single_tile(self):

        G1 = dnx.chimera_graph(1)
        G = dnx.common_graph(G1, G1)

        # should have 8 nodes
        self.assertEqual(len(G), 8)

        # nodes 0,...,7 should be in the tile
        for n in range(8):
            self.assertIn(n, G)

        # check bipartite
        for i in range(4):
            for j in range(4, 8):
                self.assertTrue((i, j) in G.edges() or (j, i) in G.edges())

    def test_c1_c2_tiles(self):
        G1 = dnx.chimera_graph(1)
        G2 = dnx.chimera_graph(2)

        G = dnx.common_graph(G1, G1)

        self.assertEqual(len(G), 8)

    def test_missing_node(self):
        G1 = dnx.chimera_graph(1)
        G1.remove_node(2)
        G2 = dnx.chimera_graph(2)

        G = dnx.common_graph(G1, G1)

        self.assertNotIn(2, G)
        self.assertNotIn((2, 4), G.edges())
