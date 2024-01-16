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

import unittest

import dwave_networkx as dnx

from dwave.cloud.testing import isolated_environ

from dwave.system import common_working_graph
from dwave.system.utilities import FeatureFlags


class TestCommonWorkingGraph(unittest.TestCase):
    def test_single_tile(self):

        G1 = dnx.chimera_graph(1)
        with self.assertWarns(DeprecationWarning):
            G = common_working_graph(G1, G1)

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

        with self.assertWarns(DeprecationWarning):
            G = common_working_graph(G1, G1)

        self.assertEqual(len(G), 8)

    def test_missing_node(self):
        G1 = dnx.chimera_graph(1)
        G1.remove_node(2)
        G2 = dnx.chimera_graph(2)

        with self.assertWarns(DeprecationWarning):
            G = common_working_graph(G1, G1)

        self.assertNotIn(2, G)
        self.assertNotIn((2, 4), G.edges())

    def test_sampler_adjacency(self):
        adj = {0: {1, 2}, 1: {2}, 2: {0, 1}}
        G = dnx.chimera_graph(1)

        with self.assertWarns(DeprecationWarning):
            H = common_working_graph(adj, G)

        self.assertEqual(set(H.nodes), {0, 1, 2})
        self.assertEqual(set(H.edges), set())


class TestFeatureFlagSupport(unittest.TestCase):
    def test_base(self):
        with isolated_environ(add={'DWAVE_FEATURE_FLAGS': '{"x": true}'}):
            self.assertTrue(FeatureFlags.get('x'))
        with isolated_environ(add={'DWAVE_FEATURE_FLAGS': '{"x": false}'}):
            self.assertFalse(FeatureFlags.get('x'))
        with isolated_environ(add={'DWAVE_FEATURE_FLAGS': '{"x": 0}'}):
            self.assertFalse(FeatureFlags.get('x'))
        with isolated_environ(add={'DWAVE_FEATURE_FLAGS': '{"x": 1}'}):
            self.assertTrue(FeatureFlags.get('x'))
        with isolated_environ(add={'DWAVE_FEATURE_FLAGS': '{"x": error}'}):
            self.assertFalse(FeatureFlags.get('x'))
        with isolated_environ(add={'DWAVE_FEATURE_FLAGS': '{"y": true}'}):
            self.assertFalse(FeatureFlags.get('x'))
        with isolated_environ(add={'DWAVE_FEATURE_FLAGS': ''}):
            self.assertFalse(FeatureFlags.get('x'))
        with isolated_environ(remove=('DWAVE_FEATURE_FLAGS',)):
            self.assertFalse(FeatureFlags.get('x'))

    def test_hss_solver_config_override(self):
        with isolated_environ(add={'DWAVE_FEATURE_FLAGS': '{"hss_solver_config_override": true}'}):
            self.assertTrue(FeatureFlags.hss_solver_config_override)
        with isolated_environ(remove=('DWAVE_FEATURE_FLAGS',)):
            self.assertFalse(FeatureFlags.hss_solver_config_override)
