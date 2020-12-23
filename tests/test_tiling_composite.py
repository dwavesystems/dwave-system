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
import random

import dimod
import dwave_networkx as dnx

from dwave.system.testing import MockDWaveSampler
from dwave.system.composites import TilingComposite


class TestTiling(unittest.TestCase):
    def test_sample_ising(self):
        mock_sampler = MockDWaveSampler()  # C4 structured sampler

        sampler = TilingComposite(mock_sampler, 2, 2)

        h = {node: random.uniform(-1, 1) for node in sampler.structure.nodelist}
        J = {(u, v): random.uniform(-1, 1) for u, v in sampler.structure.edgelist}

        response = sampler.sample_ising(h, J)

    def test_tile_around_hole(self):

        # Create a chimera graph with the following structure:
        # OOOX
        # OOOO
        # OOOO
        # OOOO
        # where O: complete cell, X: incomplete cell
        mock_sampler = MockDWaveSampler(broken_nodes=[8 * 3])  # C4 structured sampler with a node missing
        hardware_graph = dnx.chimera_graph(4)  # C4

        sampler = TilingComposite(mock_sampler, 2, 2, 4)
        # Given the above chimera graph, check that the embeddings are as follows:
        # 00XX
        # 0011
        # 2211
        # 22XX
        # where 0,1,2: belongs to correspoding embedding, X: not used in any embedding
        self.assertSetEqual({v for s in sampler.embeddings[0].values() for v in s},
                            {linear_index for linear_index, (i, j, u, k)
                             in hardware_graph.nodes(data='chimera_index')
                             if i in (0, 1) and j in (0, 1)})
        self.assertSetEqual({v for s in sampler.embeddings[1].values() for v in s},
                            {linear_index for linear_index, (i, j, u, k)
                             in hardware_graph.nodes(data='chimera_index')
                             if i in (1, 2) and j in (2, 3)})
        self.assertSetEqual({v for s in sampler.embeddings[2].values() for v in s},
                            {linear_index for linear_index, (i, j, u, k)
                             in hardware_graph.nodes(data='chimera_index')
                             if i in (2, 3) and j in (0, 1)})

    def test_sample_ising(self):
        sampler = TilingComposite(MockDWaveSampler(), 2, 2)

        h = {node: random.uniform(-1, 1) for node in sampler.structure.nodelist}
        J = {(u, v): random.uniform(-1, 1) for u, v in sampler.structure.edgelist}

        response = sampler.sample_ising(h, J)

        # nothing failed and we got at least one response back per tile
        self.assertGreaterEqual(len(response), len(sampler.embeddings))

        for sample in response.samples():
            for v in h:
                self.assertIn(v, sample)

        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(dimod.ising_energy(sample, h, J), energy)

    def test_sample_qubo(self):
        sampler = TilingComposite(MockDWaveSampler(), 2, 2)

        Q = {(u, v): random.uniform(-1, 1) for u, v in sampler.structure.edgelist}
        Q.update({(node, node): random.uniform(-1, 1) for node in sampler.structure.nodelist})

        response = sampler.sample_qubo(Q)

        # nothing failed and we got at least one response back per tile
        self.assertGreaterEqual(len(response), len(sampler.embeddings))

        for sample in response.samples():
            for u, v in Q:
                self.assertIn(v, sample)
                self.assertIn(u, sample)

        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(dimod.qubo_energy(sample, Q), energy)

    def test_too_many_nodes(self):
        mock_sampler = MockDWaveSampler()  # C4 structured sampler

        sampler = TilingComposite(mock_sampler, 2, 2)

        h = {0: -1, 1: 1}
        J = {}

        response = sampler.sample_ising(h, J)

        __, num_columns = response.record.sample.shape

        self.assertEqual(num_columns, 2)
