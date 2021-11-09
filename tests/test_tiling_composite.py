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
    def test_pegasus_single_cell(self):
        #Test trivial case of single cell (K4,4+4*odd) embedding over defect free
        mock_sampler = MockDWaveSampler(topology_type='pegasus')  # P3 structured sampler
        self.assertTrue('topology' in mock_sampler.properties and 'type' in mock_sampler.properties['topology'])
        self.assertTrue(mock_sampler.properties['topology']['type'] == 'pegasus' and 'shape' in mock_sampler.properties['topology'])
        sampler = TilingComposite(mock_sampler, 1, 1)
        h = {node: random.uniform(-1, 1) for node in sampler.structure.nodelist}
        J = {(u, v): random.uniform(-1, 1) for u, v in sampler.structure.edgelist}

        m = n = mock_sampler.properties['topology']['shape'][0] - 1
        expected_number_of_cells = m*n*3
        num_reads = 10
        response = sampler.sample_ising(h, J, num_reads=num_reads)
        self.assertTrue(sum(response.record.num_occurrences)==expected_number_of_cells*num_reads)
        
    def test_pegasus_multi_cell(self):
        #Test case of 2x3 cell embedding over defect free
        mock_sampler = MockDWaveSampler(topology_type='pegasus',topology_shape=[8])  # P8 structured sampler
        self.assertTrue('topology' in mock_sampler.properties and 'type' in mock_sampler.properties['topology'])
        self.assertTrue(mock_sampler.properties['topology']['type'] == 'pegasus' and 'shape' in mock_sampler.properties['topology'])
        sampler = TilingComposite(mock_sampler, 1, 1)
        h = {node: random.uniform(-1, 1) for node in sampler.structure.nodelist}
        J = {(u, v): random.uniform(-1, 1) for u, v in sampler.structure.edgelist}
        
        m_sub = 2
        n_sub = 3
        sampler = TilingComposite(mock_sampler, m_sub, n_sub)
        h = {node: random.uniform(-1, 1) for node in sampler.structure.nodelist}
        J = {(u, v): random.uniform(-1, 1) for u, v in sampler.structure.edgelist}

        m = n = mock_sampler.properties['topology']['shape'][0] - 1
        expected_number_of_cells = (m//m_sub)*(n//3)*3
        num_reads = 1
        response = sampler.sample_ising(h, J, num_reads = num_reads)
        self.assertTrue(sum(response.record.num_occurrences)==expected_number_of_cells*num_reads)
        
        
    def test_sample_ising(self):
        mock_sampler = MockDWaveSampler()  # C4 structured sampler

        sampler = TilingComposite(mock_sampler, 2, 2)

        h = {node: random.uniform(-1, 1) for node in sampler.structure.nodelist}
        J = {(u, v): random.uniform(-1, 1) for u, v in sampler.structure.edgelist}

        response = sampler.sample_ising(h, J)

    def test_tile_around_hole(self):

        # Create a Chimera C4 structured sampler with a node missing, so that
        # we have a defect pattern:
        # OOOX
        # OOOO
        # OOOO
        # OOOO
        # where O: complete cell, X: incomplete cell
        mock_sampler = MockDWaveSampler(broken_nodes=[8 * 3]) 
        hardware_graph = dnx.chimera_graph(4)  # C4

        # Tile with 2x2 cells:
        sampler = TilingComposite(mock_sampler, 2, 2, 4)
        # Given the above chimera graph, check that the embeddings are as
        # follows:
        # 00XX
        # 0011
        # 2211
        # 22XX
        # where 0,1,2: belongs to correspoding embedding, X: not used in any
        # embedding
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
        
    def test_tile_around_node_defects_pegasus(self):
        pegasus_shape = [5]
        # Create a pegasus P5 structured solver subject to node defects with the
        # following (nice-coordinate) 3x4x4 cell-level structure:
        # OOOX OOOO OOOO
        # OOOO OOOO OOOO
        # OOOO OOOO OOOO
        # OOOO OOOO OOOX
        # where O: complete cell, X: incomplete cell
        broken_node_nice_coordinates = [(0,0,3,0,1), (2,3,3,1,3)]
        broken_node_linear_coordinates = [
            dnx.pegasus_coordinates(pegasus_shape[0]).nice_to_linear(coord)
            for coord in broken_node_nice_coordinates]
        mock_sampler = MockDWaveSampler(topology_type='pegasus',
                                        topology_shape=pegasus_shape,
                                        broken_nodes=broken_node_linear_coordinates) 
        # Tile with 2x2 cells:
        sampler = TilingComposite(mock_sampler, 2, 2, 4)
        
        # Given the above pegasus graph, check that the embeddings are as
        # follows:
        # 00XX  3344 7788 
        # 0011  3344 7788 
        # 2211  5566 99XX  
        # 22XX  5566 99XX
        
        # Check correct number of embeddings and size of each is sufficient,
        # given chimera test checks detailed position:
        self.assertTrue(len(sampler.embeddings) == 10)
        self.assertFalse(any([len(emb) != 32 for emb in sampler.embeddings]))

        #Can be refined to check exact positioning, but a lot of ugly code:
        #For visualization in coordinate scheme use:
        #for emb in sampler.embeddings:
        #    print({key: dnx.pegasus_coordinates(pegasus_shape[0]).linear_to_nice(next(iter(val)))
        #           for key,val in emb.items()})

    def test_tile_around_edge_defects_pegasus(self):
        pegasus_shape = [5]

        # P5 structured sampler with one missing external edge that does not p
        # prevent tesselation of 2x2 blocks (12 tiles, equivalent to full yield)
        broken_edges_nice_coordinates = [(0,1,0,0,0), (0,2,0,0,0)]
        broken_edges = [tuple(
            dnx.pegasus_coordinates(pegasus_shape[0]).nice_to_linear(coord)
            for coord in broken_edges_nice_coordinates)]
        mock_sampler = MockDWaveSampler(topology_type='pegasus',
                                        topology_shape=pegasus_shape,
                                        broken_edges=broken_edges)
        sampler = TilingComposite(mock_sampler, 2, 2, 4)
        self.assertTrue(len(sampler.embeddings) == 12)
        
        # P5 structured sampler with one missing internal edge that prevents
        # tesselation of 2x2 blocks (otherwise 12 tiles, with edge defect 11)
        broken_edge_nice_coordinates = [(0,0,0,0,0), (0,0,0,1,0)]
        broken_edges = [tuple(
            dnx.pegasus_coordinates(pegasus_shape[0]).nice_to_linear(coord)
            for coord in broken_edge_nice_coordinates)]
        mock_sampler = MockDWaveSampler(topology_type='pegasus',
                                        topology_shape=pegasus_shape,
                                        broken_edges=broken_edges)
        sampler = TilingComposite(mock_sampler, 2, 2, 4)
        self.assertTrue(len(sampler.embeddings) == 11)
        
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
