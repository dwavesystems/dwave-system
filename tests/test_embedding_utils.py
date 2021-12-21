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
import itertools
import random

from collections.abc import Mapping

import networkx as nx
import numpy as np
import numpy.testing as npt

import dimod
import dwave.embedding


class TestTargetToSource(unittest.TestCase):
    def test_identity_embedding(self):
        """a 1-to-1 embedding should not change the adjacency"""
        target_adj = nx.karate_club_graph()

        embedding = {v: {v} for v in target_adj}

        source_adj = dwave.embedding.target_to_source(target_adj, embedding)

        # test the adjacencies are equal (source_adj is a dict and target_adj is a networkx graph)
        for v in target_adj:
            self.assertIn(v, source_adj)
            for u in target_adj[v]:
                self.assertIn(u, source_adj[v])

        for v in source_adj:
            self.assertIn(v, target_adj)
            for u in source_adj[v]:
                self.assertIn(u, target_adj[v])

    def test_embedding_to_one_node(self):
        """an embedding that maps everything to one node should result in a singleton graph"""
        target_adj = nx.barbell_graph(16, 7)
        embedding = {'a': set(target_adj)}  # all map to 'a'

        source_adj = dwave.embedding.target_to_source(target_adj, embedding)
        self.assertEqual(source_adj, {'a': set()})

        embedding = {'a': {0, 1}}  # not every node is assigned to a chain
        source_adj = dwave.embedding.target_to_source(target_adj, embedding)
        self.assertEqual(source_adj, {'a': set()})

    def test_embedding_overlap(self):
        """overlapping embeddings should raise an error"""
        target_adj = nx.complete_graph(5)
        embedding = {'a': {0, 1}, 'b': {1, 2}}  # overlap

        with self.assertRaises(ValueError):
            source_adj = dwave.embedding.target_to_source(target_adj, embedding)

    def test_square_to_triangle(self):
        target_adjacency = {0: {1, 3}, 1: {0, 2}, 2: {1, 3}, 3: {0, 2}}  # a square graph
        embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}
        source_adjacency = dwave.embedding.target_to_source(target_adjacency, embedding)
        self.assertEqual(source_adjacency, {'a': {'b', 'c'}, 'b': {'a', 'c'}, 'c': {'a', 'b'}})


class TestEdgelistToAdjacency(unittest.TestCase):
    def test_typical(self):
        graph = nx.barbell_graph(17, 8)

        edgelist = set(graph.edges())

        adj = dwave.embedding.utils.edgelist_to_adjacency(edgelist)

        # test that they're equal
        for u, v in edgelist:
            self.assertIn(u, adj)
            self.assertIn(v, adj)
            self.assertIn(u, adj[v])
            self.assertIn(v, adj[u])

        for u in adj:
            for v in adj[u]:
                self.assertTrue((u, v) in edgelist or (v, u) in edgelist)
                self.assertFalse((u, v) in edgelist and (v, u) in edgelist)


class TestAdjacencyToEdgeIter(unittest.TestCase):
    def test_dict(self):
        graph = nx.barbell_graph(17, 8)
        adj = {v: set(graph[v]) for v in graph}

        edgelist = dwave.embedding.utils.adjacency_to_edges(adj)
        new_adj = {}
        for u, v in edgelist:
            new_adj.setdefault(u, set()).add(v)
            new_adj.setdefault(v, set()).add(u)

        self.assertEqual(adj, new_adj)

    def test_nxgraph(self):
        graph = nx.barbell_graph(17, 8)

        edgelist = dwave.embedding.utils.adjacency_to_edges(graph)

        edges0 = sorted(map(sorted, graph.edges()))
        edges1 = sorted(map(sorted, edgelist))

        self.assertEqual(edges0, edges1)

    def test_bqm(self):
        graph = nx.barbell_graph(17, 8)

        bqm = dimod.BQM(vartype = dimod.SPIN)

        bqm.add_interactions_from((u, v, 1) for u, v in graph.edges())

        edgelist = dwave.embedding.utils.adjacency_to_edges(bqm)
        
        edges0 = sorted(map(sorted, graph.edges()))
        edges1 = sorted(map(sorted, edgelist))

        self.assertEqual(edges0, edges1)

    def test_bad(self):
        with self.assertRaises(TypeError):
            edgelist = list(dwave.embedding.utils.adjacency_to_edges([]))

class TestChainToQuadratic(unittest.TestCase):
    def test_K5(self):
        """Test that when given a chain, the returned Jc uses all
        available edges."""
        chain_variables = set(range(5))

        # fully connected
        adjacency = {u: set(chain_variables) for u in chain_variables}
        for v, neighbors in adjacency.items():
            neighbors.remove(v)

        Jc = dwave.embedding.chain_to_quadratic(chain_variables, adjacency, 1.0)

        for u, v in itertools.combinations(chain_variables, 2):
            self.assertFalse((u, v) in Jc and (v, u) in Jc)
            self.assertTrue((u, v) in Jc or (v, u) in Jc)
        for u in chain_variables:
            self.assertFalse((u, u) in Jc)

    def test_5_cycle(self):
        chain_variables = set(range(5))

        # now try a cycle
        adjacency = {v: {(v + 1) % 5, (v - 1) % 5} for v in chain_variables}

        Jc = dwave.embedding.chain_to_quadratic(chain_variables, adjacency, 1.0)

        for u in adjacency:
            for v in adjacency[u]:
                self.assertFalse((u, v) in Jc and (v, u) in Jc)
                self.assertTrue((u, v) in Jc or (v, u) in Jc)

    def test_disconnected(self):
        chain_variables = {0, 2}

        adjacency = {0: {1}, 1: {0, 2}, 2: {1}}

        with self.assertRaises(ValueError):
            dwave.embedding.chain_to_quadratic(chain_variables, adjacency, 1.0)


class TestChainBreakFrequency(unittest.TestCase):
    def test_matrix_all_ones(self):
        """should have no breaks"""

        samples = np.ones((10, 5))

        embedding = {'a': {2, 4}, 'b': {1, 3}}

        freq = dwave.embedding.chain_break_frequency(samples, embedding)

        self.assertEqual(freq, {'a': 0, 'b': 0})

    def test_matrix_all_zeros(self):
        """should have no breaks"""

        samples = np.zeros((10, 5))

        embedding = {'a': {2, 4}, 'b': {1, 3}}

        freq = dwave.embedding.chain_break_frequency(samples, embedding)

        self.assertEqual(freq, {'a': 0, 'b': 0})

    def test_matrix_mix(self):
        samples = np.array([[-1, 1], [1, 1]])

        embedding = {'a': {0, 1}}

        freq = dwave.embedding.chain_break_frequency(samples, embedding)

        self.assertEqual(freq, {'a': .5})

    def test_mix_string_labels(self):
        response = dimod.SampleSet.from_samples([{'a': 1, 'b': 0}, {'a': 0, 'b': 0}],
                                                energy=[1, 0], info={}, vartype=dimod.BINARY)
        embedding = {0: {'a', 'b'}}
        freq = dwave.embedding.chain_break_frequency(response, embedding)

        self.assertEqual(freq, {0: .5})

    def test_with_num_occurences(self):
        samples = [[-1, -1, +1],
                   [-1, +1, +1],
                   [-1, +1, +1],
                   [-1, -1, -1],
                   [-1, +1, +1]]
        labels = 'abc'

        sampleset = dimod.SampleSet.from_samples((samples, labels), energy=0,
                                                 vartype=dimod.SPIN)
        sampleset = sampleset.aggregate()

        embedding = {0: 'ab', 1: 'c'}

        freq = dwave.embedding.chain_break_frequency(sampleset, embedding)

        self.assertEqual(freq, {0: 3./5, 1: 0})

class TestIntLabelDisjointSets(unittest.TestCase):
    def test(self):
        components = map(list, [range(1), range(1, 3), range(3, 6), range(6, 12)])
        djs = dwave.embedding.utils.intlabel_disjointsets(12)

        for x in range(12):
            self.assertEqual(djs.find(x), x)
            self.assertEqual(djs.size(x), 1)

        for c in components:
            for i, j in zip(c, c[1:]):
                djs.union(i, j)

        for c in components:
            root = djs.find(c[0])
            for x in c:
                self.assertEqual(djs.find(x), root)
                self.assertEqual(djs.size(x), len(c))

        for c in components:
            for i, j in combinations(c, 2):
                djs.union(i, j)

        for c in components:
            root = djs.find(c[0])
            for x in c:
                self.assertEqual(djs.find(x), root)
                self.assertEqual(djs.size(x), len(c))

        djs.union(0, 1)
        djs.union(2, 3)
        djs.union(5, 6)
        
        root = djs.find(0)
        for x in range(12):
            self.assertEqual(djs.find(x), root)
            self.assertEqual(djs.size(x), 12)
        


