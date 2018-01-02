import unittest

import dimod
import dwave_networkx as dnx

import dwave_virtual_graph as vg
# from dwave_virtual_graph.tests.mock_sampler import MockSampler


# class TestComposite(unittest.TestCase):
#     def setUp(self):
#         self.sampler = MockSampler()

#     def test_instantiation(self):
#         """Create a composed sampler user VirtualGraph. Make sure everything
#         has been updated appropriately.
#         """
#         sampler = vg.VirtualGraph(self.sampler, embedding={}, embedding_tag='')

#         # todo: test without keywargs provided

#     def test_structure_keyword_setting(self):
#         sampler = vg.VirtualGraph(self.sampler, embedding={'a': set(range(8)),
#                                                            'b': set(range(8, 16)),
#                                                            'c': set(range(16, 24))})

#         nodelist, edgelist, adj = sampler.structure
#         self.assertEqual(nodelist, ['a', 'b', 'c'])
#         self.assertEqual(edgelist, [('a', 'b'), ('b', 'c')])
#         self.assertEqual(adj, {'a': {'b'}, 'b': {'a', 'c'}, 'c': {'b'}})

#         # unlike variable names
#         sampler = vg.VirtualGraph(self.sampler, embedding={'a': set(range(8)),
#                                                            1: set(range(8, 16)),
#                                                            'c': set(range(16, 24))})
#         nodelist, edgelist, adj = sampler.structure
#         self.assertEqual(set(nodelist), {'a', 1, 'c'})
#         self.assertEqual(adj, {'a': {1}, 1: {'a', 'c'}, 'c': {1}})
#         self.assertIsInstance(edgelist, list)
#         self.assertIsInstance(nodelist, list)

#         # edges should still be unique
#         for u in adj:
#             for v in adj[u]:
#                 if (u, v) in edgelist:
#                     assert (v, u) not in edgelist
#                 if (v, u) in edgelist:
#                     assert (u, v) not in edgelist
#                 assert (u, v) in edgelist or (v, u) in edgelist

#     def test_embedding_parameter(self):
#         """Given embedding should be saved as a parameter"""
#         sampler = self.sampler

#         __, __, adj = sampler.structure
#         embedding = {v: {v} for v in adj}

#         sampler = vg.VirtualGraph(sampler, embedding=embedding)

#         self.assertEqual(sampler.embedding, embedding)

#     def test_simple_complete_graph_sample_ising(self):
#         """sample_ising on a K4."""

#         K4 = vg.VirtualGraph(self.sampler, embedding={0: {0, 4},
#                                                       1: {1, 5},
#                                                       2: {2, 6},
#                                                       3: {3, 7}})

#         K4.sample_ising({0: .1, 1: .2}, {(0, 1): 1.5})
