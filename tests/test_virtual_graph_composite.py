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
import collections

import dimod

from dwave.system.composites import VirtualGraphComposite

from dwave.system.testing import MockDWaveSampler


class TestVirtualGraphWithMockDWaveSampler(unittest.TestCase):
    def setUp(self):
        self.sampler = MockDWaveSampler()

    def test_smoke(self):
        child_sampler = MockDWaveSampler()
        with self.assertWarns(DeprecationWarning):
            sampler = VirtualGraphComposite(child_sampler, {'a': [0]}, flux_bias_num_reads=1)

        # depending on how recenlty flux bias data was gathered, this may be true
        child_sampler.flux_biases_flag = False

        if sampler.flux_biases:
            sampler.sample_ising({'a': -1}, {})
            self.assertTrue(child_sampler.flux_biases_flag)  # true when some have been provided to sample_ising

    def test_structure_keyword_setting(self):
        with self.assertWarns(DeprecationWarning):
            sampler = VirtualGraphComposite(self.sampler, embedding={'a': set(range(8)),
                                                                     'b': set(range(8, 16)),
                                                                     'c': set(range(16, 24))},
                                            flux_biases=False)

        nodelist, edgelist, adj = sampler.structure
        self.assertEqual(nodelist, ['a', 'b', 'c'])
        self.assertEqual(edgelist, [('a', 'b'), ('b', 'c')])
        self.assertEqual(adj, {'a': {'b'}, 'b': {'a', 'c'}, 'c': {'b'}})

        # unlike variable names
        with self.assertWarns(DeprecationWarning):
            sampler = VirtualGraphComposite(self.sampler, embedding={'a': set(range(8)),
                                                                     1: set(range(8, 16)),
                                                                     'c': set(range(16, 24))},
                                            flux_biases=False)
        nodelist, edgelist, adj = sampler.structure
        self.assertEqual(set(nodelist), {'a', 1, 'c'})
        self.assertEqual(adj, {'a': {1}, 1: {'a', 'c'}, 'c': {1}})
        self.assertIsInstance(edgelist, list)
        self.assertIsInstance(nodelist, list)

        # edges should still be unique
        for u in adj:
            for v in adj[u]:
                if (u, v) in edgelist:
                    assert (v, u) not in edgelist
                if (v, u) in edgelist:
                    assert (u, v) not in edgelist
                assert (u, v) in edgelist or (v, u) in edgelist

    def test_embedding_parameter(self):
        """Given embedding should be saved as a parameter"""
        sampler = self.sampler

        __, __, adj = sampler.structure
        embedding = {v: (v,) for v in adj}

        with self.assertWarns(DeprecationWarning):
            sampler = VirtualGraphComposite(sampler, embedding=embedding, flux_biases=False)

        self.assertEqual(sampler.embedding, embedding)

    def test_simple_complete_graph_sample_ising(self):
        """sample_ising on a K4."""

        with self.assertWarns(DeprecationWarning):
            K4 = VirtualGraphComposite(self.sampler, embedding={0: {0, 4},
                                                                1: {1, 5},
                                                                2: {2, 6},
                                                                3: {3, 7}},
                                       flux_biases=False)

        K4.sample_ising({0: .1, 1: .2}, {(0, 1): 1.5})


class Test_ValidateChainStrength(unittest.TestCase):

    def test_no_properties(self):
        from dwave.system.composites.virtual_graph import _validate_chain_strength

        Sampler = collections.namedtuple('Sampler', ['properties'])
        sampler = Sampler({})

        with self.assertRaises(ValueError):
            _validate_chain_strength(sampler, None)

        with self.assertRaises(ValueError):
            _validate_chain_strength(sampler, 1.0)

    def test_j_range(self):
        from dwave.system.composites.virtual_graph import _validate_chain_strength

        Sampler = collections.namedtuple('Sampler', ['properties'])
        sampler = Sampler({'j_range': [-1.0, 1.0]})

        self.assertEqual(_validate_chain_strength(sampler, None), 1.0)
        self.assertEqual(_validate_chain_strength(sampler, .5), .5)

        with self.assertRaises(ValueError):
            _validate_chain_strength(sampler, 1.5)

    def test_extended_j_range(self):
        from dwave.system.composites.virtual_graph import _validate_chain_strength

        Sampler = collections.namedtuple('Sampler', ['properties'])
        sampler = Sampler({'j_range': [-1.0, 1.0], 'extended_j_range': [-2.0, 2.0]})

        self.assertEqual(_validate_chain_strength(sampler, None), 2.0)
        self.assertEqual(_validate_chain_strength(sampler, .5), .5)
        self.assertEqual(_validate_chain_strength(sampler, 1.5), 1.5)
