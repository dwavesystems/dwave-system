import unittest

import dimod
import dwave_micro_client_dimod as micro

import dwave_virtual_graph as vg
from tests.mock_sampler import MockSampler

####################################################################################################
# Test with the system if available
####################################################################################################

try:
    micro.DWaveSampler(url='flux_bias_test', permissive_ssl=True)
    _sampler_connection = True
except Exception as e:
    # no sapi credentials are stored on the path or credentials are out of date
    _sampler_connection = False
_sampler_connection = False


@unittest.skipUnless(_sampler_connection, "No sampler to connect to")
class TestVirtualGraphWithSystem(unittest.TestCase):
    def test_smoke(self):
        child_sampler = micro.DWaveSampler(url='flux_bias_test', permissive_ssl=True)

        # NB: this should be removed later
        child_sampler.solver.parameters['x_flux_bias'] = ''

        sampler = vg.VirtualGraph(child_sampler, {'a': [0]})

        # the structure should be very simple
        self.assertEqual(sampler.structure, (['a'], [], {'a': set()}))

        response = sampler.sample_ising({'a': -.5}, {})


####################################################################################################
# Test with mock sampler
####################################################################################################

class TestVirtualGraphWithMockSampler(unittest.TestCase):
    def setUp(self):
        self.sampler = MockSampler()

    def test_smoke(self):
        child_sampler = MockSampler()
        sampler = vg.VirtualGraph(child_sampler, {'a': [0]}, flux_bias_test_reads=10)

        # depending on how recenlty flux bias data was gathered, this may be true
        child_sampler.flux_biases_flag = False

        if sampler.flux_biases:
            sampler.sample_ising({'a': -1}, {})
            self.assertTrue(child_sampler.flux_biases_flag)  # true when some have been provided to sample_ising

    def test_structure_keyword_setting(self):
        sampler = vg.VirtualGraph(self.sampler, embedding={'a': set(range(8)),
                                                           'b': set(range(8, 16)),
                                                           'c': set(range(16, 24))},
                                  flux_biases=False)

        nodelist, edgelist, adj = sampler.structure
        self.assertEqual(nodelist, ['a', 'b', 'c'])
        self.assertEqual(edgelist, [('a', 'b'), ('b', 'c')])
        self.assertEqual(adj, {'a': {'b'}, 'b': {'a', 'c'}, 'c': {'b'}})

        # unlike variable names
        sampler = vg.VirtualGraph(self.sampler, embedding={'a': set(range(8)),
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
        embedding = {v: {v} for v in adj}

        sampler = vg.VirtualGraph(sampler, embedding=embedding, flux_biases=False)

        self.assertEqual(sampler.embedding, embedding)

    def test_simple_complete_graph_sample_ising(self):
        """sample_ising on a K4."""

        K4 = vg.VirtualGraph(self.sampler, embedding={0: {0, 4},
                                                      1: {1, 5},
                                                      2: {2, 6},
                                                      3: {3, 7}},
                             flux_biases=False)

        K4.sample_ising({0: .1, 1: .2}, {(0, 1): 1.5})
