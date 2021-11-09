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

"""who watches the watchmen?"""
import unittest

from dwave.system.testing import MockDWaveSampler, MockLeapHybridDQMSampler

import dimod.testing as dit
from dimod import DiscreteQuadraticModel, ExtendedVartype, SampleSet


class TestMockDWaveSampler(unittest.TestCase):
    def test_sampler(self):
        sampler = MockDWaveSampler()
        dit.assert_sampler_api(sampler)
        dit.assert_structured_api(sampler)
        
    def test_topology_arguments(self):
        pegasus_size = 4
        sampler = MockDWaveSampler(topology_type='pegasus',topology_shape=[pegasus_size])
        # P4 fabric only has 264 nodes
        self.assertTrue(len(sampler.nodelist)==264)
        dit.assert_sampler_api(sampler)
        dit.assert_structured_api(sampler)
        
    def test_yield_arguments(self):
        # Request 1 node and 1 edge deletion, check resulting graph 
        #    1      3
        #  X      2
        #
        #    5 -X-  7
        #  4      6
        # Defect free case: 8 nodes. 4 external edges, 4 internal edges
        # Delete first node (2 edges, 1 node)
        # Delete final external edge (1 edge)
    
        delete_nodes = [0]
        delete_edges = [(5, 7)]
        
        chimera_shape = [2, 2, 1]
        sampler = MockDWaveSampler(topology_type='chimera',
                                   topology_shape=chimera_shape,
                                   broken_nodes=delete_nodes,
                                   broken_edges=delete_edges)
        self.assertTrue(len(sampler.nodelist)==7)
        self.assertTrue(len(sampler.edgelist)==5)
        
class TestMockLeapHybridDQMSampler(unittest.TestCase):
    def test_sampler(self):
        sampler = MockLeapHybridDQMSampler()

        self.assertTrue(callable(sampler.sample_dqm))
        self.assertTrue(callable(sampler.min_time_limit))
        self.assertTrue(hasattr(sampler, 'properties'))
        self.assertTrue(hasattr(sampler, 'parameters'))

        dqm = DiscreteQuadraticModel()
        dqm.add_variable(3)
        dqm.add_variable(4)

        result = sampler.sample_dqm(dqm)
        self.assertTrue(isinstance(result, SampleSet))
        self.assertGreaterEqual(len(result), 12)  # min num of samples from dqm solver
        self.assertEqual(len(result.variables), 2)
        self.assertEqual(result.vartype, ExtendedVartype.DISCRETE)
