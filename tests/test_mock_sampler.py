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

from dwave.cloud.exceptions import ConfigFileError, SolverNotFoundError
from dwave.system import DWaveSampler
import os

@unittest.skipIf(os.getenv('SKIP_INT_TESTS'), "Skipping integration test.")

        
class TestMockDWaveSampler(unittest.TestCase):
    def test_properties_and_params(self):
        try:
            sampler = DWaveSampler()
        except (ValueError, ConfigFileError, SolverNotFoundError):
            raise unittest.SkipTest("no QPU available")

        mock = MockDWaveSampler()

        self.assertEqual(set(mock.properties), set(sampler.properties))
        self.assertEqual(set(mock.parameters), set(sampler.parameters))
        
    def test_sampler(self):
        sampler = MockDWaveSampler()
        dit.assert_sampler_api(sampler)
        dit.assert_structured_api(sampler)
        
    def test_mock_parameters(self):
        # Single cell Chimera (DW2000Q) solver:
        sampler = MockDWaveSampler(topology_type='chimera', topology_shape = [1,1,4])
        num_reads = 10
        ss = sampler.sample_ising({0 : -1}, {}, num_reads=num_reads)
        self.assertEqual(sum(ss.record.num_occurrences),num_reads)
        self.assertTrue(len(ss.record.num_occurrences)<=2) #Binary state histogram
        ss = sampler.sample_ising({0 : -1}, {}, num_reads=num_reads,
                                  answer_mode='raw')
        self.assertEqual(len(ss.record.num_occurrences),num_reads)
        
        ss = sampler.sample_ising({0 : -1}, {}, num_reads=num_reads,
                                  answer_mode='raw', max_answers=2)
        self.assertEqual(len(ss.record.num_occurrences),2)

            
        try:
            from greedy import SteepestDescentSampler as SubstituteSampler
            mock_fallback_substitute = False
        except:
            mock_fallback_substitute = True
        if not mock_fallback_substitute:
            # If the greedy is available, we can mock more parameters (than
            # is possible with the fallback dimod.SimulatedAnnealingSampler()
            # method

            #QPU format inital states:
            initial_states = [(0,1),(0,1)] + [(i,3) for i in range(2,8)]
            ss = sampler.sample_ising({0 : 1, 1 : 1}, {(0,1) : -2},
                                      num_reads=num_reads,
                                      answer_mode='raw', max_samples=2)
            #The initialized state is a local minima, and should
            #be returned under greedy descent mocking:
            self.assertEqual(ss.record.energy[0],-1)
        
    def test_chimera_topology(self):
        grid_parameter = 5
        tile_parameter = 2
        sampler = MockDWaveSampler(topology_type='chimera',
                                   topology_shape=[grid_parameter,
                                                   grid_parameter,
                                                   tile_parameter])
        # C5 (shore 2) fabric only has 200 nodes
        self.assertTrue(len(sampler.nodelist)==
                        grid_parameter*grid_parameter*tile_parameter*2)
        dit.assert_sampler_api(sampler)
        dit.assert_structured_api(sampler)

    def test_pegasus_topology(self):
        grid_parameter = 4
        sampler = MockDWaveSampler(topology_type='pegasus',
                                   topology_shape=[grid_parameter])
        # P4 fabric only has 264 nodes
        self.assertTrue(len(sampler.nodelist)==264)
        dit.assert_sampler_api(sampler)
        dit.assert_structured_api(sampler)

    def test_zephyr_topology(self):
        grid_parameter = 3
        tile_parameter = 4
        sampler = MockDWaveSampler(topology_type='zephyr',
                                   topology_shape=[grid_parameter,
                                                   tile_parameter])
        # P4 fabric only has 264 nodes
        self.assertTrue(len(sampler.nodelist)==
                        tile_parameter*grid_parameter*(8*grid_parameter + 4))
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
