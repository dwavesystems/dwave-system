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
import os
import unittest
from unittest import mock

import numpy as np
import dimod
import dimod.testing as dit

from dwave.system import DWaveSampler
from dwave.system.testing import MockDWaveSampler, MockLeapHybridDQMSampler
from dwave.cloud.exceptions import ConfigFileError, SolverNotFoundError
from dimod import DiscreteQuadraticModel, ExtendedVartype, SampleSet
from dwave.samplers import SteepestDescentSolver


class TestMockDWaveSampler(unittest.TestCase):
    @unittest.skipIf(os.getenv('SKIP_INT_TESTS'), "Skipping integration test.")
    def test_properties_and_params(self):
        try:
            sampler = DWaveSampler()
        except (ValueError, ConfigFileError, SolverNotFoundError):
            raise unittest.SkipTest("no QPU available")

        mock = MockDWaveSampler()

        self.assertEqual(set(mock.properties), set(sampler.properties))
        self.assertEqual(set(mock.parameters), set(sampler.parameters))

        #Check extraction of nodelist, edgelist and properties from 
        mock = MockDWaveSampler.from_qpu_sampler(sampler)
        self.assertEqual(mock.nodelist.sort(),sampler.nodelist.sort())
        self.assertEqual(mock.edgelist.sort(),sampler.edgelist.sort())

    def test_sampler(self):
        sampler = MockDWaveSampler()
        dit.assert_sampler_api(sampler)
        dit.assert_structured_api(sampler)

    def test_mock_parameters(self):
        # Single cell Chimera (DW2000Q) solver:
        sampler = MockDWaveSampler(topology_type='chimera', topology_shape = [1,1,4])
        num_reads = 10
        ss = sampler.sample_ising({0: -1}, {}, num_reads=num_reads)
        self.assertEqual(sum(ss.record.num_occurrences),num_reads)
        self.assertTrue(len(ss.record.num_occurrences) <= 2) #Binary state histogram
        ss = sampler.sample_ising({0: -1}, {}, num_reads=num_reads,
                                  answer_mode='raw')
        self.assertEqual(len(ss.record.num_occurrences), num_reads)
        
        ss = sampler.sample_ising({0: -1}, {}, num_reads=num_reads,
                                  answer_mode='raw', max_answers=2)
        self.assertEqual(len(ss.record.num_occurrences), 2)

        # disable exact ground state calc
        with mock.patch.object(sampler, 'exact_solver_cutoff', 0):
            #QPU format initial states:
            initial_state = [(i,1) if i%4==0 else (i,3) for i in range(8)]
            ss = sampler.sample_ising({0: 1, 4: 1}, {(0, 4): -2},
                                      num_reads=1,
                                      answer_mode='raw',
                                      initial_state=initial_state)
            #The initialized state is a local minima, and should
            #be returned under greedy descent mocking:
            self.assertEqual(ss.record.energy[0],0)

    def test_unmock_parameters(self):
        # Check valid DWaveSampler() parameter, that is not mocked throws a warning:
        sampler = MockDWaveSampler()
        with self.assertWarns(UserWarning) as w:
            ss = sampler.sample_ising({0 : -1}, {}, annealing_time=123)
        with self.assertRaises(ValueError) as e:
            ss = sampler.sample_ising({0 : -1}, {}, monkey_time=123)
        with self.assertRaises(ValueError) as e:
            sampler = MockDWaveSampler(topology_type="chimera123")
        sampler = MockDWaveSampler(parameter_warnings=False)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter(action="error", category=UserWarning)
            ss = sampler.sample_ising({0 : -1}, {}, annealing_time=123)

    def test_ground_override(self):
        # bqm with a local minimum at (-1, -1)
        bqm = dimod.BQM.from_ising({'a': -1, 'b': -1}, {'ab': -2})
        local_minimum = [-1, -1]    # energy: 0
        ground_state = [1, 1]       # energy: -4
        local_minimum_state = list(zip(bqm.variables, local_minimum))

        # simple two connected variables sampler (path(2))
        sampler = MockDWaveSampler(nodelist=['a', 'b'], edgelist=[('a', 'b')],
                                   exact_solver_cutoff=len(bqm.variables))

        # disable exact ground state calc and start from a local minimum -> greedy should stay stuck
        with mock.patch.object(sampler, 'exact_solver_cutoff', 0):
            ss = sampler.sample(bqm, initial_state=local_minimum_state)
            np.testing.assert_array_equal(ss.record[0].sample, local_minimum)

        # boundary on which exact ground state calc kicks in
        with mock.patch.object(sampler, 'exact_solver_cutoff', len(bqm.variables)):
            ss = sampler.sample(bqm, initial_state=local_minimum_state)
            np.testing.assert_array_equal(ss.record[0].sample, ground_state)

        # double-check the default
        self.assertEqual(sampler.exact_solver_cutoff, len(bqm.variables))
        ss = sampler.sample(bqm)
        np.testing.assert_array_equal(ss.record[0].sample, ground_state)

    def test_empty_bqm(self):
        sampler = MockDWaveSampler()
        bqm = dimod.BQM('SPIN')
        ss = sampler.sample(bqm)
        self.assertIs(ss.vartype, bqm.vartype)

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

    def test_properties(self):
        properties = {'topology' : {
            'type' : 'pegasus',
            'shape' : [5]}}
        # Note, topology_type and topology_shape must be consistent
        # or None.
        with self.assertRaises(ValueError) as e:
            sampler = MockDWaveSampler(topology_type='chimera',
                                       properties=properties)
        with self.assertRaises(ValueError) as e:
            sampler = MockDWaveSampler(topology_shape=[4],
                                       properties=properties)
        sampler = MockDWaveSampler(properties=properties)
        self.assertEqual(sampler.properties['topology']['type'],'pegasus')
        self.assertEqual(sampler.properties['topology']['shape'][-1],5)
        properties['category'] = 'test_choice'
        sampler = MockDWaveSampler(properties=properties)
        self.assertEqual(sampler.properties['category'],'test_choice')
        
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

        nodelist = [0,4,5,6,7]
        edgelist = [(5,7),(4,6)]
        sampler = MockDWaveSampler(topology_type='chimera',
                                   topology_shape=chimera_shape,
                                   nodelist=nodelist,
                                   edgelist=edgelist)
        self.assertTrue(len(sampler.nodelist)==5)
        self.assertTrue(len(sampler.edgelist)==2)
        
        sampler = MockDWaveSampler(topology_type='chimera',
                                   topology_shape=chimera_shape,
                                   nodelist=nodelist,
                                   edgelist=edgelist,
                                   broken_nodes=delete_nodes,
                                   broken_edges=delete_edges)
        self.assertTrue(len(sampler.nodelist)==4)
        self.assertTrue(len(sampler.edgelist)==1)

    def test_custom_substitute_sampler(self):
        """Test that MockDWaveSampler uses the provided custom substitute_sampler."""

        # Define a sampler that always returns the a constant (excited) state
        class SteepestAscentSolver(SteepestDescentSolver):
            def sample(self, bqm, **kwargs):
                # Return local (or global)  maxima instead of local minima
                # NOTE: energy returned is not faithful to the original bqm (energy calculated as `-bqm`)
                return super().sample(-bqm, **kwargs)

        inverted_sampler = SteepestAscentSolver()

        # Create a simple BQM
        bqm = dimod.BQM({'a': 1, 'b': 1}, {}, 0.0, vartype="SPIN")

        # Instantiate MockDWaveSampler with nodelist and edgelist including 'a' and 'b'
        sampler = MockDWaveSampler(
            substitute_sampler=inverted_sampler,
            nodelist=['a', 'b'],
            edgelist=[('a', 'b')]
        )

        # First Subtest: First sample does not use ExactSampler();
        # Second sample does not use SteepestDescentSampler()
        with self.subTest("Sampler without ExactSampler"):
            ss = sampler.sample(bqm, num_reads=2)
            self.assertEqual(sampler.exact_solver_cutoff, 0)
            self.assertEqual(ss.record.sample.shape, (1,2), 'Unique sample expected')
            self.assertTrue(np.all(ss.record.sample==1), 'Excited states expected')

        sampler = MockDWaveSampler(
            substitute_sampler=inverted_sampler,
            nodelist=['a', 'b'],
            edgelist=[('a', 'b')],
            exact_solver_cutoff=2
        )
        # Second Subtest: First sample uses ExactSampler();
        # Second sampler uses inverted sampler. Explicit exact_solver_cutoff overrides substitute_sampler.
        with self.subTest("Sampler with ExactSampler and substitute_sampler"):
            ss = sampler.sample(bqm, num_reads=2, answer_mode='raw')
            self.assertEqual(sampler.exact_solver_cutoff, 2)
            self.assertEqual(ss.record.sample.shape, (2,2), 'Non-unique samples expected')
            self.assertTrue(np.all(ss.record.sample[0,:] == -1), 'Excited states expected')
            self.assertTrue(np.all(ss.record.sample[1,:] == 1), 'Excited states expected')

    def test_mocking_sampler_params(self):
        """Test that substitute_kwargs are correctly passed to the substitute_sampler."""

        # Define a constant sampler that checks for a custom parameter
        class ConstantSampler(dimod.Sampler):
            properties = {}
            parameters = {'custom_param': [], 'num_reads': []}

            def sample(self, bqm, **kwargs):
                custom_param = kwargs.get('custom_param')
                num_reads = kwargs.get('num_reads')
                # Raise exception if parameters passed incorrectly
                if custom_param != 'test_value':
                    raise ValueError("custom_param not passed correctly")
                if num_reads != 10:
                    raise ValueError(f"num_reads not passed correctly, expected 10, got {num_reads}")
                # Return a default sample
                sample = {v: -1 for v in bqm.variables}
                return dimod.SampleSet.from_samples_bqm(sample, bqm)

        constant_sampler = ConstantSampler()

        # Create a simple BQM
        bqm = dimod.BQM({'a': 1, 'b': 1}, {('a', 'b'): 1}, 0.0, vartype="SPIN")

        # Instantiate MockDWaveSampler with nodelist and edgelist including 'a' and 'b'
        sampler = MockDWaveSampler(
            substitute_sampler=constant_sampler,
            substitute_kwargs={'custom_param': 'test_value'},
            nodelist=['a', 'b'],
            edgelist=[('a', 'b')]
        )

        # Sample using the MockDWaveSampler
        ss = sampler.sample(bqm, num_reads=10)

        # Check that the sample returned is as expected from the custom sampler
        expected_sample = {'a': -1, 'b': -1}
        self.assertEqual(ss.first.sample, expected_sample)
        self.assertEqual(ss.first.energy, bqm.energy(expected_sample))


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
