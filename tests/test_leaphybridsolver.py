# Copyright 2020 D-Wave Systems Inc.
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
#
# =============================================================================
import unittest
from concurrent.futures import Future
import numpy as np

import dimod
from tabu import TabuSampler
from dwave.system.samplers import LeapHybridSampler

from dwave.cloud.computation import Future as cloud_future

try:
    # py3
    import unittest.mock as mock
except ImportError:
    # py2
    import mock

class MockSolver():

    properties = {'supported_problem_types': ['bqm'],
                  'minimum_time_limit': [[1, 1.0], [1024, 1.0],
                                         [4096, 10.0], [10000, 40.0]],
                  'parameters': {'time_limit': None},
                  'category': 'hybrid',
                  'quota_conversion_rate': 1}

    def upload_bqm(self, bqm, **parameters):
        future = Future()
        future.set_result(bqm)
        return future

    def sample_bqm(self, sapi_problem_id, time_limit):
        result = TabuSampler().sample(sapi_problem_id, timeout=1000*int(time_limit))
        future = cloud_future('fake_solver', None)
        future._result = {'sampleset': result, 'problem_type': 'bqm'}
        return future

class TestLeapHybridSampler(unittest.TestCase):

    @mock.patch('dwave.system.samplers.leap_hybrid_sampler.Client.get_solver')
    def test_solver_init(self, mock_get_solver):

        mock_get_solver.return_value = MockSolver()

        mock_get_solver.reset_mock()
        LeapHybridSampler(solver="any name")
        mock_get_solver.assert_called_once()

        solver = {'category': 'not hybrid'}
        with self.assertRaises(ValueError):
            LeapHybridSampler(solver=solver)

    @mock.patch('dwave.system.samplers.leap_hybrid_sampler.Client.get_solver')
    def test_sample_bqm(self, mock_get_solver):

        mock_get_solver.return_value = MockSolver()

        bqm = dimod.BinaryQuadraticModel({'a': -1, 'b': 1, 'c': 1},
                    {'ab': -0.8, 'ac': -0.7, 'bc': -1}, 0, dimod.SPIN)

        sampler = LeapHybridSampler(solver = {'category': 'hybrid'})

        response = sampler.sample(bqm)

        rows, cols = response.record.sample.shape

        self.assertEqual(cols, 3)
        self.assertFalse(np.any(response.record.sample == 0))
        self.assertIs(response.vartype, dimod.SPIN)
        self.assertIn('num_occurrences', response.record.dtype.fields)

    @mock.patch('dwave.system.samplers.leap_hybrid_sampler.Client.get_solver')
    def test_sample_ising_variables(self, mock_get_solver):

        mock_get_solver.return_value = MockSolver()

        sampler = LeapHybridSampler(solver = {'category': 'hybrid'})

        response = sampler.sample_ising({0: -1, 1: 1}, {})

        rows, cols = response.record.sample.shape

        self.assertEqual(cols, 2)

        response = sampler.sample_ising({}, {(0, 4): 1})

        rows, cols = response.record.sample.shape
        self.assertEqual(cols, 2)
        self.assertFalse(np.any(response.record.sample == 0))
        self.assertIs(response.vartype, dimod.SPIN)

    @mock.patch('dwave.system.samplers.leap_hybrid_sampler.Client.get_solver')
    def test_sample_qubo_variables(self, mock_get_solver):

        mock_get_solver.return_value = MockSolver()

        sampler = LeapHybridSampler(solver = {'QPU': False})

        response = sampler.sample_qubo({(0, 0): -1, (1, 1): 1})

        rows, cols = response.record.sample.shape

        self.assertEqual(cols, 2)
    
        response = sampler.sample_qubo({(0, 0): -1, (1, 1): 1})

        rows, cols = response.record.sample.shape

        self.assertEqual(cols, 2)
        self.assertTrue(np.all(response.record.sample >= 0))
        self.assertIs(response.vartype, dimod.BINARY)
