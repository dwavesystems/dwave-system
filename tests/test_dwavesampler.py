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
#
# =============================================================================
import unittest
import random
import warnings

from collections import namedtuple
from concurrent.futures import Future

import numpy as np

import dimod
import dwave_networkx as dnx

from dwave.system.samplers import DWaveSampler

try:
    # py3
    import unittest.mock as mock
except ImportError:
    # py2
    import mock


C16 = dnx.chimera_graph(16)

# remove one node from C16 to simulate a not-fully-yielded system
C16.remove_node(42)

edges = set(tuple(edge) for edge in C16.edges)
edges.update([(v, u) for u, v in edges])  # solver has bi-directional


class MockSolver():
    nodes = set(range(2048))
    edges = edges
    properties = {'readout_thermalization_range': [0, 10000],
                  'annealing_time_range': [1, 2000],
                  'default_readout_thermalization': 0,
                  'parameters': {'num_spin_reversal_transforms': '',
                                 'programming_thermalization': '',
                                 'anneal_offsets': '',
                                 'num_reads': '',
                                 'max_answers': '',
                                 'readout_thermalization': '',
                                 'beta': "",
                                 'answer_mode': '',
                                 'auto_scale': '',
                                 'postprocess': "",
                                 'anneal_schedule': '',
                                 'chains': ""},
                  'chip_id': 'MockSolver'}

    def sample_ising(self, h, J, **kwargs):
        for key in kwargs:
            if key not in self.properties['parameters']:
                raise ValueError
        result = {'num_variables': 2048,
                  'format': 'qp',
                  'num_occurrences': [1],
                  'active_variables': list(range(2048)),
                  'solutions': [[random.choice((-1, +1)) for __ in range(2048)]],
                  'timing': {'total_real_time': 11511, 'anneal_time_per_run': 20,
                             'post_processing_overhead_time': 2042, 'qpu_sampling_time': 164,
                             'readout_time_per_run': 123,
                             'qpu_delay_time_per_sample': 21,
                             'qpu_anneal_time_per_sample': 20,
                             'total_post_processing_time': 2042,
                             'qpu_programming_time': 8740,
                             'run_time_chip': 164,
                             'qpu_access_time': 11511,
                             'qpu_readout_time_per_sample': 123},
                  'occurrences': [1]}
        result['samples'] = result['solutions']
        result['energies'] = [dimod.ising_energy(sample, h, J) for sample in result['samples']]
        future = Future()
        future.set_result(result)
        return future

    def sample_qubo(self, Q, **kwargs):
        for key in kwargs:
            if key not in self.properties['parameters']:
                raise ValueError
        result = {'num_variables': 2048,
                  'format': 'qp',
                  'num_occurrences': [1],
                  'active_variables': list(range(2048)),
                  'solutions': [[random.choice((0, 1)) for __ in range(2048)]],
                  'timing': {'total_real_time': 11511, 'anneal_time_per_run': 20,
                             'post_processing_overhead_time': 2042, 'qpu_sampling_time': 164,
                             'readout_time_per_run': 123,
                             'qpu_delay_time_per_sample': 21,
                             'qpu_anneal_time_per_sample': 20,
                             'total_post_processing_time': 2042,
                             'qpu_programming_time': 8740,
                             'run_time_chip': 164,
                             'qpu_access_time': 11511,
                             'qpu_readout_time_per_sample': 123},
                  'occurrences': [1]}
        result['samples'] = result['solutions']
        result['energies'] = [dimod.qubo_energy(sample, Q) for sample in result['samples']]
        future = Future()
        future.set_result(result)
        return future


class TestDwaveSampler(unittest.TestCase):
    @mock.patch('dwave.system.samplers.dwave_sampler.Client')
    def setUp(self, MockClient):

        # using the mock
        self.sampler = DWaveSampler()

        self.sampler.solver = MockSolver()

    @mock.patch('dwave.system.samplers.dwave_sampler.Client')
    def test_solver_init(self, MockClient):
        """Deprecation warning is raised for `solver_features` use, but it still works."""

        # assertWarns not available in py2
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DWaveSampler(solver_features={'qpu': True})
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DWaveSampler(solver={'qpu': True})
            self.assertEqual(len(w), 0)

        MockClient.reset_mock()
        solver = {'qpu': True, 'num_qubits__gt': 1000}
        sampler = DWaveSampler(solver=solver)
        MockClient.from_config.assert_called_once_with(solver=solver)

    def test_sample_ising_variables(self):

        sampler = self.sampler

        response = sampler.sample_ising({0: -1, 1: 1}, {})

        rows, cols = response.record.sample.shape

        self.assertEqual(cols, 2)

        response = sampler.sample_ising({}, {(0, 4): 1})

        rows, cols = response.record.sample.shape

        self.assertEqual(cols, 2)

        self.assertFalse(np.any(response.record.sample == 0))
        self.assertIs(response.vartype, dimod.SPIN)

        self.assertIn('num_occurrences', response.record.dtype.fields)
        self.assertIn('timing', response.info)

    def test_sample_qubo_variables(self):

        sampler = self.sampler

        response = sampler.sample_qubo({(0, 0): -1, (1, 1): 1})

        rows, cols = response.record.sample.shape

        self.assertEqual(cols, 2)

        response = sampler.sample_qubo({(0, 0): -1, (1, 1): 1})

        rows, cols = response.record.sample.shape

        self.assertEqual(cols, 2)

        self.assertTrue(np.all(response.record.sample >= 0))
        self.assertIs(response.vartype, dimod.BINARY)

        self.assertIn('num_occurrences', response.record.dtype.fields)
        self.assertIn('timing', response.info)


class TestDWaveSamplerAnnealSchedule(unittest.TestCase):
    def test_typical(self):
        class MockScheduleSampler(DWaveSampler):
            parameters = {'anneal_schedule': ''}
            properties = {'max_anneal_schedule_points': 4,
                          'annealing_time_range': [1, 2000]}

            def __init__(self):
                pass

        DWaveSampler.validate_anneal_schedule(MockScheduleSampler(), [(0, 1), (55.0, 0.45), (155.0, 0.45), (210.0, 1)])
