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

import sys
import unittest
import random
import warnings

from collections import namedtuple
from concurrent.futures import Future
from unittest import mock
from uuid import uuid4

import numpy as np

import dimod
import dwave_networkx as dnx

from dwave.cloud.exceptions import SolverOfflineError, SolverNotFoundError

from dwave.system.samplers import DWaveSampler
from dwave.system.warnings import EnergyScaleWarning, TooFewSamplesWarning


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

    def sample_bqm(self, bqm, num_reads=1, **kwargs):
        problem_id = str(uuid4())
        problem_label = kwargs.pop('label', None)
        info = dict(timing={'total_real_time': 11511, 'anneal_time_per_run': 20,
                            'post_processing_overhead_time': 2042,
                            'qpu_sampling_time': 164,
                            'readout_time_per_run': 123,
                            'qpu_delay_time_per_sample': 21,
                            'qpu_anneal_time_per_sample': 20,
                            'total_post_processing_time': 2042,
                            'qpu_programming_time': 8740,
                            'run_time_chip': 164,
                            'qpu_access_time': 11511,
                            'qpu_readout_time_per_sample': 123},
                    problem_id=problem_id)
        if problem_label:
            info.update(problem_label=problem_label)

        samples = np.random.choice(tuple(bqm.vartype.value),
                                   size=(num_reads, len(bqm)))

        future = Future()
        ss = dimod.SampleSet.from_samples_bqm((samples, bqm.variables), bqm,
                                              info=info)
        future.sampleset = ss
        future.id = problem_id
        future.label = problem_label

        return future


class TestDwaveSampler(unittest.TestCase):
    @mock.patch('dwave.system.samplers.dwave_sampler.Client')
    def setUp(self, MockClient):

        # using the mock
        self.sampler = DWaveSampler()

        self.sampler.solver = MockSolver()

    @mock.patch('dwave.system.samplers.dwave_sampler.Client')
    def test_init_default(self, MockClient):
        """QPU with the highest number of qubits chosen by default."""

        sampler = DWaveSampler()

        MockClient.from_config.assert_called_once_with(
            client='qpu',
            defaults={'solver': {'order_by': '-num_active_qubits'}})

    @mock.patch('dwave.system.samplers.dwave_sampler.Client')
    def test_init_generic_behavior(self, MockClient):
        """Generic solver behavior (default prior to 0.10.0) can be forced."""

        sampler = DWaveSampler(client='base')

        MockClient.from_config.assert_called_once_with(
            client='base',
            defaults={'solver': {'order_by': '-num_active_qubits'}})

    @mock.patch('dwave.system.samplers.dwave_sampler.Client')
    def test_init_solver(self, MockClient):
        """QPU can be explicitly selected (old default usage example)"""

        solver = {'qpu': True, 'num_qubits__gt': 1000}

        sampler = DWaveSampler(solver=solver)

        MockClient.from_config.assert_called_once_with(
            client='qpu',
            solver=solver,
            defaults={'solver': {'order_by': '-num_active_qubits'}})

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
        self.assertIn('problem_id', response.info)

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
        self.assertIn('problem_id', response.info)

    def test_problem_labelling(self):
        sampler = self.sampler

        # label parameter is supported
        self.assertIn('label', sampler.parameters)

        # no-label case works as before
        ss = sampler.sample_ising({}, {(0, 4): 1})

        self.assertIn('problem_id', ss.info)
        self.assertNotIn('problem_label', ss.info)

        # label is propagated to sampleset.info
        label = 'problem label'
        ss = sampler.sample_ising({}, {(0, 4): 1}, label=label)

        self.assertIn('problem_id', ss.info)
        self.assertIn('problem_label', ss.info)
        self.assertEqual(ss.info.get('problem_label'), label)

    @mock.patch('dwave.system.samplers.dwave_sampler.Client')
    def test_failover_false(self, MockClient):
        sampler = DWaveSampler(failover=False)

        sampler.solver.sample_ising.side_effect = SolverOfflineError
        sampler.solver.sample_qubo.side_effect = SolverOfflineError
        sampler.solver.sample_bqm.side_effect = SolverOfflineError

        with self.assertRaises(SolverOfflineError):
            sampler.sample_ising({}, {})

    @mock.patch('dwave.system.samplers.dwave_sampler.Client')
    def test_failover_offline(self, MockClient):
        if sys.version_info.major <= 2 or sys.version_info.minor < 6:
            raise unittest.SkipTest("need mock features only available in 3.6+")

        sampler = DWaveSampler(failover=True)

        mocksolver = sampler.solver
        edgelist = sampler.edgelist

        # call once
        ss = sampler.sample_ising({}, {})

        self.assertIs(mocksolver, sampler.solver)  # still same solver

        # one of the sample methods was called
        self.assertEqual(sampler.solver.sample_ising.call_count
                         + sampler.solver.sample_qubo.call_count
                         + sampler.solver.sample_bqm.call_count, 1)

        # add a side-effect
        sampler.solver.sample_ising.side_effect = SolverOfflineError
        sampler.solver.sample_qubo.side_effect = SolverOfflineError
        sampler.solver.sample_bqm.side_effect = SolverOfflineError

        # and make sure get_solver makes a new mock solver
        sampler.client.get_solver.reset_mock(return_value=True)

        ss = sampler.sample_ising({}, {})

        self.assertIsNot(mocksolver, sampler.solver)  # new solver
        self.assertIsNot(edgelist, sampler.edgelist)  # also should be new

    @mock.patch('dwave.system.samplers.dwave_sampler.Client')
    def test_failover_notfound_noretry(self, MockClient):

        sampler = DWaveSampler(failover=True, retry_interval=-1)

        mocksolver = sampler.solver

        # add a side-effect
        sampler.solver.sample_ising.side_effect = SolverOfflineError
        sampler.solver.sample_qubo.side_effect = SolverOfflineError
        sampler.solver.sample_bqm.side_effect = SolverOfflineError

        # and make sure get_solver makes a new mock solver
        sampler.client.get_solver.side_effect = SolverNotFoundError

        with self.assertRaises(SolverNotFoundError):
            sampler.sample_ising({}, {})

    def test_warnings_energy_range(self):
        sampler = self.sampler

        response = sampler.sample_ising({0: 10**-4, 1: 1}, {},
                                        warnings='SAVE')

        self.assertIn('warnings', response.info)
        count = 0
        for warning in response.info['warnings']:
            if issubclass(warning['type'], EnergyScaleWarning):
                count += 1
        self.assertEqual(count, 1)

        response = sampler.sample_qubo({(0, 4): -1}, warnings='SAVE')

        count = 0
        for warning in response.info['warnings']:
            if issubclass(warning['type'], EnergyScaleWarning):
                count += 1
        self.assertEqual(count, 1)

    def test_warnings_sampling_error(self):
        sampler = self.sampler

        response = sampler.sample_ising({0: 10**-4, 1: 1}, {},
                                        warnings='SAVE')

        self.assertIn('warnings', response.info)
        count = 0
        for warning in response.info['warnings']:
            if issubclass(warning['type'], TooFewSamplesWarning):
                count += 1
        self.assertEqual(count, 1)
        self.assertEqual(len(response.info['warnings']), 2)

        response = sampler.sample_qubo({(0, 4): -1}, warnings='SAVE')

        count = 0
        for warning in response.info['warnings']:
            if issubclass(warning['type'], TooFewSamplesWarning):
                count += 1
        self.assertEqual(count, 1)
        self.assertEqual(len(response.info['warnings']), 2)

    def test_to_networkx_chimera(self):
        sampler = self.sampler
        sampler.solver.properties.update({'topology': {'type': 'chimera', 'shape': [4, 4, 4]}})
        G = sampler.to_networkx_graph()

        # Create chimera graph for comparison
        chimeraG = dnx.chimera_graph(4, node_list=sampler.nodelist, edge_list=sampler.edgelist)

        self.assertEqual(set(G), set(chimeraG))

        for u, v in chimeraG.edges:
            self.assertIn(u, G[v])
            
        del sampler.solver.properties['topology']

    def test_to_networkx_pegasus(self):
        sampler = self.sampler
        sampler.solver.properties.update({'topology': {'type': 'pegasus', 'shape': [4, 4, 12]}})
        G = sampler.to_networkx_graph()

        # Create pegasus graph for comparison
        pegasusG = dnx.pegasus_graph(4, node_list=sampler.nodelist, edge_list=sampler.edgelist)

        self.assertEqual(set(G), set(pegasusG))

        for u, v in pegasusG.edges:
            self.assertIn(u, G[v])

        del sampler.solver.properties['topology']

class TestDWaveSamplerAnnealSchedule(unittest.TestCase):
    def test_typical(self):
        class MockScheduleSampler(DWaveSampler):
            parameters = {'anneal_schedule': ''}
            properties = {'max_anneal_schedule_points': 4,
                          'annealing_time_range': [1, 2000]}

            def __init__(self):
                pass

        DWaveSampler.validate_anneal_schedule(MockScheduleSampler(), [[0.0, 0.0], [0.2, 0.2], [5.2, 0.2], [6.0, 1.0]])
        DWaveSampler.validate_anneal_schedule(MockScheduleSampler(), [(0, 1), (55.0, 0.45), (155.0, 0.45), (210.0, 1)])
