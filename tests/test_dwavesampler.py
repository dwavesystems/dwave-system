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
from uuid import uuid4
from unittest import mock
from concurrent.futures import Future

import numpy as np
from networkx.utils import graphs_equal
from parameterized import parameterized

import dimod
import dwave_networkx as dnx

from dwave.cloud import exceptions
from dwave.cloud import computation
from dwave.cloud.testing import mocks
from dwave.cloud.solver import StructuredSolver
from dwave.cloud.client.base import Client

from dwave.system.samplers import DWaveSampler, qpu_graph
from dwave.system.warnings import EnergyScaleWarning, TooFewSamplesWarning
from dwave.system.exceptions import FailoverCondition, RetryCondition


C16 = dnx.chimera_graph(16)

# remove one node from C16 to simulate a not-fully-yielded system
C16.remove_node(42)

edges = set(tuple(edge) for edge in C16.edges)
edges.update([(v, u) for u, v in edges])  # solver has bi-directional


class MockSolver():
    nodes = set(range(2048))
    edges = edges
    properties = {'readout_thermalization_range': [0.0, 10000.0],
                  'annealing_time_range': [1.0, 2000.0],
                  'default_readout_thermalization': 0.0,
                  'parameters': {'num_spin_reversal_transforms': '',
                                 'programming_thermalization': '',
                                 'anneal_offsets': '',
                                 'num_reads': '',
                                 'max_answers': '',
                                 'readout_thermalization': '',
                                 'beta': '',
                                 'answer_mode': '',
                                 'auto_scale': '',
                                 'postprocess': '',
                                 'anneal_schedule': '',
                                 'chains': ''},
                  'chip_id': 'MockSolver'}

    def sample_bqm(self, bqm, num_reads=1, **kwargs):
        problem_id = str(uuid4())
        problem_label = kwargs.pop('label', None)
        info = dict(timing={'total_real_time': 11511.0, 'anneal_time_per_run': 20.0,
                            'post_processing_overhead_time': 2042.0,
                            'qpu_sampling_time': 164.0,
                            'readout_time_per_run': 123.0,
                            'qpu_delay_time_per_sample': 21.0,
                            'qpu_anneal_time_per_sample': 20.0,
                            'total_post_processing_time': 2042.0,
                            'qpu_programming_time': 8740.0,
                            'run_time_chip': 164.0,
                            'qpu_access_time': 11511.0,
                            'qpu_readout_time_per_sample': 123.0},
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


class TestDWaveSampler(unittest.TestCase):
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
    def test_failover_off(self, MockClient):
        sampler = DWaveSampler(failover=False)

        sampler.solver.sample_bqm.side_effect = exceptions.SolverOfflineError

        with self.assertRaises(exceptions.SolverOfflineError):
            sampler.sample_ising({}, {})

    @parameterized.expand([
        (exceptions.InvalidAPIResponseError, FailoverCondition),
        (exceptions.SolverNotFoundError, FailoverCondition),
        (exceptions.SolverOfflineError, FailoverCondition),
        (exceptions.SolverError, FailoverCondition),
        (exceptions.PollingTimeout, RetryCondition),
        (exceptions.SolverAuthenticationError, exceptions.SolverAuthenticationError),   # auth error propagated
        (KeyError, KeyError),   # unrelated errors propagated
    ])
    @mock.patch('dwave.system.samplers.dwave_sampler.Client')
    def test_async_failover(self, source_exc, target_exc, MockClient):
        sampler = DWaveSampler(failover=True)

        mocksolver = sampler.solver
        edgelist = sampler.edgelist

        # call once (async, no need to resolve)
        sampler.sample_ising({}, {})

        self.assertIs(mocksolver, sampler.solver)  # still same solver

        # one of the sample methods was called
        self.assertEqual(sampler.solver.sample_ising.call_count
                         + sampler.solver.sample_qubo.call_count
                         + sampler.solver.sample_bqm.call_count, 1)

        # simulate solver exception on sampleset resolve
        fut = computation.Future(mocksolver, None)
        fut._set_exception(source_exc)
        sampler.solver.sample_bqm = mock.Mock()
        sampler.solver.sample_bqm.return_value = fut

        # verify failover signalled
        with self.assertRaises(target_exc):
            sampler.sample_ising({}, {}).resolve()

        # make sure get_solver makes a new mock solver
        sampler.client.get_solver.reset_mock(return_value=True)

        # trigger failover
        sampler.trigger_failover()

        # verify failover
        self.assertIsNot(mocksolver, sampler.solver)  # new solver
        self.assertIsNot(edgelist, sampler.edgelist)  # also should be new

    @mock.patch('dwave.system.samplers.dwave_sampler.Client')
    def test_failover_penalization(self, MockClient):
        # create a working client instance that returns a few mock solvers
        client = Client(endpoint='endpoint', token='token')
        client._fetch_solvers = lambda **kw: [
            StructuredSolver(data=mocks.qpu_pegasus_solver_data(4), client=None),
            StructuredSolver(data=mocks.qpu_chimera_solver_data(4), client=None),
        ]

        # make sure sampler instance uses our client instance
        MockClient.from_config.return_value = client

        # verify we get a different solver after failover
        sampler = DWaveSampler(failover=True)
        initial_solver = sampler.solver

        sampler.trigger_failover()
        self.assertNotEqual(sampler.solver, initial_solver)

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
        G2 = qpu_graph(sampler.properties['topology']['type'],
                       sampler.properties['topology']['shape'],
                       sampler.nodelist,
                       sampler.edgelist)
        self.assertTrue(graphs_equal(G,G2))
        # Create chimera graph for comparison
        chimeraG = dnx.chimera_graph(4, node_list=sampler.nodelist, edge_list=sampler.edgelist)

        self.assertEqual(set(G), set(chimeraG))

        for u, v in chimeraG.edges:
            self.assertIn(u, G[v])
            
        del sampler.solver.properties['topology']

    def test_to_networkx_pegasus(self):
        sampler = self.sampler
        sampler.solver.properties.update({'topology': {'type': 'pegasus', 'shape': [4]}})
        G = sampler.to_networkx_graph()

        # Create pegasus graph for comparison
        pegasusG = dnx.pegasus_graph(4, node_list=sampler.nodelist, edge_list=sampler.edgelist)

        self.assertEqual(set(G), set(pegasusG))

        for u, v in pegasusG.edges:
            self.assertIn(u, G[v])

        del sampler.solver.properties['topology']

    def test_to_networkx_zephyr(self):
        sampler = self.sampler
        sampler.solver.properties.update({'topology': {'type': 'zephyr', 'shape': [4, 4]}})
        G = sampler.to_networkx_graph()

        # Create zephyr graph for comparison
        zephyrG = dnx.zephyr_graph(4, node_list=sampler.nodelist, edge_list=sampler.edgelist)

        self.assertEqual(set(G), set(zephyrG))

        for u, v in zephyrG.edges:
            self.assertIn(u, G[v])

        del sampler.solver.properties['topology']


class TestDWaveSamplerAnnealSchedule(unittest.TestCase):
    def test_typical(self):
        class MockScheduleSampler(DWaveSampler):
            parameters = {'anneal_schedule': ''}
            properties = {'max_anneal_schedule_points': 4,
                          'annealing_time_range': [1.0, 2000.0]}

            def __init__(self):
                pass

        DWaveSampler.validate_anneal_schedule(MockScheduleSampler(), [[0.0, 0.0], [0.2, 0.2], [5.2, 0.2], [6.0, 1.0]])
        DWaveSampler.validate_anneal_schedule(MockScheduleSampler(), [(0, 1), (55.0, 0.45), (155.0, 0.45), (210.0, 1)])
