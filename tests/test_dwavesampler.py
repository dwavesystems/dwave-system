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

import contextlib
import textwrap
import unittest
from uuid import uuid4
from unittest import mock
from concurrent.futures import Future

import numpy as np
from parameterized import parameterized

import dimod
import dwave.graphs

from dwave.cloud import exceptions
from dwave.cloud import computation
from dwave.cloud.testing import mocks, isolated_environ, iterable_mock_open
from dwave.cloud.solver import StructuredSolver
from dwave.cloud.api.models import Region
from dwave.cloud.config.constants import DEFAULT_REGION

from dwave.system.samplers import DWaveSampler, qpu_graph
from dwave.system.samplers.dwave_sampler import Client
from dwave.system.warnings import EnergyScaleWarning, TooFewSamplesWarning
from dwave.system.exceptions import FailoverCondition, RetryCondition


C16 = dwave.graphs.chimera_graph(16)

# remove one node from C16 to simulate a not-fully-yielded system
C16.remove_node(42)

edges = set(tuple(edge) for edge in C16.edges)
edges.update([(v, u) for u, v in edges])  # solver has bi-directional


class MockSolver:
    nodes = set(range(2048))
    edges = edges
    properties = {'readout_thermalization_range': [0.0, 10000.0],
                  'annealing_time_range': [1.0, 2000.0],
                  'default_readout_thermalization': 0.0,
                  'parameters': {'programming_thermalization': '',
                                 'anneal_offsets': '',
                                 'num_reads': '',
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


def mock_client_class():
    # client class mock that ensures instances returned via from_config have a
    # get_solvers() that returns mock solvers bound to the client instance

    MockClient = mock.MagicMock()
    client = MockClient.from_config.return_value
    client.get_solvers.side_effect = \
        lambda *args, **kwargs: [mock.MagicMock(client=client)]
    return MockClient


def patch_client():
    return mock.patch('dwave.system.samplers.dwave_sampler.Client',
                      new_callable=mock_client_class)


def mock_fetch_solvers(*solvers):
    # _fetch_solvers impl that returns solvers with bound clients
    def _fetch_solvers(client, *args, **kwargs):
        return [type(solver)(client=client, data=solver.data) for solver in solvers]
    return _fetch_solvers


class TestDWaveSampler(unittest.TestCase):
    @patch_client()
    def setUp(self, MockClient):

        # using the mock
        self.sampler = DWaveSampler()

        self.sampler.solver = MockSolver()

    @patch_client()
    def test_init_default(self, MockClient):
        """QPU with the highest number of qubits chosen by default."""

        sampler = DWaveSampler()

        MockClient.from_config.assert_called_once_with(
            client='qpu',
            defaults={'solver': {'order_by': '-num_active_qubits'}})

    @patch_client()
    def test_init_generic_behavior(self, MockClient):
        """Generic solver behavior (default prior to 0.10.0) can be forced."""

        sampler = DWaveSampler(client='base')

        MockClient.from_config.assert_called_once_with(
            client='base',
            defaults={'solver': {'order_by': '-num_active_qubits'}})

    @patch_client()
    def test_init_solver(self, MockClient):
        """QPU can be explicitly selected (old default usage example)"""

        solver = {'qpu': True, 'num_qubits__gt': 1000}

        sampler = DWaveSampler(solver=solver)

        MockClient.from_config.assert_called_once_with(
            client='qpu',
            solver=solver,
            defaults={'solver': {'order_by': '-num_active_qubits'}})

    @patch_client()
    def test_close(self, MockClient):
        """DWaveSampler.close() will close the underlying client."""

        sampler = DWaveSampler()
        sampler.close()

        MockClient.from_config.return_value.close.assert_called_once()

    @patch_client()
    def test_context_manager(self, MockClient):
        """DWaveSampler() can be used as a context manager and the underlying
        client is closed on exit.
        """

        with DWaveSampler() as sampler:
            ...

        MockClient.from_config.return_value.close.assert_called_once()

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

        self.assertEqual(set(G.nodes), set(G2.nodes))
        self.assertEqual(set(frozenset(e) for e in G.edges), set(frozenset(e) for e in G.edges))

        # Create chimera graph for comparison
        chimeraG = dwave.graphs.chimera_graph(4, node_list=sampler.nodelist, edge_list=sampler.edgelist)

        self.assertEqual(set(G), set(chimeraG))

        for u, v in chimeraG.edges:
            self.assertIn(u, G[v])
            
        del sampler.solver.properties['topology']

    def test_to_networkx_pegasus(self):
        sampler = self.sampler
        sampler.solver.properties.update({'topology': {'type': 'pegasus', 'shape': [4]}})
        G = sampler.to_networkx_graph()

        # Create pegasus graph for comparison
        pegasusG = dwave.graphs.pegasus_graph(4, node_list=sampler.nodelist, edge_list=sampler.edgelist)

        self.assertEqual(set(G), set(pegasusG))

        for u, v in pegasusG.edges:
            self.assertIn(u, G[v])

        del sampler.solver.properties['topology']

    def test_to_networkx_zephyr(self):
        sampler = self.sampler
        sampler.solver.properties.update({'topology': {'type': 'zephyr', 'shape': [4, 4]}})
        G = sampler.to_networkx_graph()

        # Create zephyr graph for comparison
        zephyrG = dwave.graphs.zephyr_graph(4, node_list=sampler.nodelist, edge_list=sampler.edgelist)

        self.assertEqual(set(G), set(zephyrG))

        for u, v in zephyrG.edges:
            self.assertIn(u, G[v])

        del sampler.solver.properties['topology']


class TestDWaveSamplerFailover(unittest.TestCase):
    @patch_client()
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
    @patch_client()
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
        sampler.client.get_solvers.reset_mock(return_value=True)

        # trigger failover
        sampler.trigger_failover()

        # verify failover
        self.assertIsNot(mocksolver, sampler.solver)  # new solver
        self.assertIsNot(edgelist, sampler.edgelist)  # also should be new

    @mock.patch.object(Client, 'create_session', lambda client: mock.Mock())
    def test_failover_penalization(self):

        # a few mock solvers with varying number of qubits
        p4 = StructuredSolver(data=mocks.qpu_pegasus_solver_data(4), client=None)
        p6 = StructuredSolver(data=mocks.qpu_pegasus_solver_data(6), client=None)
        c4 = StructuredSolver(data=mocks.qpu_chimera_solver_data(4), client=None)

        with mock.patch.object(Client, '_fetch_solvers', mock_fetch_solvers(p4, p6, c4)):

            # verify we get solvers in order that respects user (in this case default)
            # sort order -- by number of qubits descending
            sampler = DWaveSampler(failover=True, token='mock')

            # expected solver order (after each failover) is: p6, p4, c4
            self.assertEqual(sampler.solver.identity, p6.identity)

            sampler.trigger_failover()
            self.assertEqual(sampler.solver.identity, p4.identity)

            sampler.trigger_failover()
            self.assertEqual(sampler.solver.identity, c4.identity)

            # verify wrap-around in case all solvers fail same number of times
            sampler.trigger_failover()
            self.assertEqual(sampler.solver.identity, p6.identity)

    @mock.patch.object(Client, 'create_session', lambda client: mock.Mock())
    def test_failover_config_invariance(self):
        # verify failover preserves user-specified solver order

        # a few mock solvers with varying number of qubits and topology
        p1 = StructuredSolver(data=mocks.qpu_pegasus_solver_data(1), client=None)
        c1 = StructuredSolver(data=mocks.qpu_chimera_solver_data(1), client=None)
        p4 = StructuredSolver(data=mocks.qpu_pegasus_solver_data(4), client=None)
        c4 = StructuredSolver(data=mocks.qpu_chimera_solver_data(4), client=None)

        with mock.patch.object(Client, '_fetch_solvers', mock_fetch_solvers(p4, c4, p1, c1)):

            # request chimeras only, preferring smaller ones
            solver_def = dict(topology__type='chimera', order_by='num_active_qubits')
            sampler = DWaveSampler(failover=True, token='mock', solver=solver_def)

            self.assertEqual(sampler.solver.identity, c1.identity)

            sampler.trigger_failover()
            self.assertEqual(sampler.solver.identity, c4.identity)

            # verify wrap-around
            sampler.trigger_failover()
            self.assertEqual(sampler.solver.identity, c1.identity)

    @mock.patch.object(Client, 'create_session', lambda client: mock.Mock())
    def test_solver_not_found(self):
        # verify DWaveSampler still fails in case no solvers are available

        # only pegasus is available
        p1 = StructuredSolver(data=mocks.qpu_pegasus_solver_data(1), client=None)

        with mock.patch.object(Client, '_fetch_solvers', mock_fetch_solvers(p1)):

            # but we request a chimera
            solver_def = dict(topology__type='chimera')
            with self.assertRaises(exceptions.SolverNotFoundError):
                DWaveSampler(failover=True, token='mock', solver=solver_def)


class TestDWaveSamplerMultiRegionFailover(unittest.TestCase):
    @mock.patch('dwave.system.samplers.dwave_sampler.get_regions')
    @patch_client()
    def test_failover_regions_init(self, mock_client, mock_get_regions):
        with self.subTest("single default region"):
            with DWaveSampler(failover_regions=False) as sampler:
                self.assertEqual(len(sampler._clients), 1)

        with self.subTest("single explicit region"):
            mock_client.reset_mock()
            region = 'mock-region-1'
            with DWaveSampler(failover_regions=[region]) as sampler:
                self.assertEqual(len(sampler._clients), 1)
                self.assertEqual(mock_client.from_config.call_args.kwargs.get('region'), region)

        with self.subTest("explicit list of regions"):
            mock_client.reset_mock()
            regions = ['mock-region-1', 'mock-region-2']
            default_region = regions[0]
            mock_client.from_config.return_value.__enter__.return_value.config.region = default_region
            with DWaveSampler(failover_regions=regions) as sampler:
                self.assertEqual(len(sampler._clients), len(regions))
                calls = mock_client.from_config.call_args_list
                calls.pop(0)    # first client is used to fetch regions properly
                self.assertEqual(len(calls), len(regions))
                for i, region in enumerate(regions):
                    self.assertEqual(calls[i].kwargs.get('region'), region)

        with self.subTest("explicit list of regions with explicit default region"):
            mock_client.reset_mock()
            regions = ['mock-region-1', 'mock-region-2']
            default_region = regions[-1]
            mock_client.from_config.return_value.__enter__.return_value.config.region = default_region
            with DWaveSampler(failover_regions=regions, region=default_region) as sampler:
                self.assertEqual(len(sampler._clients), len(regions))
                calls = mock_client.from_config.call_args_list
                calls.pop(0)    # first client is used to fetch regions properly
                self.assertEqual(len(calls), len(regions))
                # just verify the default client matches the default region
                self.assertEqual(sampler.failover_regions[0], default_region)
                self.assertEqual(calls[0].kwargs.get('region'), default_region)

        with self.subTest("all available regions"):
            mock_client.reset_mock()
            regions = ['mock-region-1', 'mock-region-2']
            default_region = regions[0]
            mock_get_regions.return_value = [
                Region(code=r, name=r, endpoint=r) for r in regions]
            mock_client.from_config.return_value.__enter__.return_value.config.region = default_region
            with DWaveSampler(failover_regions=True) as sampler:
                self.assertEqual(len(sampler._clients), len(regions))
                calls = mock_client.from_config.call_args_list
                calls.pop(0)    # first client is used to fetch regions properly
                self.assertEqual(len(calls), len(regions))
                for i, region in enumerate(regions):
                    self.assertEqual(calls[i].kwargs.get('region'), region)


class TestDWaveSamplerMultiRegionFailoverConfig(unittest.TestCase):
    """Verify sampler/client(s) initialization when (region) config is loaded
    from environment variables, config files and/or kwargs.

    A real `Client` is used, but all network access (regions and solvers
    fetching) is mocked out.
    """

    metadata_api_endpoint = 'https://mock.dwavesys.com/metadata/v1/'

    # note: DEFAULT_REGION has to be available in the mock metadata api, as it's
    # used when region is not defined; also, we don't want it to be the first
    # region returned by the metadata api (so we can test region prioritization)
    region_codes = ['mock-region-1', 'mock-region-2', DEFAULT_REGION]

    config_body = textwrap.dedent(f"""
        [defaults]
        token = mock-token

        [default]
        metadata_api_endpoint = {metadata_api_endpoint}

        [regional]
        metadata_api_endpoint = {metadata_api_endpoint}
        region = {region_codes[0]}
    """)

    def setUp(self):
        self.regions = [Region(code=code, name=code,
                               endpoint=f'https://{code}.mock.dwavesys.com/sapi/v2/')
                        for code in self.region_codes]
        self.endpoints = {r.code: r.endpoint for r in self.regions}

        def start(patcher):
            self.addCleanup(patcher.stop)
            return patcher.start()

        # prevent run-time env vars and config files interfering with tests
        start(isolated_environ(remove_dwave=True))
        start(mock.patch('dwave.cloud.config.loaders.get_configfile_paths', lambda **kw: []))

        # token is required for client init, but it's otherwise irrelevant here
        start(isolated_environ(add={'DWAVE_API_TOKEN': 'mock-token'}))

        # mock regions fetch from metadata api (used by both `Client` endpoint
        # resolution and `get_regions` in the sampler)
        self.fetch_regions = start(mock.patch(
            'dwave.cloud.regions._fetch_available_regions', return_value=self.regions))

        # mock solvers fetch from solver api (one solver per client/region)
        solver = StructuredSolver(data=mocks.qpu_pegasus_solver_data(4), client=None)
        start(mock.patch.object(Client, 'create_session', lambda client: mock.Mock()))
        start(mock.patch.object(Client, '_fetch_solvers', mock_fetch_solvers(solver)))

    @contextlib.contextmanager
    def config_file(self, body=None):
        """Simulate a config file, referenced via `DWAVE_CONFIG_FILE` env var."""
        if body is None:
            body = self.config_body
        with isolated_environ(add={'DWAVE_CONFIG_FILE': 'mock-config-file'}):
            with mock.patch('dwave.cloud.config.loaders.open', iterable_mock_open(body)):
                yield

    @contextlib.contextmanager
    def sampler(self, **kwargs):
        """Create a sampler, and make sure all clients are closed on exit."""
        sampler = DWaveSampler(**kwargs)
        try:
            yield sampler
        finally:
            sampler.close()

    def assert_regions_fetched_from(self, metadata_api_endpoint):
        self.assertTrue(self.fetch_regions.called)
        for call in self.fetch_regions.call_args_list:
            self.assertEqual(call.kwargs['config'].metadata_api_endpoint,
                             metadata_api_endpoint)

    def assert_clients_consistent(self, sampler):
        # default client is the first one, and client regions match failover regions
        self.assertIs(sampler.client, sampler._clients[0])
        self.assertIs(sampler.solver.client, sampler.client)
        self.assertEqual([client.config.region for client in sampler._clients],
                         sampler.failover_regions)

    def verify_metadata_api_endpoint(self):
        # DWaveSampler() with nonstandard metadata api endpoint configured
        with self.subTest("default region"):
            self.fetch_regions.reset_mock()
            with self.sampler() as sampler:
                self.assertEqual(len(sampler._clients), 1)
                self.assertEqual(sampler.client.config.metadata_api_endpoint,
                                 self.metadata_api_endpoint)
                self.assertEqual(sampler.client.config.region, DEFAULT_REGION)
                self.assertEqual(sampler.client.config.endpoint,
                                 self.endpoints[DEFAULT_REGION])
                self.assert_regions_fetched_from(self.metadata_api_endpoint)

        with self.subTest("explicit region"):
            self.fetch_regions.reset_mock()
            region = self.region_codes[0]
            with self.sampler(region=region) as sampler:
                self.assertEqual(len(sampler._clients), 1)
                self.assertEqual(sampler.client.config.metadata_api_endpoint,
                                 self.metadata_api_endpoint)
                self.assertEqual(sampler.client.config.region, region)
                self.assertEqual(sampler.client.config.endpoint, self.endpoints[region])
                self.assert_regions_fetched_from(self.metadata_api_endpoint)

    def test_metadata_api_endpoint_from_env(self):
        with isolated_environ(add={'DWAVE_METADATA_API_ENDPOINT': self.metadata_api_endpoint}):
            self.verify_metadata_api_endpoint()

    def test_metadata_api_endpoint_from_config_file(self):
        with self.config_file():
            self.verify_metadata_api_endpoint()

    def test_metadata_api_endpoint_from_env_over_config_file(self):
        # env var overrides the value from config file
        config_body = textwrap.dedent("""
            [defaults]
            metadata_api_endpoint = https://unused.mock.dwavesys.com/metadata/v1/
        """)
        with self.config_file(config_body):
            with isolated_environ(add={'DWAVE_METADATA_API_ENDPOINT': self.metadata_api_endpoint}):
                self.verify_metadata_api_endpoint()

    def test_profile_from_config_file(self):
        with self.config_file():
            with self.subTest("region from profile"):
                self.fetch_regions.reset_mock()
                region = self.region_codes[0]
                with self.sampler(profile='regional') as sampler:
                    self.assertEqual(len(sampler._clients), 1)
                    self.assertEqual(sampler.client.config.metadata_api_endpoint,
                                     self.metadata_api_endpoint)
                    self.assertEqual(sampler.client.config.region, region)
                    self.assertEqual(sampler.client.config.endpoint, self.endpoints[region])
                    self.assert_regions_fetched_from(self.metadata_api_endpoint)

            with self.subTest("region kwarg overrides region from profile"):
                self.fetch_regions.reset_mock()
                region = self.region_codes[1]
                with self.sampler(profile='regional', region=region) as sampler:
                    self.assertEqual(len(sampler._clients), 1)
                    self.assertEqual(sampler.client.config.metadata_api_endpoint,
                                     self.metadata_api_endpoint)
                    self.assertEqual(sampler.client.config.region, region)
                    self.assertEqual(sampler.client.config.endpoint, self.endpoints[region])
                    self.assert_regions_fetched_from(self.metadata_api_endpoint)

            with self.subTest("nonexisting profile"):
                with self.assertRaises(ValueError):
                    DWaveSampler(profile='nonexisting')

    def test_failover_regions_all_from_nonstandard_metadata_api(self):
        # with `failover_regions=True`, regions are fetched from the metadata
        # api endpoint configured via env or config file

        @contextlib.contextmanager
        def endpoint_from_env():
            with isolated_environ(add={'DWAVE_METADATA_API_ENDPOINT': self.metadata_api_endpoint}):
                yield

        @contextlib.contextmanager
        def endpoint_from_file():
            with self.config_file():
                yield

        for source in (endpoint_from_env, endpoint_from_file):
            with self.subTest(source=source.__name__):
                self.fetch_regions.reset_mock()
                with source(), self.sampler(failover_regions=True) as sampler:
                    # all regions from the metadata api are used, default region first
                    self.assertEqual(len(sampler._clients), len(self.regions))
                    self.assertEqual(sampler.failover_regions[0], DEFAULT_REGION)
                    self.assertEqual(set(sampler.failover_regions), set(self.region_codes))
                    self.assert_regions_fetched_from(self.metadata_api_endpoint)
                    self.assert_clients_consistent(sampler)
                    # all clients use the configured metadata api endpoint, and
                    # each client's solver api endpoint is resolved from it
                    for client in sampler._clients:
                        self.assertEqual(client.config.metadata_api_endpoint,
                                         self.metadata_api_endpoint)
                        self.assertEqual(client.config.endpoint,
                                         self.endpoints[client.config.region])

    def test_failover_regions_prioritize_configured_region(self):
        # when region is defined (kwarg, env var, or config file), it's the first
        # in the list of failover regions, regardless of how failover_regions are
        # defined (all available, or an explicit list)
        region = self.region_codes[1]
        self.assertNotEqual(region, DEFAULT_REGION)

        @contextlib.contextmanager
        def region_from_kwarg():
            yield dict(region=region)

        @contextlib.contextmanager
        def region_from_env():
            with isolated_environ(add={'DWAVE_API_REGION': region}):
                yield {}

        @contextlib.contextmanager
        def region_from_file():
            config_body = textwrap.dedent(f"""
                [defaults]
                region = {region}
            """)
            with self.config_file(config_body):
                yield {}

        @contextlib.contextmanager
        def region_from_profile():
            config_body = textwrap.dedent(f"""
                [unused]
                region = {self.region_codes[0]}

                [regional]
                region = {region}
            """)
            with self.config_file(config_body):
                yield dict(profile='regional')

        # explicit failover regions list, in which `region` is not first
        explicit_regions = list(reversed(self.region_codes))
        self.assertNotEqual(explicit_regions[0], region)

        for source in (region_from_kwarg, region_from_env, region_from_file, region_from_profile):
            for failover_regions in (True, explicit_regions):
                with self.subTest(source=source.__name__, failover_regions=failover_regions):
                    with source() as kwargs, self.sampler(failover_regions=failover_regions, **kwargs) as sampler:
                        self.assertEqual(sampler.failover_regions[0], region)
                        self.assertEqual(set(sampler.failover_regions), set(self.region_codes))
                        self.assertEqual(len(sampler._clients), len(self.region_codes))
                        self.assertEqual(sampler.client.config.region, region)
                        self.assert_clients_consistent(sampler)

    def test_failover_across_regions(self):
        # on failover, the active client follows the selected solver, cycling
        # through regions in failover_regions order
        with self.sampler(failover=True, failover_regions=self.region_codes) as sampler:
            # default region (resolved on client init) is prioritized
            self.assertEqual(sampler.failover_regions[0], DEFAULT_REGION)
            self.assertEqual(set(sampler.failover_regions), set(self.region_codes))

            for _ in range(2):      # verify wrap-around as well
                for client in sampler._clients:
                    self.assertIs(sampler.client, client)
                    self.assertIs(sampler.solver.client, client)
                    sampler.trigger_failover()

    def test_failover_regions_exclude_configured_region(self):
        # configured region has to be in the explicit list of failover regions
        region = self.region_codes[0]
        failover_regions = self.region_codes[1:]
        self.assertNotIn(region, failover_regions)

        with self.subTest("region from kwarg"):
            with self.assertRaises(ValueError):
                DWaveSampler(failover_regions=failover_regions, region=region)

        with self.subTest("region from env"):
            with isolated_environ(add={'DWAVE_API_REGION': region}):
                with self.assertRaises(ValueError):
                    DWaveSampler(failover_regions=failover_regions)

        with self.subTest("region from config file"):
            with self.config_file():
                with self.assertRaises(ValueError):
                    DWaveSampler(failover_regions=failover_regions, profile='regional')


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
