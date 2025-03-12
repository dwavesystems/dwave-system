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

import itertools
import unittest
import unittest.mock

from parameterized import parameterized

import dimod
import dwave_networkx as dnx

from dwave.cloud import exceptions
from dwave.cloud import computation
from dwave.system import DWaveCliqueSampler, DWaveSampler
from dwave.system.exceptions import FailoverCondition, RetryCondition


class MockDWaveSampler(dimod.RandomSampler, dimod.Structured):
    # contains the minimum needed to work with DWaveCliqueSampler

    edgelist = None
    nodelist = None

    def __init__(self, **kwargs):
        self.properties = dict(h_range=[-2, 2],
                               j_range=[-1, 1],
                               extended_j_range=[-2, 1],)
        self.parameters = {'auto_scale': None}

    def sample(self, bqm, auto_scale=True):
        assert not auto_scale
        assert bqm.vartype is dimod.SPIN

        h_range = self.properties['h_range']
        j_range = self.properties['extended_j_range']

        for bias in bqm.linear.values():
            assert h_range[0] <= bias <= h_range[1]

        for bias in bqm.quadratic.values():
            assert j_range[0] <= bias <= j_range[1]

        return super().sample(bqm)

    def trigger_failover(self):
        pass

    to_networkx_graph = DWaveSampler.to_networkx_graph


class MockChimeraDWaveSampler(MockDWaveSampler):
    def __init__(self, **config):
        super().__init__()

        self.properties.update(topology=dict(shape=[4, 4, 4], type='chimera'))

        G = dnx.chimera_graph(4, 4, 4)

        self.nodelist = list(G.nodes)
        self.edgelist = list(G.edges)

    def sample(self, bqm, **kwargs):

        # per_qubit_coupling_range
        ran = (-9, 6)

        # check the total coupling range
        for v in bqm.variables:
            bias = sum(bqm.adj[v].values())
            assert ran[0] <= bias <= ran[1]

        return super().sample(bqm, **kwargs)


class MockPegasusDWaveSampler(MockDWaveSampler):
    def __init__(self, **config):
        super().__init__()

        self.properties.update(topology=dict(shape=[6], type='pegasus'))

        G = dnx.pegasus_graph(6)

        self.nodelist = list(G.nodes)
        self.edgelist = list(G.edges)


with unittest.mock.patch('dwave.system.samplers.clique.DWaveSampler',
                         MockChimeraDWaveSampler):
    chimera_sampler = DWaveCliqueSampler()

with unittest.mock.patch('dwave.system.samplers.clique.DWaveSampler',
                         MockPegasusDWaveSampler):
    pegasus_sampler = DWaveCliqueSampler()


@dimod.testing.load_sampler_bqm_tests(chimera_sampler)
@dimod.testing.load_sampler_bqm_tests(pegasus_sampler)
class TestDWaveCliqueSampler(unittest.TestCase):
    def test_api(self):
        dimod.testing.assert_sampler_api(chimera_sampler)
        dimod.testing.assert_sampler_api(pegasus_sampler)

    def test_clique(self):
        self.assertEqual(len(chimera_sampler.clique(2)), 2)
        self.assertEqual(len(chimera_sampler.clique(16)), 16)

        self.assertEqual(len(chimera_sampler.clique(['a', 1])), 2)
        self.assertEqual(set(chimera_sampler.clique(['a', 1])), {'a', 1})

    def test_largest_clique(self):
        self.assertEqual(len(chimera_sampler.largest_clique()), 16)

    def test_ferromagnet_chimera(self):
        # submit a maximum ferromagnet
        bqm = dimod.BinaryQuadraticModel('SPIN')
        for u, v in itertools.combinations(chimera_sampler.largest_clique(), 2):
            bqm.quadratic[u, v] = -1

        chimera_sampler.sample(bqm).resolve()

    def test_too_large(self):
        num_variables = chimera_sampler.largest_clique_size + 1

        bqm = dimod.BinaryQuadraticModel(num_variables, 'SPIN')

        with self.assertRaises(ValueError):
            chimera_sampler.sample(bqm)

    def test_qubit_coupling_range(self):
        n = pegasus_sampler.largest_clique_size

        bqm = dimod.BinaryQuadraticModel({}, 
                {(u, v): -2 for u in range(n) for v in range(u+1, n)}, 'SPIN')

        pegasus_sampler.sample(bqm, chain_strength=-0.5).resolve()


class TestFailover(unittest.TestCase):

    @staticmethod
    def patch_solver(solver, exc):
        """Patch mock `solver` to fail with `exc` on sampleset resolve."""

        # clique sampler needs a valid topology
        solver.properties = {
            'topology': {
                'type': 'pegasus',
                'shape': [4]
            },
            'parameters': {},
            'h_range': [-2, 2],
            'j_range': [-1, 1]
        }

        # simulate solver failure on sampleset resolve
        fut = computation.Future(solver, None)
        fut._set_exception(exc)
        solver.sample_bqm = unittest.mock.Mock()
        solver.sample_bqm.return_value = fut

        return solver

    @unittest.mock.patch('dwave.system.samplers.dwave_sampler.Client')
    def test_default(self, mock_client):
        sampler = DWaveCliqueSampler()

        self.patch_solver(sampler.child.solver, exceptions.SolverOfflineError)

        with self.assertRaises(exceptions.SolverOfflineError):
            sampler.sample_ising({}, {}).resolve()

    @parameterized.expand([
        (exceptions.InvalidAPIResponseError, FailoverCondition),
        (exceptions.SolverNotFoundError, FailoverCondition),
        (exceptions.SolverOfflineError, FailoverCondition),
        (exceptions.SolverError, FailoverCondition),
        (exceptions.PollingTimeout, RetryCondition),
        (exceptions.SolverAuthenticationError, exceptions.SolverAuthenticationError),   # auth error propagated
        (KeyError, KeyError),   # unrelated errors propagated
    ])
    @unittest.mock.patch('dwave.system.samplers.dwave_sampler.Client')
    def test_async_failover(self, source_exc, target_exc, mock_client):
        sampler = DWaveCliqueSampler(failover=True)

        self.patch_solver(sampler.child.solver, source_exc)

        with self.assertRaises(target_exc):
            sampler.sample_ising({}, {}).resolve()

    @unittest.mock.patch('dwave.system.samplers.clique.DWaveSampler',
                         MockChimeraDWaveSampler)
    def test_properties_reinit(self):
        sampler = DWaveCliqueSampler()

        G = sampler.target_graph
        qlr = sampler.qpu_linear_range
        qqr = sampler.qpu_quadratic_range

        sampler.trigger_failover()

        self.assertIsNot(G, sampler.target_graph)
        self.assertIsNot(qlr, sampler.qpu_linear_range)
        self.assertIsNot(qqr, sampler.qpu_quadratic_range)

    @unittest.mock.patch('dwave.system.samplers.clique.DWaveSampler')
    def test_close(self, mock_child_sampler):
        sampler = DWaveCliqueSampler()
        sampler.close()

        mock_child_sampler.return_value.close.assert_called_once()

    @unittest.mock.patch('dwave.system.samplers.clique.DWaveSampler')
    def test_context_manager(self, mock_child_sampler):
        with DWaveCliqueSampler() as sampler:
            ...

        mock_child_sampler.return_value.close.assert_called_once()
