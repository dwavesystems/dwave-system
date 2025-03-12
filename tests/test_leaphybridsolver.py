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

import unittest
import unittest.mock as mock
import numpy as np

import dimod
from dwave.cloud import Client
from dwave.cloud.exceptions import SolverNotFoundError

from dwave.system.samplers import LeapHybridSampler
from dwave.system.testing import MockLeapHybridSolver

# Called only for named solver
class MockBadLeapHybridSolver:

    properties = {'category': 'not hybrid'}

class MockClient:

    def __init__(self, **kwargs):

        self.args = kwargs

    def get_solver(self, **filters):

        if self.args.get('solver') == 'not_hybrid_solver':
            return MockBadLeapHybridSolver()

        if self.args.get('client', 'base') not in ['base', 'hybrid']:
            raise SolverNotFoundError

        return MockLeapHybridSolver()

class TestLeapHybridSampler(unittest.TestCase):

    @mock.patch('dwave.system.samplers.leap_hybrid_sampler.Client')
    def test_solver_init(self, mock_client):
        mock_client.from_config.side_effect = MockClient

        default_solver = dict(
            supported_problem_types__contains='bqm',
            order_by='-properties.version')
        self.assertEqual(LeapHybridSampler.default_solver, default_solver)

        defaults = dict(solver=default_solver)

        # Default call
        mock_client.reset_mock()
        LeapHybridSampler()
        mock_client.from_config.assert_called_once_with(
            client='hybrid',
            connection_close=True,
            defaults=defaults)

        # Non-hybrid client setting
        mock_client.reset_mock()
        with self.assertRaises(SolverNotFoundError):
            LeapHybridSampler(client='qpu')

        # Explicitly set solver def
        mock_client.reset_mock()
        LeapHybridSampler(solver={'supported_problem_types__contains': 'bqm'})
        mock_client.from_config.assert_called_once_with(
            client='hybrid',
            solver={'supported_problem_types__contains': 'bqm'},
            connection_close=True,
            defaults=defaults)

        # Named solver
        solver_name = 'hybrid-solver-name'
        mock_client.reset_mock()
        LeapHybridSampler(solver=solver_name)
        mock_client.from_config.assert_called_once_with(
            client='hybrid',
            solver=solver_name,
            connection_close=True,
            defaults=defaults)

        # Named solver: non-hybrid
        with self.assertRaises(ValueError):
            LeapHybridSampler(solver='not_hybrid_solver')

        # Set connection_close to False
        mock_client.reset_mock()
        LeapHybridSampler(connection_close=False)
        mock_client.from_config.assert_called_once_with(
            client='hybrid',
            connection_close=False,
            defaults=defaults)

        mock_client.reset_mock()
        LeapHybridSampler(connection_close=False, solver=solver_name)
        mock_client.from_config.assert_called_once_with(
            client='hybrid',
            solver=solver_name,
            connection_close=False,
            defaults=defaults)

    @mock.patch('dwave.system.samplers.leap_hybrid_sampler.Client')
    def test_close(self, mock_client):
        mock_solver = mock_client.from_config.return_value.get_solver.return_value
        mock_solver.properties = {'category': 'hybrid'}
        mock_solver.supported_problem_types = ['bqm']

        with self.subTest('manual close'):
            sampler = LeapHybridSampler()
            sampler.close()
            mock_client.from_config.return_value.close.assert_called_once()

        mock_client.reset_mock()

        with self.subTest('context manager'):
            with LeapHybridSampler():
                ...
            mock_client.from_config.return_value.close.assert_called_once()

    @mock.patch('dwave.system.samplers.leap_hybrid_sampler.Client')
    def test_sample_bqm(self, mock_client):

        mock_client.from_config.side_effect = MockClient

        bqm = dimod.BinaryQuadraticModel({'a': -1, 'b': 1, 'c': 1},
                    {'ab': -0.8, 'ac': -0.7, 'bc': -1}, 0, dimod.SPIN)

        sampler = LeapHybridSampler()

        response = sampler.sample(bqm)

        rows, cols = response.record.sample.shape

        self.assertEqual(cols, 3)
        self.assertFalse(np.any(response.record.sample == 0))
        self.assertIs(response.vartype, dimod.SPIN)
        self.assertIn('num_occurrences', response.record.dtype.fields)

    @mock.patch('dwave.system.samplers.leap_hybrid_sampler.Client')
    def test_sample_ising_variables(self, mock_client):

        mock_client.from_config.side_effect = MockClient

        sampler = LeapHybridSampler()

        response = sampler.sample_ising({0: -1, 1: 1}, {})

        rows, cols = response.record.sample.shape

        self.assertEqual(cols, 2)

        response = sampler.sample_ising({}, {(0, 4): 1})

        rows, cols = response.record.sample.shape
        self.assertEqual(cols, 2)
        self.assertFalse(np.any(response.record.sample == 0))
        self.assertIs(response.vartype, dimod.SPIN)

    @mock.patch('dwave.system.samplers.leap_hybrid_sampler.Client')
    def test_sample_qubo_variables(self, mock_client):

        mock_client.from_config.side_effect = MockClient

        sampler = LeapHybridSampler()

        response = sampler.sample_qubo({(0, 0): -1, (1, 1): 1})

        rows, cols = response.record.sample.shape

        self.assertEqual(cols, 2)

        response = sampler.sample_qubo({(0, 0): -1, (1, 1): 1})

        rows, cols = response.record.sample.shape

        self.assertEqual(cols, 2)
        self.assertTrue(np.all(response.record.sample >= 0))
        self.assertIs(response.vartype, dimod.BINARY)
