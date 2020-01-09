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
import sys
import unittest
import random
import warnings

from collections import namedtuple
from concurrent.futures import Future

import numpy as np

import dimod
import dwave_networkx as dnx
from tabu import TabuSampler

from dwave.cloud.exceptions import SolverOfflineError, SolverNotFoundError

from dwave.system.samplers import LeapHybridSampler

try:
    # py3
    import unittest.mock as mock
except ImportError:
    # py2
    import mock

class MockSolver():

    def upload_bqm(bqm):
        return bqm

    def sample_bqm(sapi_problem_id, time_limit):
        return TabuSampler.sample(sapi_problem_id, timelimit=time_limit)


class TestLeapHybridSampler(unittest.TestCase):
    @mock.patch('dwave.system.samplers.leap_hybrid_sampler.Client')
    def setUp(self, MockClient):

        # using the mock
        self.sampler = LeapHybridSampler()

        self.sampler.solver = MockSolver()

    @mock.patch('dwave.system.samplers.leap_hybrid_sampler.Client')
    def test_solver_init(self, MockClient):
        """..."""

        MockClient.reset_mock()
        solver = {'qpu': False}
        sampler = LeapHybridSampler(solver=solver)

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
