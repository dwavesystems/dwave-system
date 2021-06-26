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

from unittest.mock import patch

import dimod

from dwave.system import LeapHybridDQMSampler


class TestLeapHybridDQMSampler(unittest.TestCase):
    def test_time_limit_exceptions(self):

        class MockSolver():
            properties = dict(category='hybrid',
                              minimum_time_limit=[[20000, 5.0],
                                                  [100000, 6.0],
                                                  [200000, 13.0],
                                                  [500000, 34.0],
                                                  [1000000, 71.0],
                                                  [2000000, 152.0],
                                                  [5000000, 250.0],
                                                  [20000000, 400.0],
                                                  [250000000, 1200.0]],
                              maximum_time_limit_hrs=24.0,
                              )
            supported_problem_types = ['dqm']

            def sample_dqm(self, *args, **kwargs):
                raise RuntimeError

        class MockClient():
            @classmethod
            def from_config(cls, *args, **kwargs):
                return cls()

            def get_solver(self, *args, **kwargs):
                return MockSolver()

        with patch('dwave.system.samplers.leap_hybrid_sampler.Client', MockClient):
            sampler = LeapHybridDQMSampler()

            dqm = dimod.DQM()

        with self.assertRaises(ValueError):
            sampler.sample_dqm(dqm, time_limit=1)

        with self.assertRaises(ValueError):
            sampler.sample_dqm(dqm, time_limit=10000000)

    def test_DQM_subclass_without_serialization_can_be_sampled(self):
        """Test that DQM subclasses that do not implement serialization can
        still be sampled by LeapHybridDQMSampler.

        Sampling requires serialization.  LeapHybridDQMSampler calls
        DQM.to_file(dqm, ...) if dqm.to_file(...) raises NotImplementedError.
        """

        class DQMWithoutSerialization(dimod.DQM):
            def to_file(self, *args, **kwargs):
                raise NotImplementedError

        class MockSampleset:
            def relabel_variables(self, *args):
                return self

        class MockFuture:
            sampleset = MockSampleset()

        class MockSolver:
            properties = dict(category='hybrid',
                              minimum_time_limit=[[10000, 1.0]]
                              )
            supported_problem_types = ['dqm']

            def sample_dqm(self, *args, **kwargs):
                return MockFuture()

        class MockClient:
            @classmethod
            def from_config(cls, *args, **kwargs):
                return cls()

            def get_solver(self, *args, **kwargs):
                return MockSolver()

        with patch('dwave.system.samplers.leap_hybrid_sampler.Client', MockClient):
            sampler = LeapHybridDQMSampler()

            dqm = DQMWithoutSerialization()

        sampler.sample_dqm(dqm)

    def test_filter_sampleset(self):

        class DQMWithFilterSampleset(dimod.DQM):
            def _filter_sampleset(self, sampleset):
                sampleset.transformed = True
                return sampleset

        class MockSampleset:
            pass

        class MockFuture:
            sampleset = MockSampleset()

        class MockSolver:
            properties = dict(category='hybrid',
                              minimum_time_limit=[[10000, 1.0]]
                              )
            supported_problem_types = ['dqm']

            def sample_dqm(self, *args, **kwargs):
                return MockFuture()

        class MockClient:
            @classmethod
            def from_config(cls, *args, **kwargs):
                return cls()

            def get_solver(self, *args, **kwargs):
                return MockSolver()

        with patch('dwave.system.samplers.leap_hybrid_sampler.Client', MockClient):
            sampler = LeapHybridDQMSampler()

            dqm = DQMWithFilterSampleset()

        self.assertTrue(sampler.sample_dqm(dqm).transformed)
