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

import os
import unittest

import dimod
from dwave.cloud.exceptions import ConfigFileError, SolverNotFoundError

from dwave.system import LeapHybridSampler


@unittest.skipIf(os.getenv('SKIP_INT_TESTS'), "Skipping integration test.")
# @dimod.testing.load_sampler_bqm_tests(sampler)  # these take a while
class TestLeapHybridSampler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.sampler = LeapHybridSampler()
        except (ValueError, ConfigFileError, SolverNotFoundError):
            raise unittest.SkipTest("hybrid sampler not available")

    @classmethod
    def tearDownClass(cls):
        cls.sampler.close()

    def test_smoke(self):
        sampleset = self.sampler.sample_ising({'a': -1}, {'ab': 1})
        sampleset.resolve()

    def test_problem_labelling(self):
        bqm = dimod.BQM.from_ising({'a': -1}, {'ab': 1})
        label = 'problem label'

        # label set
        sampleset = self.sampler.sample(bqm, label=label)
        self.assertIn('problem_id', sampleset.info)
        self.assertIn('problem_label', sampleset.info)
        self.assertEqual(sampleset.info['problem_label'], label)

        # label unset
        sampleset = self.sampler.sample(bqm)
        self.assertIn('problem_id', sampleset.info)
        self.assertNotIn('problem_label', sampleset.info)
