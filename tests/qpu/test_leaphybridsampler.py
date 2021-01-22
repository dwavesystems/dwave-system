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

import unittest

import dimod

from dwave.cloud.exceptions import ConfigFileError, SolverNotFoundError
from dwave.system import LeapHybridSampler

try:
    sampler = LeapHybridSampler()
except (ValueError, ConfigFileError, SolverNotFoundError):
    sampler = None


@unittest.skipIf(sampler is None, "no hybrid sampler available")
# @dimod.testing.load_sampler_bqm_tests(sampler)  # these take a while
class TestLeapHybridSampler(unittest.TestCase):
    def test_smoke(self):
        sampleset = sampler.sample_ising({'a': -1}, {'ab': 1})
        sampleset.resolve()

    # NOTE: enable when problem labelling deployed to prod
    @unittest.skipIf(sampler is not None and 'cloud' in sampler.client.endpoint,
                     "labels not supported in production")
    def test_problem_labelling(self):
        bqm = dimod.BQM.from_ising({'a': -1}, {'ab': 1})
        label = 'problem label'

        # label set
        sampleset = sampler.sample(bqm, label=label)
        self.assertIn('problem_id', sampleset.info)
        self.assertIn('problem_label', sampleset.info)
        self.assertEqual(sampleset.info['problem_label'], label)

        # label unset
        sampleset = sampler.sample(bqm)
        self.assertIn('problem_id', sampleset.info)
        self.assertNotIn('problem_label', sampleset.info)
