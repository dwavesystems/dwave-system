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
import numpy as np

from dwave.cloud.exceptions import ConfigFileError, SolverNotFoundError
from dwave.system import LeapHybridDQMSampler

try:
    sampler = LeapHybridDQMSampler()
except (ValueError, ConfigFileError, SolverNotFoundError):
    sampler = None


@unittest.skipIf(os.getenv('SKIP_INT_TESTS'), "Skipping integration test.")
@unittest.skipIf(sampler is None, "no hybrid sampler available")
class TestLeapHybridSampler(unittest.TestCase):
    def test_smoke(self):
        dqm = dimod.DQM()
        u = dqm.add_variable(2, 'a')
        v = dqm.add_variable(3)
        dqm.set_linear(u, [1, 2])
        dqm.set_quadratic(u, v, {(0, 1): 1, (0, 2): 1})

        try:
            # dimod 0.10+
            dqm.offset += 3
        except AttributeError:
            pass

        sampleset = sampler.sample_dqm(dqm)

        np.testing.assert_array_almost_equal(dqm.energies(sampleset),
                                             sampleset.record.energy)

    @unittest.skipIf(tuple(map(int, dimod.__version__.split(".")[:3])) < (0, 12, 7),
                     "Skipping for dimod version < 0.12.7, "
                     "see https://github.com/dwavesystems/dimod/pull/1332")
    def test_smoke_case_label(self):
        try:
            from dimod import CaseLabelDQM
        except ImportError:
            self.skipTest("need dimod 0.10+")

        dqm = CaseLabelDQM()
        u = dqm.add_variable({'red', 'green', 'blue'}, shared_labels=True)
        dqm.offset = 5

        sampleset = sampler.sample_dqm(dqm)

        np.testing.assert_array_almost_equal(dqm.energies(sampleset),
                                             sampleset.record.energy)

    # NOTE: enable when problem labelling deployed to prod
    @unittest.skipIf(sampler is not None and 'cloud' in sampler.client.endpoint,
                     "labels not supported in production")
    def test_problem_labelling(self):
        dqm = dimod.DQM()
        u = dqm.add_variable(2, 'a')
        v = dqm.add_variable(3)
        dqm.set_linear(u, [1, 2])
        dqm.set_quadratic(u, v, {(0, 1): 1, (0, 2): 1})
        label = 'problem label'

        sampleset = sampler.sample_dqm(dqm, label=label)
        self.assertIn('problem_id', sampleset.info)
        self.assertIn('problem_label', sampleset.info)
        self.assertEqual(sampleset.info['problem_label'], label)
