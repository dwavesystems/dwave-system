# Copyright 2019 D-Wave Systems Inc.
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

import os
import unittest

from dwave.cloud.exceptions import ConfigFileError, SolverNotFoundError

from dwave.system.schedules import ramp
from dwave.system.samplers import DWaveSampler


@unittest.skipIf(os.getenv('SKIP_INT_TESTS'), "Skipping integration test.")
class TestRamp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.qpu = DWaveSampler(solver=dict(h_gain_schedule=True))
        except (ValueError, ConfigFileError, SolverNotFoundError):
            raise unittest.SkipTest("no qpu available")

    @classmethod
    def tearDownClass(cls):
        cls.qpu.client.close()

    def test_with_h_gain_schedule(self):
        sampler = self.qpu

        schedule = ramp(.5, .2, sampler.properties['default_annealing_time'])

        sampler.validate_anneal_schedule(schedule)

        h = {v: 1 for v in sampler.nodelist}

        sampleset = sampler.sample_ising(h, {}, h_gain_schedule=schedule)
        sampleset.record  # resolve the future
