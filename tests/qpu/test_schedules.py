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
#
# =============================================================================
import unittest

from dwave.cloud.exceptions import ConfigFileError

from dwave.system.schedules import ramp
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

try:
    splr = DWaveSampler()
except (ValueError, ConfigFileError):
    _config_found = False
else:
    _config_found = True
    splr.client.close()


@unittest.skipUnless(_config_found, "no configuration found to connect to a system")
class TestRamp(unittest.TestCase):
    def test_with_h_gain_schedule(self):

        sampler = DWaveSampler(solver=dict(qpu=True, h_gain_schedule=True))

        schedule = ramp(.5, .2, sampler.properties['default_annealing_time'])

        #sampler.validate_anneal_schedule(schedule)

        h = {v: 1 for v in sampler.nodelist}

        sampleset = sampler.sample_ising(h, {}, h_gain_schedule=schedule)
        sampleset.record  # resolve the future
