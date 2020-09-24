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
from dwave.system import LeapHybridDQMSampler

try:
    sampler = LeapHybridDQMSampler()
except (ValueError, ConfigFileError, SolverNotFoundError):
    sampler = None


@unittest.skipIf(sampler is None, "no hybrid sampler available")
class TestLeapHybridSampler(unittest.TestCase):
    def test_smoke(self):
        dqm = dimod.DQM()
        u = dqm.add_variable(2, 'a')
        v = dqm.add_variable(3)
        dqm.set_linear(u, [1, 2])
        dqm.set_quadratic(u, v, {(0, 1): 1, (0, 2): 1})

        sampleset = sampler.sample_dqm(dqm)
        sampleset.resolve()
