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

import dimod

from dwave.cloud.exceptions import ConfigFileError, SolverNotFoundError
from dwave.system import DWaveCliqueSampler


class TestDWaveCliqueSampler(unittest.TestCase):
    def test_chimera(self):
        try:
            sampler = DWaveCliqueSampler(solver=dict(topology__type='chimera'))
        except (ValueError, ConfigFileError, SolverNotFoundError):
            raise unittest.SkipTest("no Chimera-structured QPU available")

        dimod.testing.assert_sampler_api(sampler)

        # submit a maximum ferromagnet
        bqm = dimod.AdjVectorBQM('SPIN')
        for u, v in itertools.combinations(sampler.largest_clique(), 2):
            bqm.quadratic[u, v] = -1

        sampler.sample(bqm).resolve()

    def test_pegasus(self):
        try:
            sampler = DWaveCliqueSampler(solver=dict(topology__type='pegasus'))
        except (ValueError, ConfigFileError, SolverNotFoundError):
            raise unittest.SkipTest("no Pegasus-structured QPU available")

        dimod.testing.assert_sampler_api(sampler)

        # submit a maximum ferromagnet
        bqm = dimod.AdjVectorBQM('SPIN')
        for u, v in itertools.combinations(sampler.largest_clique(), 2):
            bqm.quadratic[u, v] = -1

        sampler.sample(bqm).resolve()
