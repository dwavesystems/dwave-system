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
import os
import unittest

import dimod

from dwave.cloud.exceptions import ConfigFileError, SolverNotFoundError
from dwave.system import DWaveCliqueSampler

from parameterized import parameterized


def get_sampler(topology):
    try:
        return DWaveCliqueSampler(solver=dict(topology__type=topology.lower()))
    except (ValueError, ConfigFileError, SolverNotFoundError):
        raise unittest.SkipTest(f"no {topology}-structured QPU available")


@unittest.skipIf(os.getenv('SKIP_INT_TESTS'), "Skipping integration test.")
class TestDWaveCliqueSampler(unittest.TestCase):
    @parameterized.expand([['Chimera'], ['Pegasus'], ['Zephyr']])
    def test_maximum_ferromagnet(self, topology):
        sampler = get_sampler(topology)

        dimod.testing.assert_sampler_api(sampler)

        bqm = dimod.BinaryQuadraticModel('SPIN')
        for u, v in itertools.combinations(sampler.largest_clique(), 2):
            bqm.quadratic[u, v] = -1

        sampler.sample(bqm).resolve()
