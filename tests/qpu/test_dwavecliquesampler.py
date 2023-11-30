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
import warnings

import dimod

from dwave.cloud.exceptions import ConfigFileError, SolverNotFoundError
from dwave.system import DWaveCliqueSampler

from parameterized import parameterized

_SAMPLERS = {}


def get_sampler(topology):
    if topology in _SAMPLERS:
        return _SAMPLERS[topology]
    try:
        _SAMPLERS[topology] = DWaveCliqueSampler(solver=dict(topology__type=topology.lower()))
        return _SAMPLERS[topology]
    except (ValueError, ConfigFileError, SolverNotFoundError):
        raise unittest.SkipTest(f"no {topology}-structured QPU available")


@unittest.skipIf(os.getenv('SKIP_INT_TESTS'), "Skipping integration test.")
class TestDWaveCliqueSampler(unittest.TestCase):
    @parameterized.expand([['Pegasus'], ['Zephyr']])
    def test_maximum_ferromagnet(self, topology):
        sampler = get_sampler(topology)

        dimod.testing.assert_sampler_api(sampler)

        bqm = dimod.BinaryQuadraticModel('SPIN')
        for u, v in itertools.combinations(sampler.largest_clique(), 2):
            bqm.quadratic[u, v] = -1

        sampler.sample(bqm).resolve()

    @parameterized.expand(itertools.product(('Pegasus', 'Zephyr'), (None, 0.5)))
    def test_per_qubit_coupling_range(self, topology, chain_strength):
        sampler = get_sampler(topology)
        n = sampler.largest_clique_size

        bqm = dimod.BinaryQuadraticModel({},
                {(u, v): -2 for u in range(n) for v in range(u+1, n)}, 'SPIN')

        with warnings.catch_warnings(record=True) as w:
            sampler.sample(bqm, chain_strength=chain_strength).resolve()

        if topology == 'Pegasus':
            limit_name = 'per_qubit_coupling_range'
        else:
            limit_name = 'per_group_coupling_range'

        if chain_strength is not None:
            self.assertGreaterEqual(len(w), 1)
            self.assertEqual(sum(f'{limit_name} is violated' in str(w[i].message)
                                 for i in range(len(w))), 1)
        else:
            # this is not very robust, but does the job
            self.assertEqual(sum(f'{limit_name} is violated' in str(w[i].message)
                                 for i in range(len(w))), 0)
