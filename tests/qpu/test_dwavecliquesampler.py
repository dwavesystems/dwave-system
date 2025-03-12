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

import dwave.cloud.exceptions
from dwave.system import DWaveCliqueSampler

from parameterized import parameterized

_SAMPLERS = {}


def get_sampler(topology):
    if topology in _SAMPLERS:
        return _SAMPLERS[topology]
    try:
        _SAMPLERS[topology] = DWaveCliqueSampler(solver=dict(topology__type=topology.lower()))
        return _SAMPLERS[topology]
    except (ValueError,
            dwave.cloud.exceptions.ConfigFileError,
            dwave.cloud.exceptions.SolverNotFoundError):
        raise unittest.SkipTest(f"no {topology}-structured QPU available")


def tearDownModule():
    # make sure all cached samplers are closed and resources released at exit
    for sampler in _SAMPLERS.values():
        sampler.close()


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

        # if the range was not adjusted, this would raise an error.
        sampler.sample(bqm, chain_strength=chain_strength).resolve()

    @unittest.skipUnless(hasattr(dwave.cloud.exceptions, 'UseAfterClose'), 'dwave-cloud-client>=0.13.3 required')
    def test_close(self):
        sampler = DWaveCliqueSampler()
        sampler.close()

        with self.assertRaises(dwave.cloud.exceptions.UseAfterCloseError):
            sampler.sample_qubo({(0, 1): 1})
