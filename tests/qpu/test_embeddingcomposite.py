# Copyright 2018 D-Wave Systems Inc.
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
import warnings

import dimod

from dwave.cloud.exceptions import ConfigFileError

from dwave.system import DWaveSampler, EmbeddingComposite


@unittest.skipIf(os.getenv('SKIP_INT_TESTS'), "Skipping integration test.")
class TestEmbeddingCompositeExactSolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.qpu = DWaveSampler(
                solver=dict(initial_state=True, anneal_schedule=True))
        except (ValueError, ConfigFileError):
            raise unittest.SkipTest("no qpu available")

    @classmethod
    def tearDownClass(cls):
        cls.qpu.client.close()

    def test_initial_state(self):
        sampler = EmbeddingComposite(self.qpu)

        bqm = dimod.BinaryQuadraticModel.from_ising({'a': 2.0, 'b': -2.0},
                                                    {('a', 'b'): -1})

        kwargs = {'initial_state': {'a': 1, 'b': 1},
                  'anneal_schedule': [(0, 1), (55.0, 0.45),
                                      (155.0, 0.45), (210.0, 1)]}

        sampler.sample(bqm, **kwargs).resolve()

    def test_many_bqm_async(self):
        sampler = EmbeddingComposite(self.qpu)

        # in the future it would be good to test a wide variety of BQMs,
        # see https://github.com/dwavesystems/dimod/issues/671
        # but for now let's just test a few
        bqm0 = dimod.BinaryQuadraticModel.from_ising({'a': 2.0, 'b': -2.0},
                                                     {('a', 'b'): -1})
        bqm1 = dimod.BinaryQuadraticModel.from_ising({2: 4},
                                                     {(0, 1): 1.5, (1, 2): 5})

        samplesets0 = []
        samplesets1 = []
        for _ in range(10):
            # this should be async
            samplesets0.append(sampler.sample(bqm0))
            samplesets1.append(sampler.sample(bqm1))

        if all(ss.done() for ss in samplesets0):
            warnings.warn("Sampler calls appear to be synchronous")

        for ss0, ss1 in zip(samplesets0, samplesets1):
            dimod.testing.assert_sampleset_energies(ss0, bqm0)
            dimod.testing.assert_sampleset_energies(ss1, bqm1)

        self.assertTrue(all(ss.done() for ss in samplesets0))
        self.assertTrue(all(ss.done() for ss in samplesets1))
