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
#
# =============================================================================
import unittest

import dimod

from dwave.cloud.exceptions import ConfigFileError

from dwave.system import DWaveSampler, EmbeddingComposite


class TestEmbeddingCompositeExactSolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.qpu = DWaveSampler(solver=dict(qpu=True, initial_state=True))
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
