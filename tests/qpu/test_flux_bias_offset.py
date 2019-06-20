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

import minorminer

from dwave.cloud.exceptions import ConfigFileError

import dwave.system.flux_bias_offsets as fb


from dwave.system.samplers import DWaveSampler


class TestGetFluxBiases(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.qpu = DWaveSampler(solver=dict(qpu=True, flux_biases=True))
        except (ValueError, ConfigFileError):
            raise unittest.SkipTest("no qpu available")

    @classmethod
    def tearDownClass(cls):
        cls.qpu.client.close()

    def step1(self, sampler, embedding):
        """get new flux bias"""
        return fb.get_flux_biases(sampler, embedding, chain_strength=1, max_age=0)

    def step2(self, sampler, embedding):
        """get flux biases from cache"""
        return fb.get_flux_biases(sampler, embedding, chain_strength=1)

    def test_simple(self):
        sampler = self.qpu

        embedding = minorminer.find_embedding([[0, 1], [1, 2], [0, 2]], sampler.edgelist)

        fbo1 = self.step1(sampler, embedding)
        fbo2 = self.step2(sampler, embedding)

        self.assertEqual(fbo1, fbo2)
