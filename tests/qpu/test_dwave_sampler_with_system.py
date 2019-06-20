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

from dwave.system.samplers import DWaveSampler


class TestDWaveSamplerSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.qpu = DWaveSampler(solver=dict(qpu=True))
        except (ValueError, ConfigFileError):
            cls.qpu = False

    @classmethod
    def tearDownClass(cls):
        if cls.qpu:
            cls.qpu.client.close()

    def setUp(self):
        if not self.qpu:
            self.skipTest("no qpu available")

    def test_smoke_sample_ising(self):
        sampler = self.qpu

        h = {v: 0 for v in sampler.nodelist}
        J = {interaction: 0 for interaction in sampler.edgelist}

        sampleset = sampler.sample_ising(h, J)
        sampleset.resolve()

    def test_smoke_sample_qubo(self):
        sampler = self.qpu

        Q = {interaction: 0 for interaction in sampler.edgelist}

        sampleset = sampler.sample_qubo(Q)
        sampleset.resolve()

    def test_mismatched_ising(self):
        sampler = self.qpu

        h = {len(sampler.nodelist)*100: 1}  # get a qubit we know isn't there
        J = {}

        with self.assertRaises(dimod.exceptions.BinaryQuadraticModelStructureError):
            sampler.sample_ising(h, J).resolve()

    def test_mismatched_qubo(self):
        sampler = self.qpu

        v = len(sampler.nodelist)*100
        Q = {(v, v): -1}

        with self.assertRaises(dimod.exceptions.BinaryQuadraticModelStructureError):
            sampler.sample_qubo(Q).resolve()
