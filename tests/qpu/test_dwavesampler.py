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


class TestDWaveSampler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.qpu = DWaveSampler(solver=dict(qpu=True))
        except (ValueError, ConfigFileError):
            raise unittest.SkipTest("no qpu available")

    @classmethod
    def tearDownClass(cls):
        cls.qpu.client.close()

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


class TestMissingQubits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            # get a QPU with less than 100% yield
            cls.qpu = DWaveSampler(solver=dict(qpu=True,
                                               num_active_qubits__lt=2048))
        except (ValueError, ConfigFileError):
            raise unittest.SkipTest("no qpu available")

    @classmethod
    def tearDownClass(cls):
        cls.qpu.client.close()

    def test_sample_ising_h_list(self):
        sampler = self.qpu

        h = [0 for _ in range(2048)]
        J = {edge: 0 for edge in sampler.edgelist}

        sampleset = sampler.sample_ising(h, J)

        self.assertEqual(set(sampleset.variables), set(sampler.nodelist))
        assert len(sampleset.variables) < 2048  # sanity check
