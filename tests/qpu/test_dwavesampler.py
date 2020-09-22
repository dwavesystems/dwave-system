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
from unittest import mock

import numpy

import dimod
from dwave.cloud.exceptions import ConfigFileError
from dwave.cloud.client import Client

from dwave.system.samplers import DWaveSampler


class TestDWaveSampler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.qpu = DWaveSampler()
        except (ValueError, ConfigFileError):
            raise unittest.SkipTest("no qpu available")

    @classmethod
    def tearDownClass(cls):
        cls.qpu.client.close()

    @staticmethod
    def nonzero_ising_problem(sampler):
        max_h = max(sampler.properties.get('h_range', [-1, 1]))
        max_j = max(sampler.properties.get('j_range', [-1, 1]))

        h = {v: max_h for v in sampler.nodelist}
        J = {interaction: max_j for interaction in sampler.edgelist}

        return h, J

    def test_sample_ising(self):
        sampler = self.qpu

        h, J = self.nonzero_ising_problem(sampler)

        sampleset = sampler.sample_ising(h, J)

        bqm = dimod.BQM.from_ising(h, J)
        numpy.testing.assert_array_almost_equal(
            bqm.energies(sampleset), sampleset.record.energy)

    def test_sample_qubo(self):
        sampler = self.qpu

        Q, _ = dimod.ising_to_qubo(*self.nonzero_ising_problem(sampler))

        sampleset = sampler.sample_qubo(Q)

        bqm = dimod.BQM.from_qubo(Q)
        numpy.testing.assert_array_almost_equal(
            bqm.energies(sampleset), sampleset.record.energy)

    def test_sample_bqm_ising(self):
        sampler = self.qpu

        bqm = dimod.BQM.from_ising(*self.nonzero_ising_problem(sampler))

        sampleset = sampler.sample(bqm)

        numpy.testing.assert_array_almost_equal(
            bqm.energies(sampleset), sampleset.record.energy)

    def test_sample_bqm_qubo(self):
        sampler = self.qpu

        Q, _ = dimod.ising_to_qubo(*self.nonzero_ising_problem(sampler))
        bqm = dimod.BQM.from_qubo(Q)

        sampleset = sampler.sample(bqm)

        numpy.testing.assert_array_almost_equal(
            bqm.energies(sampleset), sampleset.record.energy)

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
            cls.qpu = DWaveSampler(solver=dict(num_active_qubits__lt=2048))
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


class TestClientSelection(unittest.TestCase):

    def test_client_type(self):
        with mock.patch('dwave.cloud.qpu.Client') as qpu:
            self.assertEqual(DWaveSampler().client, qpu())
            self.assertEqual(DWaveSampler(client='qpu').client, qpu())

        with mock.patch('dwave.cloud.sw.Client') as sw:
            self.assertEqual(DWaveSampler(client='sw').client, sw())

        with mock.patch('dwave.cloud.hybrid.Client') as hybrid:
            self.assertEqual(DWaveSampler(client='hybrid').client, hybrid())

    def test_base_client(self):
        # to test 'base' client instantiation offline,
        # we would need a mock client and a mock solver
        try:
            self.assertEqual(type(DWaveSampler(client=None).client), Client)
            self.assertEqual(type(DWaveSampler(client='base').client), Client)
        except (ValueError, ConfigFileError):
            raise unittest.SkipTest("no API token available")
