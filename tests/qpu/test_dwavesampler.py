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

import threading
import unittest
import os
from unittest import mock

import numpy

import dimod
from dwave.cloud.exceptions import ConfigFileError, SolverNotFoundError
from dwave.cloud.client import Client

from dwave.system.samplers import DWaveSampler


@unittest.skipIf(os.getenv('SKIP_INT_TESTS'), "Skipping integration test.")
class TestDWaveSampler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.qpu = DWaveSampler()
        except (ValueError, ConfigFileError, SolverNotFoundError):
            raise unittest.SkipTest("no qpu available")

    @classmethod
    def tearDownClass(cls):
        cls.qpu.client.close()

    def test_close(self):
        n_initial = threading.active_count()
        sampler = DWaveSampler()
        n_active = threading.active_count()
        sampler.close()
        n_closed = threading.active_count()

        with self.subTest('verify all client threads shutdown'):
            self.assertGreater(n_active, n_initial)
            self.assertEqual(n_closed, n_initial)

        try:
            # requires `dwave-cloud-client>=0.13.3`
            from dwave.cloud.exceptions import UseAfterCloseError
        except:
            pass
        else:
            with self.subTest('verify use after close disallowed'):
                with self.assertRaises(UseAfterCloseError):
                    h, J = self.nonzero_ising_problem(sampler)
                    sampler.sample_ising(h, J)

        with self.subTest('verify context manager calls close'):
            n_initial = threading.active_count()
            with DWaveSampler():
                n_active = threading.active_count()
            n_closed = threading.active_count()

            self.assertGreater(n_active, n_initial)
            self.assertEqual(n_closed, n_initial)

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

    def test_problem_labelling(self):
        sampler = self.qpu
        h, J = self.nonzero_ising_problem(sampler)
        label = 'problem label'

        # label set
        sampleset = sampler.sample_ising(h, J, label=label)
        self.assertIn('problem_label', sampleset.info)
        self.assertEqual(sampleset.info['problem_label'], label)

        # label unset
        sampleset = sampler.sample_ising(h, J)
        self.assertNotIn('problem_label', sampleset.info)


@unittest.skipIf(os.getenv('SKIP_INT_TESTS'), "Skipping integration test.")
class TestMissingQubits(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            # select a QPU solver with the smallest yield
            cls.qpu = DWaveSampler(solver=dict(
                order_by=lambda solver: solver.num_active_qubits / solver.num_qubits
            ))

            # double-check the yield is below 100%
            if cls.qpu.solver.num_active_qubits >= cls.qpu.solver.num_qubits:
                raise unittest.SkipTest("no qpu with less than 100% yield found")

        except (ValueError, ConfigFileError, SolverNotFoundError):
            raise unittest.SkipTest("no qpu available")

    @classmethod
    def tearDownClass(cls):
        cls.qpu.client.close()

    def test_sample_ising_h_list(self):
        sampler = self.qpu

        h = [0 for _ in range(self.qpu.solver.num_qubits)]
        J = {edge: 0 for edge in sampler.edgelist}

        sampleset = sampler.sample_ising(h, J)

        self.assertEqual(set(sampleset.variables), set(sampler.nodelist))
        self.assertLessEqual(len(sampleset.variables), self.qpu.solver.num_qubits)  # sanity check


@unittest.skipIf(os.getenv('SKIP_INT_TESTS'), "Skipping integration test.")
class TestClientSelection(unittest.TestCase):

    def test_client_type(self):
        with mock.patch('dwave.cloud.client.qpu.Client') as qpu:
            self.assertEqual(DWaveSampler().client, qpu())
            self.assertEqual(DWaveSampler(client='qpu').client, qpu())

        with mock.patch('dwave.cloud.client.sw.Client') as sw:
            self.assertEqual(DWaveSampler(client='sw').client, sw())

        with mock.patch('dwave.cloud.client.hybrid.Client') as hybrid:
            self.assertEqual(DWaveSampler(client='hybrid').client, hybrid())

    def test_base_client(self):
        # to test 'base' client instantiation offline,
        # we would need a mock client and a mock solver
        try:
            self.assertEqual(type(DWaveSampler(client=None).client), Client)
            self.assertEqual(type(DWaveSampler(client='base').client), Client)
        except (ValueError, ConfigFileError):
            raise unittest.SkipTest("no API token available")
