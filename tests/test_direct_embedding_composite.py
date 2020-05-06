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
#
# ================================================================================================
import unittest
from collections import Mapping

import dimod
import dimod.testing as dtest
import dwave_networkx as dnx
from dimod.exceptions import BinaryQuadraticModelStructureError

from dwave.system.testing import MockDWaveSampler
from dwave.system.composites import DirectChimeraTilesEmbeddingComposite


class TestDirect(unittest.TestCase):

    def test_instantiation_smoketest(self):
        sampler = DirectChimeraTilesEmbeddingComposite(MockDWaveSampler())

        dtest.assert_sampler_api(sampler)

    def test_sample_ising(self):
        mock_sampler_chimera = MockDWaveSampler()  # C4 structured sampler

        sampler = DirectChimeraTilesEmbeddingComposite(mock_sampler_chimera)

        h = {0: 1.0, 1: -1.0, 2: -1.0, 3: 1.0, 4: 1.0, 5: -1.0, 6: 0.0, 7: 1.0,
             8: 1.0, 9: -1.0, 10: -1.0, 11: 1.0, 12: 1.0, 13: 0.0, 14: -1.0,
             15: 1.0}
        J = {(9, 13): -1, (2, 6): -1, (8, 13): -1, (9, 14): -1, (9, 15): -1,
             (10, 13): -1, (5, 13): -1, (10, 12): -1, (1, 5): -1, (10, 14): -1,
             (0, 5): -1, (1, 6): -1, (3, 6): -1, (1, 7): -1, (11, 14): -1,
             (2, 5): -1, (2, 4): -1, (6, 14): -1}

        sampleset = sampler.sample_ising(h, J)

        self.assertGreaterEqual(len(sampleset), 1)

        for sample in sampleset.samples():
            self.assertIsInstance(sample, Mapping)
            self.assertEqual(set(sample), set(h))

        for sample, energy in sampleset.data(['sample', 'energy']):
            self.assertIsInstance(sample, Mapping)
            self.assertEqual(set(sample), set(h))
            self.assertAlmostEqual(dimod.ising_energy(sample, h, J),
                                   energy)

    def test_sample_ising_not_index_inputs(self):
        sampler = DirectChimeraTilesEmbeddingComposite(MockDWaveSampler())

        h = {'a': -1., 'b': 2}
        J = {('a', 'b'): 1.5}

        with self.assertRaises(BinaryQuadraticModelStructureError):
            sampler.sample_ising(h, J)

    def test_sample_qubo(self):
        sampler = DirectChimeraTilesEmbeddingComposite(MockDWaveSampler())

        Q = {(0, 0): .1, (0, 4): -.8, (4, 4): 1}

        sampleset = sampler.sample_qubo(Q)

        self.assertGreaterEqual(len(sampleset), 1)

        for sample in sampleset.samples():
            for u, v in Q:
                self.assertIn(v, sample)
                self.assertIn(u, sample)

        for sample, energy in sampleset.data(['sample', 'energy']):
            self.assertAlmostEqual(dimod.qubo_energy(sample, Q),
                                   energy)

    def test_singleton_variables(self):
        sampler = DirectChimeraTilesEmbeddingComposite(MockDWaveSampler())

        h = {0: -1., 4: 2}
        J = {}

        with self.assertRaises(BinaryQuadraticModelStructureError):
            sampleset = sampler.sample_ising(h, J)

    def test_embedding_chiera_first_cell(self):
        sampler = DirectChimeraTilesEmbeddingComposite(MockDWaveSampler())

        h = {}
        J = {(0, 4): 1, (4, 1): 1, (1, 6): 1}

        sampleset = sampler.sample_ising(h, J, return_embedding=True)

        self.assertEqual(sampleset.info["embedding_context"]["embedding"],
             {0: [0], 1: [1], 4: [4], 6: [6]})

    def test_embedding_chiera_first_two_cells(self):
        sampler = DirectChimeraTilesEmbeddingComposite(MockDWaveSampler())

        h = {}
        J = {(0, 4): 1, (4, 1): 1, (1, 6): 1, (6, 14): -1, (14, 11): 1,
             (11, 15): 1}

        sampleset = sampler.sample_ising(h, J, return_embedding=True)

        self.assertEqual(sampleset.info["embedding_context"]["embedding"],
             {0: [0], 1: [1], 4: [4], 6: [6], 11: [11], 14: [14], 15: [15]})
