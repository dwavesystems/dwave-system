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

        sampler = DirectChimeraTilesEmbeddingComposite(MockDWaveSampler())

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

        sampler = DirectChimeraTilesEmbeddingComposite(MockDWaveSampler(
                  topology="pegasus"))

        h = {0: -1., 4: 2}
        J = {}

        with self.assertRaises(BinaryQuadraticModelStructureError):
            sampleset = sampler.sample_ising(h, J)

    def test_embedding_chimera_first_cell(self):
        sampler = DirectChimeraTilesEmbeddingComposite(MockDWaveSampler())

        h = {}
        J = {(0, 4): 1, (4, 1): 1, (1, 6): 1}

        sampleset = sampler.sample_ising(h, J, return_embedding=True)

        self.assertEqual(sampleset.info["embedding_context"]["embedding"],
             {0: [0], 1: [1], 4: [4], 6: [6]})

    def test_embedding_chimera_first_two_cells(self):
        sampler = DirectChimeraTilesEmbeddingComposite(MockDWaveSampler())

        h = {}
        J = {(0, 4): 1, (4, 1): 1, (1, 6): 1, (6, 14): -1, (14, 11): 1,
             (11, 15): 1}

        sampleset = sampler.sample_ising(h, J, return_embedding=True)

        self.assertEqual(sampleset.info["embedding_context"]["embedding"],
             {0: [0], 1: [1], 4: [4], 6: [6], 11: [11], 14: [14], 15: [15]})

    def test_embedding_chimera_first_cell_broken(self):
        sampler = DirectChimeraTilesEmbeddingComposite(MockDWaveSampler(
                  broken_nodes=[1]))

        h = {}
        J = {(0, 4): 1, (4, 1): 1, (1, 6): 1}

        sampleset = sampler.sample_ising(h, J, return_embedding=True)

        self.assertEqual(sampleset.info["embedding_context"]["embedding"],
             {0: [8], 1: [9], 4: [12], 6: [14]})

    def test_embedding_chimera_two_cells_second_broken(self):
        sampler = DirectChimeraTilesEmbeddingComposite(MockDWaveSampler(
                  broken_nodes=[14]))

        h = {}
        J = {(0, 4): 1, (4, 1): 1, (1, 6): 1, (6, 14): -1, (14, 11): 1,
             (11, 15): 1}

        sampleset = sampler.sample_ising(h, J, return_embedding=True)

        self.assertEqual(sampleset.info["embedding_context"]["embedding"],
             {0: [16], 1: [17], 4: [20], 6: [22], 11: [27], 14: [30], 15: [31]})

    def test_embedding_chimera_two_cells_broken_row(self):
        sampler = DirectChimeraTilesEmbeddingComposite(MockDWaveSampler(
                  broken_nodes=[x for x in range(0, 4*8, 8)]))  # C4 tester

        h = {}
        J = {(0, 4): 1, (4, 1): 1, (1, 6): 1, (6, 14): -1, (14, 11): 1,
             (11, 15): 1}

        sampleset = sampler.sample_ising(h, J, return_embedding=True)

        self.assertEqual(sampleset.info["embedding_context"]["embedding"],
             {0: [32], 1: [33], 4: [36], 6: [38], 11: [43], 14: [46], 15: [47]})

    # def test_embedding_pegasus_first_cell(self):
    #     sampler = DirectChimeraTilesEmbeddingComposite(MockDWaveSampler(
    #               topology="pegasus"))
    #
    #
    #     h = {}
    #     J = {(0, 4): 1, (4, 1): 1, (1, 6): 1}
    #
    #     sampleset = sampler.sample_ising(h, J, return_embedding=True)
    #
    #     self.assertEqual(sampleset.info["embedding_context"]["embedding"],
    #          {0: [0], 1: [1], 4: [4], 6: [6]})

    def test_embedding_pegasus_two_cell(self):
        sampler = DirectChimeraTilesEmbeddingComposite(MockDWaveSampler(
                  topology="pegasus"))

        Q = {(0, 0): 4.0, (1, 1): 4.0, (2, 2): 4.0, (3, 3): 4.0, (4, 4): 4.0,
             (5, 5): 6.0, (6, 6): 8.0, (7, 7): 4.0, (8, 8): 4.0,
             (9, 9): 4.0, (10, 10): 4.0, (11, 11): 4.0, (12, 12): 4.0,
             (13, 13): 8.0, (14, 14): 6.0, (15, 15): 4.0, (9, 13): -4.0,
             (2, 6): -4.0, (8, 13): -4.0,  (9, 14): -4.0, (9, 15): -4.0,
             (10, 13): -4.0, (5, 13): -4.0, (10, 12): -4.0, (1, 5): -4.0,
             (10, 14): -4.0, (0, 5): -4.0, (1, 6): -4.0, (3, 6): -4.0,
             (1, 7): -4.0, (11, 14): -4.0, (2, 5): -4.0, (2, 4): -4.0,
             (6, 14): -4.0}

        sampleset = sampler.sample_qubo(Q, return_embedding=True)

    # def test_embedding_chimera_first_two_cells(self):
    #     sampler = DirectChimeraTilesEmbeddingComposite(MockDWaveSampler())
    #
    #     h = {}
    #     J = {(0, 4): 1, (4, 1): 1, (1, 6): 1, (6, 14): -1, (14, 11): 1,
    #          (11, 15): 1}
    #
    #     sampleset = sampler.sample_ising(h, J, return_embedding=True)
    #
    #     self.assertEqual(sampleset.info["embedding_context"]["embedding"],
    #          {0: [0], 1: [1], 4: [4], 6: [6], 11: [11], 14: [14], 15: [15]})
    #
    # def test_embedding_chimera_first_cell_broken(self):
    #     sampler = DirectChimeraTilesEmbeddingComposite(MockDWaveSampler(
    #               broken_nodes=[1]))
    #
    #     h = {}
    #     J = {(0, 4): 1, (4, 1): 1, (1, 6): 1}
    #
    #     sampleset = sampler.sample_ising(h, J, return_embedding=True)
    #
    #     self.assertEqual(sampleset.info["embedding_context"]["embedding"],
    #          {0: [8], 1: [9], 4: [12], 6: [14]})
    #
    # def test_embedding_chimera_two_cells_second_broken(self):
    #     sampler = DirectChimeraTilesEmbeddingComposite(MockDWaveSampler(
    #               broken_nodes=[14]))
    #
    #     h = {}
    #     J = {(0, 4): 1, (4, 1): 1, (1, 6): 1, (6, 14): -1, (14, 11): 1,
    #          (11, 15): 1}
    #
    #     sampleset = sampler.sample_ising(h, J, return_embedding=True)
    #
    #     self.assertEqual(sampleset.info["embedding_context"]["embedding"],
    #          {0: [16], 1: [17], 4: [20], 6: [22], 11: [27], 14: [30], 15: [31]})
    #
    # def test_embedding_chimera_two_cells_broken_row(self):
    #     sampler = DirectChimeraTilesEmbeddingComposite(MockDWaveSampler(
    #               broken_nodes=[x for x in range(0, 4*8, 8)]))  # C4 tester
    #
    #     h = {}
    #     J = {(0, 4): 1, (4, 1): 1, (1, 6): 1, (6, 14): -1, (14, 11): 1,
    #          (11, 15): 1}
    #
    #     sampleset = sampler.sample_ising(h, J, return_embedding=True)
    #
    #     self.assertEqual(sampleset.info["embedding_context"]["embedding"],
    #          {0: [32], 1: [33], 4: [36], 6: [38], 11: [43], 14: [46], 15: [47]})
