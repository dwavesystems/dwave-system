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

import unittest

from contextlib import contextmanager

import dimod
import numpy as np

import dwave.embedding


class TestBrokenChains(unittest.TestCase):
    def test_broken_chains_typical(self):
        S = np.array([[-1, 1, -1, 1],
                      [1, 1, -1, -1],
                      [-1, 1, -1, -1]])
        chains = [[0, 1], [2, 3]]

        broken = dwave.embedding.broken_chains(S, chains)

        np.testing.assert_array_equal([[1, 1], [0, 0], [1, 0]], broken)

    def test_broken_chains_chains_length_0(self):
        S = np.array([[-1, 1, -1, 1],
                      [1, 1, -1, -1],
                      [-1, 1, -1, -1]])
        chains = [[0, 1], [], [2, 3]]

        broken = dwave.embedding.broken_chains(S, chains)

        np.testing.assert_array_equal([[1, 0, 1], [0, 0, 0], [1, 0, 0]], broken)

    def test_broken_chains_single_sample(self):
        S = [-1, 1, 1, 1]
        chains = [[0, 1], [2, 3]]

        broken = dwave.embedding.broken_chains(S, chains)

        np.testing.assert_array_equal([[True, False]], broken)

    def test_matrix(self):
        samples_matrix = np.array([[-1, +1, -1, +1],
                                   [+1, +1, +1, +1],
                                   [-1, -1, +1, -1],
                                   [-1, -1, +1, +1]], dtype='int8')
        chain_list = [(0, 1), (2, 3)]

        broken = dwave.embedding.broken_chains(samples_matrix, chain_list)


class ChainBreakResolutionAPI():
    chains = [[0, 1], [2, 4], [3]]  # needs to be available to MinimizeEnergy

    def test_api(self):
        cases = {}
        cases['ndarray_binary'] = np.triu(np.ones((5, 5)))
        cases['ndarray_spin'] = 2*np.triu(np.ones((5, 5))) - 1
        cases['listoflists_binary'] = [[1, 1, 1, 1, 1],
                                       [0, 1, 1, 1, 1],
                                       [0, 0, 1, 1, 1],
                                       [0, 0, 0, 1, 1],
                                       [0, 0, 0, 0, 1]]

        ss0 = dimod.SampleSet.from_samples(np.triu(np.ones((5, 5))),
                                           energy=0,
                                           vartype=dimod.BINARY)
        cases['sampleset_ordered'] = ss0

        ss1 = dimod.SampleSet.from_samples((np.triu(np.ones((5, 5))),
                                            [0, 4, 1, 3, 2]),
                                           energy=0,
                                           vartype=dimod.BINARY)
        cases['sampleset_unordered'] = ss1

        for description, samples in cases.items():
            with self.subTest(description):
                resolved, idxs = self.chain_break_method(samples, self.chains)

                # correct number of variables (1 per chain)
                self.assertEqual(resolved.shape[1], len(self.chains))

                # correct number of rows
                self.assertEqual(resolved.shape[0], len(idxs))


class TestDiscard(ChainBreakResolutionAPI, unittest.TestCase, ):

    # for the API tests
    def setUp(self):
        self.chain_break_method = dwave.embedding.discard

    def test_discard_no_breaks_all_ones_identity_embedding(self):

        samples_matrix = np.array(np.ones((100, 50)), dtype='int8')
        chain_list = [[idx] for idx in range(50)]

        new_matrix, idxs = dwave.embedding.discard(samples_matrix, chain_list)

        np.testing.assert_equal(new_matrix, samples_matrix)

    def test_discard_no_breaks_all_ones_one_var_embedding(self):

        samples_matrix = np.array(np.ones((100, 50)), dtype='int8')
        chain_list = [[idx for idx in range(50)]]

        new_matrix, idxs = dwave.embedding.discard(samples_matrix, chain_list)

        self.assertEqual(new_matrix.shape, (100, 1))

    def test_discard_typical(self):

        samples_matrix = np.array([[-1, +1, -1, +1],
                                   [+1, +1, +1, +1],
                                   [-1, -1, +1, -1],
                                   [-1, -1, +1, +1]], dtype='int8')
        chain_list = [(0, 1), (2, 3)]

        new_matrix, idxs = dwave.embedding.discard(samples_matrix, chain_list)

        np.testing.assert_equal(new_matrix, [[+1, +1],
                                             [-1, +1]])

    def test_mixed_chain_types(self):
        chains = [(0, 1), [2, 3], {4, 5}]
        samples = [[1, 1, 1, 1, 1, 1], [0, 1, 0, 1, 0, 1]]
        unembedded, idx = dwave.embedding.discard(samples, chains)

        np.testing.assert_array_equal(unembedded, [[1, 1, 1]])
        np.testing.assert_array_equal(idx, [0])


class TestMajorityVote(ChainBreakResolutionAPI, unittest.TestCase):

    # for the API tests
    def setUp(self):
        self.chain_break_method = dwave.embedding.majority_vote

    def test_typical_spin(self):
        S = np.array([[-1, +1, -1, +1],
                      [+1, +1, -1, +1],
                      [-1, +1, -1, -1]])
        chains = [[0, 1, 2], [3]]

        samples, idx = dwave.embedding.majority_vote(S, chains)

        np.testing.assert_equal(samples, [[-1, +1],
                                          [+1, +1],
                                          [-1, -1]])

    def test_typical_binary(self):
        S = np.array([[0, 1, 0, 1],
                      [1, 1, 0, 1],
                      [0, 1, 0, 0]])
        chains = [[0, 1, 2], [3]]

        samples, idx = dwave.embedding.majority_vote(S, chains)

        np.testing.assert_equal(samples, [[0, 1],
                                          [1, 1],
                                          [0, 0]])

    def test_four_chains(self):
        S = [[-1, -1, -1, -1],
             [+1, -1, -1, -1],
             [+1, +1, -1, -1],
             [-1, +1, -1, -1],
             [-1, +1, +1, -1],
             [+1, +1, +1, -1],
             [+1, -1, +1, -1],
             [-1, -1, +1, -1],
             [-1, -1, +1, +1],
             [+1, -1, +1, +1],
             [+1, +1, +1, +1],
             [-1, +1, +1, +1],
             [-1, +1, -1, +1],
             [+1, +1, -1, +1],
             [+1, -1, -1, +1],
             [-1, -1, -1, +1]]
        chains = [[0], [1], [2, 3]]

        samples, idx = dwave.embedding.majority_vote(S, chains)

        self.assertEqual(samples.shape, (16, 3))
        self.assertEqual(set().union(*samples), {-1, 1})  # should be spin-valued


class TestMinimizeEnergy(ChainBreakResolutionAPI, unittest.TestCase):
    # for the API tests
    def setUp(self):
        embedding = dict(zip('abcdefghijk', self.chains))
        bqm = dimod.BinaryQuadraticModel.from_ising({v: 0 for v in embedding}, {})
        self.chain_break_method = dwave.embedding.MinimizeEnergy(bqm, embedding)

    def test_minimize_energy(self):
        embedding = {0: (0, 5), 1: (1, 6), 2: (2, 7), 3: (3, 8), 4: (4, 10)}
        h = []
        j = {(0, 1): -1, (0, 2): 2, (0, 3): 2, (0, 4): -1,
             (2, 1): -1, (1, 3): 2, (3, 1): -1, (1, 4): -1,
             (2, 3): 1, (4, 2): -1, (2, 4): -1, (3, 4): 1}

        bqm = dimod.BinaryQuadraticModel.from_ising(h, j)

        solutions = [
            [-1, -1, -1, -1, -1, -1, +1, +1, +1, 3, +1],
            [+1, +1, +1, +1, +1, -1, +1, -1, -1, 3, -1],
            [+1, +1, -1, +1, -1, -1, -1, -1, -1, 3, -1]
        ]
        expected = [
            [-1, -1, +1, +1, -1],
            [+1, +1, +1, -1, +1],
            [-1, -1, -1, +1, -1]
        ]

        cbm = dwave.embedding.MinimizeEnergy(bqm, embedding)

        unembedded, idx = cbm(solutions, [embedding[v] for v in range(5)])

        np.testing.assert_array_equal(expected, unembedded)

    def test_minimize_energy_non_clique(self):
        embedding = {0: (0, 5), 1: (1, 6), 2: (2, 7), 3: (3, 8), 4: (4, 10)}
        h = []
        j = {(0, 1): -1, (0, 2): 2, (0, 3): 2, (0, 4): -1,
             (1, 3): 2, (3, 1): -1, (1, 4): -1,
             (2, 3): 1, (4, 2): -1, (2, 4): -1, (3, 4): 1}

        bqm = dimod.BinaryQuadraticModel.from_ising(h, j)

        solutions = [
            [-1, -1, -1, -1, -1, -1, +1, +1, +1, 3, +1],
            [+1, +1, +1, +1, +1, -1, +1, -1, -1, 3, -1],
            [+1, +1, -1, +1, -1, -1, -1, -1, -1, 3, -1]
        ]

        expected = [
            [-1, -1, +1, +1, -1],
            [+1, +1, +1, -1, +1],
            [-1, -1, -1, +1, -1]
        ]

        cbm = dwave.embedding.MinimizeEnergy(bqm, embedding)
        unembedded, idx = cbm(solutions, [embedding[v] for v in range(5)])

        np.testing.assert_array_equal(expected, unembedded)

    def test_minimize_energy_easy(self):
        chains = ({0, 1}, [2], (4, 5, 6))
        embedding = {v: chain for v, chain in enumerate(chains)}
        h = [-1, 0, 0]
        j = {}
        bqm = dimod.BinaryQuadraticModel.from_ising(h, j)
        solutions = [
            [-1, -1, +1, 3, -1, -1, -1],
            [-1, +1, -1, 3, +1, +1, +1]
        ]
        expected = [
            [-1, +1, -1],
            [+1, -1, +1]
        ]
        cbm = dwave.embedding.MinimizeEnergy(bqm, embedding)

        unembedded, idx = cbm(solutions, chains)

        np.testing.assert_array_equal(expected, unembedded)

    def test_empty_matrix(self):
        chains = []
        bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
        solutions = [[]]
        embedding = {}
        cbm = dwave.embedding.MinimizeEnergy(bqm, embedding)
        unembedded, idx = cbm(solutions, chains)

        np.testing.assert_array_equal([[]], unembedded)
        np.testing.assert_array_equal(idx, [0])

    def test_empty_chains(self):
        embedding = {}
        h = []
        j = {(0, 1): -1, (0, 2): 2, (0, 3): 2, (0, 4): -1,
             (2, 1): -1, (1, 3): 2, (3, 1): -1, (1, 4): -1,
             (2, 3): 1, (4, 2): -1, (2, 4): -1, (3, 4): 1}

        bqm = dimod.BinaryQuadraticModel.from_ising(h, j)

        solutions = [
            [-1, -1, -1, -1, -1, -1, +1, +1, +1, 3, +1],
            [+1, +1, +1, +1, +1, -1, +1, -1, -1, 3, -1],
            [+1, +1, -1, +1, -1, -1, -1, -1, -1, 3, -1]
        ]
        expected = [
            [-1, -1, +1, +1, -1],
            [+1, +1, +1, -1, +1],
            [-1, -1, -1, +1, -1]
        ]

        cbm = dwave.embedding.MinimizeEnergy(bqm, embedding)

        unembedded, idx = cbm(solutions, [])

        np.testing.assert_array_equal(unembedded, [[], [], []])

    def test_unordered_labels(self):
        Q = {(0, 1): 1, (0, 2): 1, (0, 3): 1, (1, 2): 1, (1, 3): 1, (2, 3): 1,
             (0, 0): 1, (1, 1): 1, (2, 2): 1, (3, 3): 1}
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

        embedding = {0: [55], 1: [48], 2: [50, 53], 3: [52, 51]}

        samples = (np.triu(np.ones(6)), [48, 50, 51, 52, 53, 55])
        sampleset = dimod.SampleSet.from_samples(samples, energy=0,
                                                 vartype=dimod.BINARY)

        cbm = dwave.embedding.MinimizeEnergy(bqm, embedding)

        unembedded, idx = cbm(sampleset, [[55], [48], [50, 53], [52, 51]])


class TestWeightedRandom(ChainBreakResolutionAPI, unittest.TestCase):
    # for the API tests
    def setUp(self):
        self.chain_break_method = dwave.embedding.weighted_random
