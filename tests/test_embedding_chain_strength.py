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

import unittest

import networkx as nx

import dimod
import dwave.embedding
from dwave.embedding.chain_strength import uniform_torque_compensation, scaled

class TestUniformTorqueCompensation(unittest.TestCase):
    def setUp(self):
        h = {0: 0, 1: 0, 2: 0}
        J = {(0, 1): 1, (1, 2): 1, (0, 2): 1}
        self.bqm = dimod.BinaryQuadraticModel.from_ising(h, J)

    def test_empty(self):
        empty_bqm = dimod.BinaryQuadraticModel({}, {}, 0, 'SPIN')
        chain_strength = uniform_torque_compensation(empty_bqm, prefactor=2)
        self.assertEqual(chain_strength, 1)

    def test_default_prefactor(self):
        chain_strength = uniform_torque_compensation(self.bqm)
        self.assertAlmostEqual(chain_strength, 1.9997, places=4)

    def test_typical(self):
        chain_strength = uniform_torque_compensation(self.bqm, prefactor=2)
        self.assertAlmostEqual(chain_strength, 2.8284, places=4)

    def test_as_callable(self):
        embedding = {0: {0}, 1: {1}, 2: {2, 3}}
        embedded_bqm = dwave.embedding.embed_bqm(self.bqm, embedding, nx.cycle_graph(4), 
                                                 chain_strength=uniform_torque_compensation)

        expected_bqm = dimod.BinaryQuadraticModel({0: 0, 1: 0, 2: 0, 3: 0},
                                                  {(0, 1): 1, (1, 2): 1, (2, 3): -1.9997, (0, 3): 1},
                                                  1.9997,  # offset the energy from satisfying chains
                                                  dimod.SPIN)

        dimod.testing.assert_bqm_almost_equal(embedded_bqm, expected_bqm, places=4) 

class TestScaled(unittest.TestCase):
    def setUp(self):
        h = {0: 0, 1: -1, 2: 2}
        J = {(0, 1): 1, (1, 2): -2, (0, 2): -1}
        self.bqm = dimod.BinaryQuadraticModel.from_ising(h, J)

    def test_empty(self):
        empty_bqm = dimod.BinaryQuadraticModel({}, {}, 0, 'SPIN')
        chain_strength = scaled(empty_bqm, prefactor=2)
        self.assertEqual(chain_strength, 1)

    def test_default_prefactor(self):
        chain_strength = scaled(self.bqm)
        self.assertEqual(chain_strength, 2)

    def test_typical(self):
        chain_strength = scaled(self.bqm, prefactor=1.5)
        self.assertEqual(chain_strength, 3)

    def test_as_callable(self):
        embedding = {0: {0}, 1: {1}, 2: {2, 3}}
        embedded_bqm = dwave.embedding.embed_bqm(self.bqm, embedding, nx.cycle_graph(4), 
                                                 chain_strength=scaled)

        expected_bqm = dimod.BinaryQuadraticModel({0: 0, 1: -1, 2: 1, 3: 1},
                                                  {(0, 1): 1, (1, 2): -2, (2, 3): -2, (0, 3): -1},
                                                  2,  # offset the energy from satisfying chains
                                                  dimod.SPIN)
        self.assertEqual(embedded_bqm, expected_bqm)
