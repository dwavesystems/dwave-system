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
# ================================================================================================
import unittest

from collections import Mapping

import dimod
import dimod.testing as dtest

from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite, LazyEmbeddingComposite
import dwavebinarycsp as dbc
from dwavebinarycsp.factories.constraint.gates import and_gate, or_gate

from tests.unit.mock_sampler import MockSampler


class TestEmbeddingComposite(unittest.TestCase):
    def test_instantiation_smoketest(self):
        sampler = EmbeddingComposite(MockSampler())

        dtest.assert_sampler_api(sampler)

    def test_sample_ising(self):
        sampler = EmbeddingComposite(MockSampler())

        h = {0: -1., 4: 2}
        J = {(0, 4): 1.5}

        response = sampler.sample_ising(h, J)

        # nothing failed and we got at least one response back
        self.assertGreaterEqual(len(response), 1)

        for sample in response.samples():
            self.assertIsInstance(sample, Mapping)
            self.assertEqual(set(sample), set(h))

        for sample, energy in response.data(['sample', 'energy']):
            self.assertIsInstance(sample, Mapping)
            self.assertEqual(set(sample), set(h))
            self.assertAlmostEqual(dimod.ising_energy(sample, h, J),
                                   energy)

    def test_sample_ising_unstructured_not_integer_labelled(self):
        sampler = EmbeddingComposite(MockSampler())

        h = {'a': -1., 'b': 2}
        J = {('a', 'b'): 1.5}

        response = sampler.sample_ising(h, J)

        # nothing failed and we got at least one response back
        self.assertGreaterEqual(len(response), 1)

        for sample in response.samples():
            for v in h:
                self.assertIn(v, sample)

        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(dimod.ising_energy(sample, h, J),
                                   energy)

    def test_sample_qubo(self):
        sampler = EmbeddingComposite(MockSampler())

        Q = {(0, 0): .1, (0, 4): -.8, (4, 4): 1}

        response = sampler.sample_qubo(Q)

        # nothing failed and we got at least one response back
        self.assertGreaterEqual(len(response), 1)

        for sample in response.samples():
            for u, v in Q:
                self.assertIn(v, sample)
                self.assertIn(u, sample)

        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(dimod.qubo_energy(sample, Q),
                                   energy)

    def test_max_cut(self):
        sampler = EmbeddingComposite(MockSampler())

        m = 2
        n = 2
        t = 2

        hoff = 2 * t
        voff = n * hoff
        mi = m * voff
        ni = n * hoff

        edges = []

        # tile edges
        edges.extend((k0, k1)
                     for i in range(0, ni, hoff)
                     for j in range(i, mi, voff)
                     for k0 in range(j, j + t)
                     for k1 in range(j + t, j + 2 * t))
        # horizontal edges
        edges.extend((k, k + hoff)
                     for i in range(t, 2 * t)
                     for j in range(i, ni - hoff, hoff)
                     for k in range(j, mi, voff))
        # vertical edges
        edges.extend((k, k + voff)
                     for i in range(t)
                     for j in range(i, ni, hoff)
                     for k in range(j, mi - voff, voff))

        J = {edge: 1 for edge in edges}
        h = {v: 0 for v in set().union(*J)}

        response = sampler.sample_ising(h, J)

    def test_singleton_variables(self):
        sampler = EmbeddingComposite(MockSampler())

        h = {0: -1., 4: 2}
        J = {}

        response = sampler.sample_ising(h, J)

        # nothing failed and we got at least one response back
        self.assertGreaterEqual(len(response), 1)


class TestFixedEmbeddingComposite(unittest.TestCase):
    def test_instantiation_empty(self):
        sampler = FixedEmbeddingComposite(MockSampler(), {})

        dtest.assert_sampler_api(sampler)  # checks adj consistent with nodelist/edgelist

        self.assertEqual(sampler.edgelist, [])

        self.assertTrue(hasattr(sampler, 'embedding'))
        self.assertIn('embedding', sampler.properties)

    def test_instantiation_triangle(self):
        embedding = {'a': [0, 4], 'b': [1, 5], 'c': [2, 6]}
        sampler = FixedEmbeddingComposite(MockSampler(), embedding)
        self.assertEqual(embedding, sampler.embedding)

        dtest.assert_sampler_api(sampler)  # checks adj consistent with nodelist/edgelist

        self.assertEqual(sampler.nodelist, ['a', 'b', 'c'])
        self.assertEqual(sampler.edgelist, [('a', 'b'), ('a', 'c'), ('b', 'c')])

    def test_sample_bqm_triangle(self):
        sampler = FixedEmbeddingComposite(MockSampler(), {'a': [0, 4], 'b': [1, 5], 'c': [2, 6]})

        resp = sampler.sample_ising({'a': 1, 'b': 1, 'c': 0}, {})

        self.assertEqual(set(resp.variable_labels), {'a', 'b', 'c'})


class TestLazyEmbeddingComposite(unittest.TestCase):
    def test_sample_instantiation(self):
        # Check that values have not been instantiated
        sampler = LazyEmbeddingComposite(MockSampler())
        self.assertIsNone(sampler.embedding)
        self.assertIsNone(sampler.nodelist)
        self.assertIsNone(sampler.edgelist)
        self.assertIsNone(sampler.adjacency)
        self.assertIsNone(sampler.parameters)
        self.assertIsNone(sampler.properties)

        # Set up BQM and sample
        csp = dbc.ConstraintSatisfactionProblem(dbc.BINARY)
        csp.add_constraint(and_gate(['a', 'b', 'c']))
        bqm = dbc.stitch(csp)
        sampler.sample(bqm)

        # Check that values have been populated
        self.assertIsNotNone(sampler.embedding)
        self.assertEqual(sampler.nodelist, ['a', 'b', 'c'])
        self.assertEqual(sampler.edgelist, [('a', 'b'), ('a', 'c'), ('b', 'c')])
        self.assertEqual(sampler.adjacency, {'a': {'b', 'c'}, 'b': {'a', 'c'}, 'c': {'a', 'b'}})
        self.assertIsNotNone(sampler.parameters)
        self.assertIsNotNone(sampler.properties)

    def test_same_embedding(self):
        sampler = LazyEmbeddingComposite(MockSampler())

        # Set up Ising and sample
        h = {'a': 1, 'b': 1, 'c': 1}
        J = {('a', 'b'): 3, ('b', 'c'): -2, ('a', 'c'): 1}
        sampler.sample_ising(h, J)

        # Store embedding
        prev_embedding = sampler.embedding

        # Check that the same embedding is used
        csp2 = dbc.ConstraintSatisfactionProblem(dbc.BINARY)
        csp2.add_constraint(or_gate(['a', 'b', 'c']))   #TODO: Same naming must be used. Weird
        bqm2 = dbc.stitch(csp2)
        sampler.sample(bqm2)

        self.assertEqual(sampler.embedding, prev_embedding)

    def test_ising(self):
        h = {0: 11, 5: 2}
        J = {(0, 5): -8}
        sampler = LazyEmbeddingComposite(MockSampler())
        response = sampler.sample_ising(h, J)

        # Check embedding
        self.assertIsNotNone(sampler.embedding)
        self.assertEqual(sampler.nodelist, [0, 5])
        self.assertEqual(sampler.edgelist, [(0, 5)])
        self.assertEqual(sampler.adjacency, {0: {5}, 5: {0}})

        # Check that at least one response was found
        self.assertGreaterEqual(len(response), 1)

    def test_qubo(self):
        Q = {(1, 1): 1, (2, 2): 2, (3, 3): 3, (1, 2): 4, (2, 3): 5, (1, 3): 6}
        sampler = LazyEmbeddingComposite(MockSampler())
        response = sampler.sample_qubo(Q)

        # Check embedding
        self.assertIsNotNone(sampler.embedding)
        self.assertEqual(sampler.nodelist, [1, 2, 3])
        self.assertEqual(sampler.edgelist, [(1, 2), (1, 3), (2, 3)])
        self.assertEqual(sampler.adjacency, {1: {2, 3}, 2: {1, 3}, 3: {1, 2}})

        # Check that at least one response was found
        self.assertGreaterEqual(len(response), 1)

    def test_sparse_qubo(self):
        # There is no relationship between nodes 2 and 3
        Q = {(1, 1): 1, (2, 2): 2, (3, 3): 3, (1, 2): 4, (1, 3): 6}
        sampler = LazyEmbeddingComposite(MockSampler())
        response = sampler.sample_qubo(Q)

        # Check embedding
        self.assertIsNotNone(sampler.embedding)
        self.assertEqual(sampler.nodelist, [1, 2, 3])
        self.assertEqual(sampler.edgelist, [(1, 2), (1, 3)])

        # Check that at least one response was found
        self.assertGreaterEqual(len(response), 1)
