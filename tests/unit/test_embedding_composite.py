import unittest

from collections import Mapping

import dimod
import dimod.testing as dtest

from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite

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

    def test_instantiation_triangle(self):
        sampler = FixedEmbeddingComposite(MockSampler(), {'a': [0, 4], 'b': [1, 5], 'c': [2, 6]})

        dtest.assert_sampler_api(sampler)  # checks adj consistent with nodelist/edgelist

        self.assertEqual(sampler.nodelist, ['a', 'b', 'c'])
        self.assertEqual(sampler.edgelist, [('a', 'b'), ('a', 'c'), ('b', 'c')])

    def test_sample_bqm_triangle(self):
        sampler = FixedEmbeddingComposite(MockSampler(), {'a': [0, 4], 'b': [1, 5], 'c': [2, 6]})

        resp = sampler.sample_ising({'a': 1, 'b': 1, 'c': 0}, {})

        self.assertEqual(set(resp.variable_labels), {'a', 'b', 'c'})
