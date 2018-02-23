import unittest

import dwave_micro_client as microclient
import dimod

try:
    # py3
    import unittest.mock as mock
except ImportError:
    # py2
    import mock

import dwave.system as micro

try:
    microclient.Connection()
    _sapi_connection = True
except (IOError, OSError):
    # no sapi credentials are stored on the path
    _sapi_connection = False


@unittest.skipUnless(_sapi_connection, "no connection to sapi web services")
class TestComposite(unittest.TestCase):
    def test_instantiation_smoketest(self):
        sampler = micro.EmbeddingComposite(micro.DWaveSampler())

    def test_sample_ising(self):
        sampler = micro.EmbeddingComposite(micro.DWaveSampler())

        h = {0: -1., 4: 2}
        J = {(0, 4): 1.5}

        response = sampler.sample_ising(h, J)

        # nothing failed and we got at least one response back
        self.assertGreaterEqual(len(response), 1)

        for sample in response.samples():
            self.assertIsInstance(sample, dict)
            self.assertEqual(set(sample), set(h))

        for sample, energy in response.data(['sample', 'energy']):
            self.assertIsInstance(sample, dict)
            self.assertEqual(set(sample), set(h))
            self.assertAlmostEqual(dimod.ising_energy(sample, h, J),
                                   energy)

    def test_sample_ising_unstructured_not_integer_labelled(self):
        sampler = micro.EmbeddingComposite(micro.DWaveSampler())

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
        sampler = micro.EmbeddingComposite(micro.DWaveSampler())

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
        sampler = micro.EmbeddingComposite(micro.DWaveSampler())

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


# @mock.patch('dwave_micro_client_dimod.sampler.microclient')
class TestCompositeMock(unittest.TestCase):
    def test_sample_ising(self):

        mock_sampler = mock.MagicMock()
        mock_sampler.structure = (range(3), [(0, 1), (1, 2), (0, 2)], {0: {1, 2}, 1: {2, 0}, 2: {0, 1}})

        sampler = micro.EmbeddingComposite(mock_sampler)

        h = {'a': -1., 'b': 2}
        J = {('a', 'b'): 1.5}

        response = sampler.sample_ising(h, J)
