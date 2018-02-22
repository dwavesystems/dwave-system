import unittest
import random

import dwave_micro_client as microclient
import dimod
import dwave_networkx as dnx
from dwave_micro_client_dimod.sampler import Structure

try:
    # py3
    import unittest.mock as mock
except ImportError:
    # py2
    import mock

import dwave_micro_client_dimod as micro

try:
    microclient.Connection()
    _sapi_connection = True
except (IOError, OSError):
    # no sapi credentials are stored on the path
    _sapi_connection = False


@unittest.skipUnless(_sapi_connection, "no connection to sapi web services")
class TestTiling(unittest.TestCase):
    def test_instantiation_smoketest(self):
        sampler = micro.TilingComposite(micro.DWaveSampler(), 2, 2)

    def test_sample_ising(self):
        sampler = micro.TilingComposite(micro.DWaveSampler(), 2, 2)

        h = {node: random.uniform(-1, 1) for node in sampler.structure.nodelist}
        J = {edge: random.uniform(-1, 1) for edge in sampler.structure.edgelist}

        response = sampler.sample_ising(h, J)

        # nothing failed and we got at least one response back per tile
        self.assertGreaterEqual(len(response), len(sampler.embeddings))

        for sample in response.samples():
            for v in h:
                self.assertIn(v, sample)

        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(dimod.ising_energy(sample, h, J), energy)

    def test_sample_qubo(self):
        sampler = micro.TilingComposite(micro.DWaveSampler(), 2, 2)

        Q = {edge: random.uniform(-1, 1) for edge in sampler.structure.edgelist}
        Q.update({(node, node): random.uniform(-1, 1) for node in sampler.structure.nodelist})

        response = sampler.sample_qubo(Q)

        # nothing failed and we got at least one response back per tile
        self.assertGreaterEqual(len(response), len(sampler.embeddings))

        for sample in response.samples():
            for u, v in Q:
                self.assertIn(v, sample)
                self.assertIn(u, sample)

        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(dimod.qubo_energy(sample, Q), energy)


# @mock.patch('dwave_micro_client_dimod.sampler.microclient')
class TestTilingMock(unittest.TestCase):
    def test_sample_ising(self):
        mock_sampler = mock.MagicMock()
        hardware_graph = dnx.chimera_graph(4)
        mock_sampler.structure = Structure(hardware_graph.nodes, hardware_graph.edges, hardware_graph.adj)

        sampler = micro.TilingComposite(mock_sampler, 2, 2)

        h = {node: random.uniform(-1, 1) for node in sampler.structure.nodelist}
        J = {edge: random.uniform(-1, 1) for edge in sampler.structure.edgelist}

        response = sampler.sample_ising(h, J)

    def test_tile_around_hole(self):
        mock_sampler = mock.MagicMock()

        # Create a chimera graph with the following structure:
        # OOOX
        # OOOO
        # OOOO
        # OOOO
        # where O: complete cell, X: incomplete cell
        size = 4
        nodes_per_cell = 8
        nodes = list(range(size * size * nodes_per_cell))
        nodes.remove(nodes_per_cell * 3)

        hardware_graph = dnx.chimera_graph(size, node_list=nodes)
        mock_sampler.structure = Structure(hardware_graph.nodes, hardware_graph.edges, hardware_graph.adj)

        sampler = micro.TilingComposite(mock_sampler, 2, 2, 4)
        # Given the above chimera graph, check that the embeddings are as follows:
        # 00XX
        # 0011
        # 2211
        # 22XX
        # where 0,1,2: belongs to correspoding embedding, X: not used in any embedding
        self.assertSetEqual({v for s in sampler.embeddings[0].values() for v in s},
                            {linear_index for (linear_index, (i, j, u, k)) in hardware_graph.nodes(data='chimera_index')
                             if i in (0, 1) and j in (0, 1)})
        self.assertSetEqual({v for s in sampler.embeddings[1].values() for v in s},
                            {linear_index for (linear_index, (i, j, u, k)) in hardware_graph.nodes(data='chimera_index')
                             if i in (1, 2) and j in (2, 3)})
        self.assertSetEqual({v for s in sampler.embeddings[2].values() for v in s},
                            {linear_index for (linear_index, (i, j, u, k)) in hardware_graph.nodes(data='chimera_index')
                             if i in (2, 3) and j in (0, 1)})
