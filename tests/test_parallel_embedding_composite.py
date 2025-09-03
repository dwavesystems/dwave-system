# Copyright 2025 D-Wave
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
import random
from unittest import mock

import numpy as np
import networkx as nx

import dimod
import dwave_networkx as dnx

from dwave.system.testing import MockDWaveSampler
from dwave.system.composites import ParallelEmbeddingComposite
from dwave.preprocessing import SpinReversalTransformComposite
from minorminer.utils.parallel_embeddings import find_sublattice_embeddings
from minorminer import find_embedding


class TestParallelEmbeddings(unittest.TestCase):

    def test_assertions(self):
        with self.assertRaises(ValueError):
            sampler = dimod.ExactSolver()  # Not structured.
            source = nx.from_edgelist([(0, 1)])
            sampler = ParallelEmbeddingComposite(sampler, source=source)

        mock_sampler = MockDWaveSampler()
        with self.assertRaises(ValueError):
            sampler = ParallelEmbeddingComposite(mock_sampler)  # unclear embedding

        with self.assertRaises(ValueError):
            too_big = mock_sampler.to_networkx_graph()
            too_big.add_edge("a", "b")  # Too many nodes and edges
            sampler = ParallelEmbeddingComposite(mock_sampler, source=too_big)

        with self.assertRaises(ValueError):
            source = nx.from_edgelist([(0, 1)])
            embeddings = [{0: ("a",), 2: ("b",)}]  # Bad values
            sampler = ParallelEmbeddingComposite(
                mock_sampler, embeddings=embeddings, source=source
            )

        with self.assertRaises(ValueError):
            embeddings = [
                {0: (e[0],), 1: (e[1], 1)} for e in mock_sampler.edgelist
            ]  # Not disjoint
            sampler = ParallelEmbeddingComposite(mock_sampler, embeddings=embeddings)

        with self.assertRaises(ValueError):
            source = nx.from_edgelist([(0, 1)])
            embeddings = [
                {0: (mock_sampler.edgelist[0][0],), 2: (mock_sampler.edgelist[0][1],)}
            ]  # Bad keys
            sampler = ParallelEmbeddingComposite(
                mock_sampler, source=source, embeddings=embeddings
            )

    def test_basic(self):
        mock_sampler = MockDWaveSampler()

        # Single nodes, MockSampler solves to optimality on every embedding:
        h = {"a": -2}  # +1 state of energy -1 every case.
        J = {}
        embeddings = [
            {"a": (n,)} for n in mock_sampler.nodelist
        ]  # NB: The default embedder (find_subgraph) doesn't respect disconnected nodes
        sampler = ParallelEmbeddingComposite(mock_sampler, embeddings=embeddings)
        num_reads = 3
        ss = sampler.sample_ising(h, J, num_reads=num_reads)
        self.assertEqual(num_reads * len(embeddings), sum(ss.record.num_occurrences))
        self.assertTrue(np.all(ss.record.energy == -2))
        self.assertTrue(np.all(ss.record.sample == 1))

        # Greedy matching: (-1,-1) state of energy -1.75 every case (no local minima)
        h = {0: 1, 1: 0.5}
        J = {(0, 1): -0.25}
        used_nodes = set()
        embeddings0 = []
        for e in mock_sampler.edgelist:
            if e[0] not in used_nodes and e[1] not in used_nodes:
                used_nodes.add(e[0])
                used_nodes.add(e[1])
                embeddings0.append({idx: (n,) for idx, n in enumerate(e)})

        source = nx.from_edgelist([(0, 1)])
        for embeddings in [embeddings0, None]:  # Provided or searched
            sampler = ParallelEmbeddingComposite(
                mock_sampler, embeddings=embeddings, source=source
            )
            num_reads = 2
            ss = sampler.sample_ising(h, J, num_reads=num_reads)
            if embeddings is None:
                self.assertEqual(
                    num_reads * len(sampler.embeddings), sum(ss.record.num_occurrences)
                )
            else:
                self.assertEqual(
                    num_reads * len(embeddings), sum(ss.record.num_occurrences)
                )
            self.assertTrue(np.all(ss.record.energy == -1.75))
            self.assertTrue(np.all(ss.record.sample == -1))

    def test_composite_propagation(self):
        # Propagation fails for TilingComposite but succeeds here.
        # When using find_sublattice_embedding it is necessayr to specify
        # the family and shape of the QPU as part of embedder_kwargs
        mock_sampler0 = MockDWaveSampler()
        mock_sampler = SpinReversalTransformComposite(mock_sampler0)
        embeddings = []
        used_nodes = set()
        for e in mock_sampler.child.edgelist:
            if e[0] not in used_nodes and e[1] not in used_nodes:
                used_nodes.add(e[0])
                used_nodes.add(e[1])
                embeddings.append({idx: (n,) for idx, n in enumerate(e)})
        sampler = ParallelEmbeddingComposite(mock_sampler, embeddings=embeddings)
        source = tile = dnx.chimera_graph(1, 1, 4)  # A 1:1 mapping assumed
        J = {e: -1 for e in tile.edges}  # A ferromagnet on the Chimera tile.
        embedder_kwargs = {"max_num_emb": None}
        sampler = ParallelEmbeddingComposite(
            mock_sampler, source=source, embedder_kwargs=embedder_kwargs
        )
        sampleset = sampler.sample_ising({}, J, num_reads=1)
        self.assertGreater(
            len(sampleset), 1
        )  # Equal to the number of parallel embeddings

        embedder = find_sublattice_embeddings
        embedder_kwargs = {
            "max_num_emb": None,
            "tile": tile,
            "T_family": mock_sampler0.properties["topology"]["type"],
            "T_kwargs": {"m": mock_sampler0.properties["topology"]["shape"][0]},
        }
        sampler = ParallelEmbeddingComposite(
            mock_sampler,
            source=source,
            embedder=embedder,
            embedder_kwargs=embedder_kwargs,
        )
        sampleset = sampler.sample_ising({}, J, num_reads=1)
        self.assertGreater(
            len(sampleset), 1
        )  # Equal to the number of parallel embeddings


class TestTiling(unittest.TestCase):
    """Testing for purposes of TilingComposite deprecation. See also testing of
    find_sublattice_embeddings in minorminer."""

    def test_pegasus_cell(self):
        # Test trivial case of single cell (K4,4) embedding over defect free
        mock_sampler = MockDWaveSampler(
            topology_type="pegasus"
        )  # P3 structured sampler
        m = n = mock_sampler.properties["topology"]["shape"][0] - 1  # Nice dimensions.
        expected_number_of_cells = m * n * 3  # Upper bound without tiling
        num_reads = 10

        t = 4
        tile = dnx.chimera_graph(1, 1, t)
        source = nx.complete_graph(t)  # Embeds easily on a tile, chain length 2

        # By find_multiple_embedding (default)
        sampler = ParallelEmbeddingComposite(mock_sampler, source=source)
        h = {node: random.uniform(-1, 1) for node in sampler.structure.nodelist}
        J = {(u, v): random.uniform(-1, 1) for u, v in sampler.structure.edgelist}
        response = sampler.sample_ising(h, J, num_reads=num_reads)
        self.assertLess(
            sum(response.record.num_occurrences) - 1,
            expected_number_of_cells * num_reads,
        )

        # By tiling (find_sublattice_embeddings)
        tile_embedding0 = {i: (i, i + t) for i in range(4)}  # K4 clique embedding
        for tile_embedding in [
            None,
            tile_embedding0,
        ]:  # Embedding searched, or provided.
            embedder = find_sublattice_embeddings
            embedder_kwargs = {
                "tile": tile,
                "tile_embedding": tile_embedding,
                "one_to_iterable": True,
                "max_num_emb": None,
                "use_tile_embedding": True,
                "embedder": find_embedding,  # Succeeds deterministically under defaults for this example.
            }
            sampler = ParallelEmbeddingComposite(
                mock_sampler,
                source=source,
                embedder=embedder,
                embedder_kwargs=embedder_kwargs,
                one_to_iterable=True,
            )
            response = sampler.sample_ising(h, J, num_reads=num_reads)
            self.assertEqual(
                sum(response.record.num_occurrences),
                expected_number_of_cells * num_reads,
            )

    def test_pegasus_multi_cell(self):
        # Test case of 2x3 nice cell embedding over defect free
        mock_sampler = MockDWaveSampler(
            topology_type="pegasus", topology_shape=[8]
        )  # P8 structured sampler
        self.assertTrue(
            "topology" in mock_sampler.properties
            and "type" in mock_sampler.properties["topology"]
        )
        self.assertTrue(
            mock_sampler.properties["topology"]["type"] == "pegasus"
            and "shape" in mock_sampler.properties["topology"]
        )
        # sampler = TilingComposite(mock_sampler, 1, 1)
        tile = dnx.chimera_graph(1)
        embedder = find_sublattice_embeddings
        embedder_kwargs = {
            "tile": tile,
            "max_num_emb": None,
            "use_tile_embedding": True,
        }
        sampler = ParallelEmbeddingComposite(
            mock_sampler,
            source=tile,
            embedder=embedder,
            embedder_kwargs=embedder_kwargs,
        )
        h = {node: random.uniform(-1, 1) for node in sampler.structure.nodelist}
        J = {(u, v): random.uniform(-1, 1) for u, v in sampler.structure.edgelist}

        m_sub = 2
        n_sub = 3
        # sampler = TilingComposite(mock_sampler, m_sub, n_sub)
        tile = dnx.chimera_graph(m=m_sub, n=n_sub)
        embedder = find_sublattice_embeddings
        embedder_kwargs = {
            "tile": tile,
            "max_num_emb": None,
            "use_tile_embedding": True,
        }
        sampler = ParallelEmbeddingComposite(
            mock_sampler,
            source=tile,
            embedder=embedder,
            embedder_kwargs=embedder_kwargs,
        )
        h = {node: random.uniform(-1, 1) for node in sampler.structure.nodelist}
        J = {(u, v): random.uniform(-1, 1) for u, v in sampler.structure.edgelist}

        m = n = mock_sampler.properties["topology"]["shape"][0] - 1
        expected_number_of_cells = (m // m_sub) * (n // 3) * 3
        num_reads = 1
        response = sampler.sample_ising(h, J, num_reads=num_reads)
        self.assertTrue(
            sum(response.record.num_occurrences) == expected_number_of_cells * num_reads
        )

    def test_tile_around_hole(self):

        # Create a Chimera C4 structured sampler with a node missing, so that
        # we have a defect pattern:
        # OOOX
        # OOOO
        # OOOO
        # OOOO
        # where O: complete cell, X: incomplete cell
        chimera_shape = [4, 4, 4]
        mock_sampler = MockDWaveSampler(
            broken_nodes=[8 * 3], topology_type="chimera", topology_shape=chimera_shape
        )
        hardware_graph = dnx.chimera_graph(*chimera_shape)  # C4

        # Tile with 2x2 cells:
        # sampler = TilingComposite(mock_sampler, 2, 2, 4)
        tile = dnx.chimera_graph(2, 2, 4)
        embedder = find_sublattice_embeddings
        embedder_kwargs = {
            "tile": tile,
            "max_num_emb": None,
            "use_tile_embedding": True,
        }
        sampler = ParallelEmbeddingComposite(
            mock_sampler,
            source=tile,
            embedder=embedder,
            embedder_kwargs=embedder_kwargs,
        )

        # Given the above chimera graph, check that the embeddings are as
        # follows:
        # 00XX
        # 0011
        # 2211
        # 22XX
        # where 0,1,2: belongs to correspoding embedding, X: not used in any
        # embedding
        self.assertSetEqual(
            {v for s in sampler.embeddings[0].values() for v in s},
            {
                linear_index
                for linear_index, (i, j, u, k) in hardware_graph.nodes(
                    data="chimera_index"
                )
                if i in (0, 1) and j in (0, 1)
            },
        )
        self.assertSetEqual(
            {v for s in sampler.embeddings[1].values() for v in s},
            {
                linear_index
                for linear_index, (i, j, u, k) in hardware_graph.nodes(
                    data="chimera_index"
                )
                if i in (1, 2) and j in (2, 3)
            },
        )
        self.assertSetEqual(
            {v for s in sampler.embeddings[2].values() for v in s},
            {
                linear_index
                for linear_index, (i, j, u, k) in hardware_graph.nodes(
                    data="chimera_index"
                )
                if i in (2, 3) and j in (0, 1)
            },
        )

    def test_tile_around_node_defects_pegasus(self):
        pegasus_shape = [5]
        # Create a pegasus P5 structured solver subject to node defects with the
        # following (nice-coordinate) 3x4x4 cell-level structure:
        # OOOX OOOO OOOO
        # OOOO OOOO OOOO
        # OOOO OOOO OOOO
        # OOOO OOOO OOOX
        # where O: complete cell, X: incomplete cell
        broken_node_nice_coordinates = [(0, 0, 3, 0, 1), (2, 3, 3, 1, 3)]
        broken_node_linear_coordinates = [
            dnx.pegasus_coordinates(pegasus_shape[0]).nice_to_linear(coord)
            for coord in broken_node_nice_coordinates
        ]
        mock_sampler = MockDWaveSampler(
            topology_type="pegasus",
            topology_shape=pegasus_shape,
            broken_nodes=broken_node_linear_coordinates,
        )
        # Tile with 2x2 cells:

        # sampler = TilingComposite(mock_sampler, 2, 2, 4)  # Before!
        tile = dnx.chimera_graph(2, 2, 4)
        embedder = find_sublattice_embeddings
        embedder_kwargs = {
            "tile": tile,
            "max_num_emb": None,
            "use_tile_embedding": True,
        }
        sampler = ParallelEmbeddingComposite(
            mock_sampler,
            source=tile,
            embedder=embedder,
            embedder_kwargs=embedder_kwargs,
        )
        # Given the above pegasus graph, check that the embeddings are as
        # follows:
        # 00XX  3344 7788
        # 0011  3344 7788
        # 2211  5566 99XX
        # 22XX  5566 99XX

        # For additional insight try:
        # import matplotlib.pyplot as plt
        # from dwave_networkx import draw_parallel_embeddings

        # draw_parallel_embeddings(mock_sampler.to_networkx_graph(), sampler.embeddings)
        # plt.show()
        all_embedded_to_nodes = [
            n for emb in sampler.embeddings for c in emb.values() for n in c
        ]
        set_embedded_to_nodes = set(all_embedded_to_nodes)
        self.assertTrue(
            set_embedded_to_nodes.issubset(set(mock_sampler.nodelist)),
            "embedded-to nodes are valid",
        )
        self.assertEqual(
            len(all_embedded_to_nodes),
            len(set_embedded_to_nodes),
            "embeddings should be disjoint",
        )
        self.assertFalse(any([len(emb) != 32 for emb in sampler.embeddings]))
        self.assertTrue(len(sampler.embeddings) == 10)

    def test_tile_around_edge_defects_pegasus(self):
        pegasus_shape = [5]

        # P5 structured sampler with one missing external edge that does not p
        # prevent tesselation of 2x2 blocks (12 tiles, equivalent to full yield)
        broken_edges_nice_coordinates = [(0, 1, 0, 0, 0), (0, 2, 0, 0, 0)]
        broken_edges = [
            tuple(
                dnx.pegasus_coordinates(pegasus_shape[0]).nice_to_linear(coord)
                for coord in broken_edges_nice_coordinates
            )
        ]
        mock_sampler = MockDWaveSampler(
            topology_type="pegasus",
            topology_shape=pegasus_shape,
            broken_edges=broken_edges,
        )
        # sampler = TilingComposite(mock_sampler, 2, 2, 4) Deprecated
        tile = dnx.chimera_graph(2, 2, 4)
        embedder = find_sublattice_embeddings
        embedder_kwargs = {
            "tile": tile,
            "max_num_emb": None,
            "use_tile_embedding": True,
        }
        sampler = ParallelEmbeddingComposite(
            mock_sampler,
            source=tile,
            embedder=embedder,
            embedder_kwargs=embedder_kwargs,
        )

        self.assertTrue(len(sampler.embeddings) == 12)

        # P5 structured sampler with one missing internal edge that prevents
        # tesselation of 2x2 blocks (otherwise 12 tiles, with edge defect 11)
        broken_edge_nice_coordinates = [(0, 0, 0, 0, 0), (0, 0, 0, 1, 0)]
        broken_edges = [
            tuple(
                dnx.pegasus_coordinates(pegasus_shape[0]).nice_to_linear(coord)
                for coord in broken_edge_nice_coordinates
            )
        ]
        mock_sampler = MockDWaveSampler(
            topology_type="pegasus",
            topology_shape=pegasus_shape,
            broken_edges=broken_edges,
        )
        # sampler = TilingComposite(mock_sampler, 2, 2, 4)  # Deprecated
        tile = dnx.chimera_graph(2, 2, 4)
        embedder = find_sublattice_embeddings
        embedder_kwargs = {
            "tile": tile,
            "max_num_emb": None,
            "use_tile_embedding": True,
        }
        sampler = ParallelEmbeddingComposite(
            mock_sampler,
            source=tile,
            embedder=embedder,
            embedder_kwargs=embedder_kwargs,
        )
        self.assertTrue(len(sampler.embeddings) == 11)

    def test_sample_ising(self):
        # sampler = TilingComposite(MockDWaveSampler(), 2, 2)  # Deprecated
        mock_sampler = MockDWaveSampler()
        tile = dnx.chimera_graph(m=2, n=2)
        embedder = find_sublattice_embeddings
        embedder_kwargs = {
            "tile": tile,
            "max_num_emb": None,
            "use_tile_embedding": True,
        }
        sampler = ParallelEmbeddingComposite(
            mock_sampler,
            source=tile,
            embedder=embedder,
            embedder_kwargs=embedder_kwargs,
        )
        h = {node: random.uniform(-1, 1) for node in sampler.structure.nodelist}
        J = {(u, v): random.uniform(-1, 1) for u, v in sampler.structure.edgelist}

        num_reads = 2
        response = sampler.sample_ising(h, J)
        response = sampler.sample_ising(h, J, num_reads=num_reads)

        self.assertEqual(
            len(sampler.embeddings) * num_reads, sum(response.record.num_occurrences)
        )
        self.assertTrue(set(np.unique(response.record.sample)).issubset({-1, 1}))

        # nothing failed and we got at least one response back per tile
        self.assertGreaterEqual(len(response), len(sampler.embeddings))

        for sample in response.samples():
            for v in h:
                self.assertIn(v, sample)

        for sample, energy in response.data(["sample", "energy"]):
            self.assertAlmostEqual(dimod.ising_energy(sample, h, J), energy)

    def test_sample_qubo(self):
        # sampler = TilingComposite(MockDWaveSampler(), 2, 2)
        mock_sampler = MockDWaveSampler()
        tile = dnx.chimera_graph(m=2, n=2)
        embedder = find_sublattice_embeddings
        embedder_kwargs = {
            "tile": tile,
            "max_num_emb": None,
            "use_tile_embedding": True,
        }
        sampler = ParallelEmbeddingComposite(
            mock_sampler,
            source=tile,
            embedder=embedder,
            embedder_kwargs=embedder_kwargs,
        )

        Q = {(u, v): random.uniform(-1, 1) for u, v in sampler.structure.edgelist}
        Q.update(
            {(node, node): random.uniform(-1, 1) for node in sampler.structure.nodelist}
        )

        response = sampler.sample_qubo(Q)

        # nothing failed and we got at least one response back per tile
        self.assertGreaterEqual(len(response), len(sampler.embeddings))

        for sample in response.samples():
            for u, v in Q:
                self.assertIn(v, sample)
                self.assertIn(u, sample)

        for sample, energy in response.data(["sample", "energy"]):
            self.assertAlmostEqual(dimod.qubo_energy(sample, Q), energy)

    def test_too_many_nodes(self):
        mock_sampler = MockDWaveSampler()  # C4 structured sampler

        # sampler = TilingComposite(mock_sampler, 2, 2)  # Deprecated
        tile = dnx.chimera_graph(m=2, n=2)
        embedder = find_sublattice_embeddings
        embedder_kwargs = {
            "tile": tile,
            "max_num_emb": None,
            "use_tile_embedding": True,
        }
        sampler = ParallelEmbeddingComposite(
            mock_sampler,
            source=tile,
            embedder=embedder,
            embedder_kwargs=embedder_kwargs,
        )

        h = {0: -1, 1: 1}
        J = {}

        response = sampler.sample_ising(h, J)

        __, num_columns = response.record.sample.shape

        self.assertEqual(num_columns, 2)

    def test_chain_strength(self):
        t = 3
        mock_sampler = MockDWaveSampler(
            topology_type="chimera", topology_shape=[1, 1, t]
        )  # A structured solver for K_{3,3}
        h = {i: -0.25 for i in range(3)}
        J = {(i, (i + 1) % 3): 1 for i in range(3)}
        embeddings = [{i: (i, t + i) for i in range(3)}]
        sampler = ParallelEmbeddingComposite(mock_sampler, embeddings=embeddings)
        ss = sampler.sample_ising(h, J, chain_strength=1)
        with self.subTest("Sufficient chain_strength finds the ground state"):
            self.assertEqual(ss.record.energy, -1.25)
        ss = sampler.sample_ising(h, J, chain_strength=0)  # 6-loop embedded model
        self.assertEqual(
            ss.record.energy,
            2.25,
            "Ground state by ExactSolver is chain broken and voted back to excited state.",
        )

    def test_initial_state(self):
        t = 2
        nr = 1
        nc = 1
        sampler = MockDWaveSampler(topology_type="chimera", topology_shape=[nr, nc, t])

        class MockDWaveSamplerAlt(MockDWaveSampler):
            """Replace when initial_state tuple functionality in MockDWaveSampler is
            corrected."""

            def sample(self, bqm, **kwargs):
                initial_state = kwargs.pop("initial_state")
                initial_state_tuple = [
                    (i, initial_state[i]) if i in initial_state else (i, 3)
                    for i in self.nodelist
                ]
                return super().sample(
                    bqm=bqm, initial_state=initial_state_tuple, **kwargs
                )

        sampler = MockDWaveSamplerAlt(
            topology_type="chimera", topology_shape=[nr, nc, t]
        )
        embeddings = [
            {0: (cell * 2 * t,), 1: (cell * 2 * t + t,)} for cell in range(nr * nc)
        ]
        h = {0: 1, 1: 1}
        J = {(0, 1): -2}
        initial_state = {i: 1 for i in range(2)}  # local minima.
        with mock.patch.object(sampler, "exact_solver_cutoff", 0):  # Steepest decent.
            # QPU format initial states:
            psampler = ParallelEmbeddingComposite(sampler, embeddings=embeddings)
            ss = psampler.sample_ising(
                h=h,
                J=J,
                num_reads=1,
                answer_mode="raw",
                initial_state=initial_state,
            )
            self.assertTrue(ss.record.sample.size == len(h) * len(embeddings))
            self.assertTrue(np.all(ss.record.sample == 1))

    def test_sample_multiple(self):
        # Two identical models, with two different chain strengths (see test chain_strength):
        t = 3
        mock_sampler = MockDWaveSampler(
            topology_type="chimera", topology_shape=[2, 1, t]
        )  # A structured solver with two K_{3,3} cells
        h = {i: -0.25 for i in range(3)}
        J = {(i, (i + 1) % 3): 1 for i in range(3)}
        embedding0 = {i: (i, t + i) for i in range(3)}
        embeddings = [
            embedding0,
            {k: tuple(q + 2 * t for q in v) for k, v in embedding0.items()},
        ]  # Shift second embedding one cell down.
        sampler = ParallelEmbeddingComposite(mock_sampler, embeddings=embeddings)
        bqms = [dimod.BinaryQuadraticModel("SPIN").from_ising(h, J)] * 2
        ss, info = sampler.sample_multiple(bqms=bqms, chain_strengths=[1, 0])
        self.assertEqual(len(ss), len(embeddings))
        self.assertEqual(type(info), dict)
        self.assertEqual(
            ss[0].record.energy,
            -1.25,
            "Sufficient chain_strength finds the ground state",
        )
        self.assertEqual(
            ss[1].record.energy,
            2.25,
            "Ground state by ExactSolver is chain broken and voted back to excited state.",
        )
        # solve a frustrated and unfrustrated model:
        J_ferro = {k: -abs(v) for k, v in J.items()}
        bqms[0] = dimod.BinaryQuadraticModel("SPIN").from_ising(h, J_ferro)
        # check None works for chain strengths:
        ss, info = sampler.sample_multiple(bqms=bqms)
        self.assertEqual(
            ss[0].record.energy,
            -3.75,
            "Sufficient chain_strength finds the ground state",
        )
        self.assertEqual(
            ss[1].record.energy,
            -1.25,
            "Sufficient chain_strength finds the ground state",
        )
