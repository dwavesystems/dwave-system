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

import itertools
import unittest
import warnings

from collections.abc import Mapping
from unittest import mock

import dimod
import dwave_networkx as dnx
from parameterized import parameterized_class

import dwave.embedding

from dwave.system.composites import (EmbeddingComposite,
                                     FixedEmbeddingComposite,
                                     LazyFixedEmbeddingComposite,
                                     LazyEmbeddingComposite,
                                     AutoEmbeddingComposite,
                                     )

from dwave.system.testing import MockDWaveSampler
from dwave.embedding import chain_breaks
from dwave.system.warnings import ChainStrengthWarning

try:
    from dwave.preprocessing import ScaleComposite
except ImportError:
    # fall back on dimod of dwave.preprocessing is not installed
    from dimod import ScaleComposite


@dimod.testing.load_sampler_bqm_tests(EmbeddingComposite(MockDWaveSampler()))
class TestEmbeddingComposite(unittest.TestCase):
    def test_instantiation_smoketest(self):
        sampler = EmbeddingComposite(MockDWaveSampler())

        dimod.testing.assert_sampler_api(sampler)

    def test_sample_ising(self):
        sampler = EmbeddingComposite(MockDWaveSampler())

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
        sampler = EmbeddingComposite(MockDWaveSampler())

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
        sampler = EmbeddingComposite(MockDWaveSampler())

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
        sampler = EmbeddingComposite(MockDWaveSampler())

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
        sampler = EmbeddingComposite(MockDWaveSampler())

        h = {0: -1., 4: 2}
        J = {}

        response = sampler.sample_ising(h, J)

        # nothing failed and we got at least one response back
        self.assertGreaterEqual(len(response), 1)

    def test_chain_break_method_customization(self):
        sampler = EmbeddingComposite(MockDWaveSampler())

        def mock_unembed(*args, **kwargs):
            self.assertIn('chain_break_method', kwargs)
            self.assertEqual(kwargs['chain_break_method'], chain_breaks.discard)
            mock_unembed.call_count += 1
            return dwave.embedding.unembed_sampleset(*args, **kwargs)

        mock_unembed.call_count = 0

        with mock.patch('dwave.system.composites.embedding.unembed_sampleset', mock_unembed):
            sampler.sample_ising({'a': 1}, {}, chain_break_method=chain_breaks.discard).resolve()

        self.assertEqual(mock_unembed.call_count, 1)

    def test_find_embedding_kwarg(self):
        child = dimod.StructureComposite(dimod.NullSampler(), [0, 1], [(0, 1)])

        def my_find_embedding(S, T):
            # does nothing
            return {v: [v] for v in set().union(*S)}

        sampler = EmbeddingComposite(child, find_embedding=my_find_embedding)

        # nothing breaks
        sampler.sample_ising({0: -1}, {})

    def test_embedding_parameters_construction(self):
        child = dimod.StructureComposite(dimod.NullSampler(), [0, 1], [(0, 1)])

        def my_find_embedding(S, T, a):
            assert a == -1
            return {v: [v] for v in set().union(*S)}

        sampler = EmbeddingComposite(child, find_embedding=my_find_embedding,
                                     embedding_parameters={'a': -1})

        # nothing breaks
        sampler.sample_ising({0: -1}, {})

    def test_embedding_parameters_sample(self):
        child = dimod.StructureComposite(dimod.NullSampler(), [0, 1], [(0, 1)])

        def my_find_embedding(S, T, a):
            assert a == -1
            return {v: [v] for v in set().union(*S)}

        sampler = EmbeddingComposite(child, find_embedding=my_find_embedding)

        # nothing breaks
        sampler.sample_ising({0: -1}, {}, embedding_parameters={'a': -1})

    def test_intermediate_composites(self):
        child = dimod.StructureComposite(dimod.NullSampler(), [0, 1], [(0, 1)])
        intermediate = dimod.TrackingComposite(child)
        sampler = EmbeddingComposite(intermediate)
        self.assertEqual(sampler.target_structure.nodelist, [0, 1])

    def test_scale_aware_scale_composite(self):

        nodelist = [0, 1, 2]
        edgelist = [(0, 1), (1, 2), (0, 2)]
        embedding = {'a': [0], 'b': [1, 2]}

        sampler = FixedEmbeddingComposite(
            dimod.TrackingComposite(
                ScaleComposite(
                    dimod.StructureComposite(
                        dimod.NullSampler(),
                        nodelist, edgelist))),
            embedding=embedding, scale_aware=True)

        _ = sampler.sample_ising({'a': 100, 'b': -100}, {'ab': 300})

        self.assertIn('ignored_interactions', sampler.child.input)
        ignored = sampler.child.input['ignored_interactions']

        self.assertTrue(ignored == [(1, 2)] or ignored == [(2, 1)])

    def test_return_embedding_subgraph(self):
        # problem is on a subgraph - embedding is reduced to relabeling
        nodelist = [0, 1, 2]
        edgelist = [(0, 1), (1, 2), (0, 2)]

        sampler = EmbeddingComposite(
            dimod.StructureComposite(dimod.NullSampler(), nodelist, edgelist))

        sampleset = sampler.sample_ising({'a': -1}, {'ac': 1}, return_embedding=True)

        self.assertIn('embedding', sampleset.info['embedding_context'])
        embedding = sampleset.info['embedding_context']['embedding']
        self.assertEqual(set(embedding), {'a', 'c'})

        self.assertIn('chain_break_method', sampleset.info['embedding_context'])
        self.assertIsNone(sampleset.info['embedding_context']['chain_break_method'])

        self.assertIn('embedding_parameters', sampleset.info['embedding_context'])
        self.assertEqual(sampleset.info['embedding_context']['embedding_parameters'], {})

        self.assertIn('chain_strength', sampleset.info['embedding_context'])
        self.assertIsNone(sampleset.info['embedding_context']['chain_strength'])

        self.assertIn('timing', sampleset.info['embedding_context'])
        self.assertIn('embedding', sampleset.info['embedding_context']['timing'])
        self.assertIn('unembedding', sampleset.info['embedding_context']['timing'])

        # default False
        sampleset = sampler.sample_ising({'a': -1}, {'ac': 1})
        self.assertNotIn('embedding_context', sampleset.info)

    def test_return_embedding(self):
        # problem graph requires embedding
        nodelist = [0, 1, 2, 3]
        edgelist = [(0, 1), (1, 2), (2, 3), (3, 0)]

        sampler = EmbeddingComposite(
            dimod.StructureComposite(dimod.NullSampler(), nodelist, edgelist))

        sampleset = sampler.sample_ising({}, {'ab': 1, 'bc': 1, 'ca': 1}, return_embedding=True)

        self.assertIn('embedding', sampleset.info['embedding_context'])
        embedding = sampleset.info['embedding_context']['embedding']
        self.assertEqual(set(embedding), {'a', 'b', 'c'})

        self.assertIn('chain_break_method', sampleset.info['embedding_context'])
        self.assertEqual(sampleset.info['embedding_context']['chain_break_method'], 'majority_vote')

        self.assertIn('embedding_parameters', sampleset.info['embedding_context'])
        self.assertEqual(sampleset.info['embedding_context']['embedding_parameters'], {})

        self.assertIn('chain_strength', sampleset.info['embedding_context'])
        self.assertEqual(round(sampleset.info['embedding_context']['chain_strength'], 3), 2)

        self.assertIn('timing', sampleset.info['embedding_context'])
        self.assertIn('embedding', sampleset.info['embedding_context']['timing'])
        self.assertIn('unembedding', sampleset.info['embedding_context']['timing'])

        # default False
        sampleset = sampler.sample_ising({}, {'ab': 1, 'bc': 1, 'ca': 1})
        self.assertNotIn('embedding_context', sampleset.info)

    def test_return_embedding_as_class_variable(self):
        nodelist = [0, 1, 2]
        edgelist = [(0, 1), (1, 2), (0, 2)]

        sampler = EmbeddingComposite(
            dimod.StructureComposite(dimod.NullSampler(), nodelist, edgelist))

        # temporarily change return_embedding_default
        EmbeddingComposite.return_embedding_default = True

        sampleset = sampler.sample_ising({'a': -1}, {'ac': 1})

        self.assertIn('embedding', sampleset.info['embedding_context'])
        embedding = sampleset.info['embedding_context']['embedding']
        self.assertEqual(set(embedding), {'a', 'c'})

        self.assertIn('chain_break_method', sampleset.info['embedding_context'])
        self.assertIsNone(sampleset.info['embedding_context']['chain_break_method'])

        self.assertIn('embedding_parameters', sampleset.info['embedding_context'])
        self.assertEqual(sampleset.info['embedding_context']['embedding_parameters'], {})

        self.assertIn('chain_strength', sampleset.info['embedding_context'])
        self.assertIsNone(sampleset.info['embedding_context']['chain_strength'])

        # restore the default
        EmbeddingComposite.return_embedding_default = False

    def test_warnings(self):
        G = dnx.chimera_graph(12)

        sampler = EmbeddingComposite(
            dimod.StructureComposite(dimod.RandomSampler(), G.nodes, G.edges))

        # this will need chains lengths > 7
        J = {uv: -1 for uv in itertools.combinations(range(40), 2)}

        ss = sampler.sample_ising({}, J, warnings='SAVE')

        self.assertIn('warnings', ss.info)

    def test_warning_chain_strength(self):
        G = dnx.chimera_graph(12)

        sampler = EmbeddingComposite(
            dimod.StructureComposite(dimod.RandomSampler(), G.nodes, G.edges))

        # use a triangle so there is a chain of length 2
        J = {(0, 1): 100, (1, 2): 100, (0, 2): 1}

        ss = sampler.sample_ising({}, J, chain_strength=1, warnings='SAVE')
        self.assertIn('warnings', ss.info)

        count = 0
        for warning in ss.info['warnings']:
            if issubclass(warning['type'], ChainStrengthWarning):
                count += 1
        self.assertEqual(count, 1)

    def test_warnings_chain_strength_len1(self):
        G = dnx.chimera_graph(12)

        sampler = EmbeddingComposite(
            dimod.StructureComposite(dimod.RandomSampler(), G.nodes, G.edges))

        J = {(0, 1): 100}

        ss = sampler.sample_ising({}, J, warnings='SAVE')
        self.assertIn('warnings', ss.info)

        # should have no chain length
        count = 0
        for warning in ss.info['warnings']:
            if issubclass(warning['type'], ChainStrengthWarning):
                count += 1
        self.assertEqual(count, 0)

    def test_warnings_chain_strength_dict(self):
        sampler = EmbeddingComposite(MockDWaveSampler())

        linear = {'a': -1, 'b': -2, 'c': -3}
        quadratic = {('a', 'b'): 1, ('a', 'c'): -1, ('b', 'c'): 2}
        bqm = dimod.BQM(linear, quadratic, 0, dimod.SPIN)

        chain_strength = {'a': 10, 'b': 20, 'c': 1.5}   
        ss = sampler.sample(bqm, chain_strength=chain_strength, warnings='SAVE')

        self.assertIn('warnings', ss.info)
        self.assertEqual(len(ss.info['warnings']), 1)

        warning = ss.info['warnings'][0]
        self.assertEqual(warning['type'], ChainStrengthWarning)

        interactions = warning['data']['source_interactions']
        self.assertEqual(len(interactions), 1)
        self.assertCountEqual(interactions[0], ('b','c'))

    def test_warnings_as_class_variable(self):
        G = dnx.chimera_graph(12)

        sampler = EmbeddingComposite(
            dimod.StructureComposite(dimod.RandomSampler(), G.nodes, G.edges))

        # this will need chains lengths > 7
        J = {uv: -1 for uv in itertools.combinations(range(40), 2)}

        EmbeddingComposite.warnings_default = 'SAVE'

        ss = sampler.sample_ising({}, J)

        self.assertIn('warnings', ss.info)

        EmbeddingComposite.warnings_default = 'IGNORE'  # restore default


class TestFixedEmbeddingComposite(unittest.TestCase):
    def test_without_embedding_and_adjacency(self):
        self.assertRaises(TypeError, lambda: FixedEmbeddingComposite(MockDWaveSampler()))

    def test_instantiation_empty_embedding(self):
        sampler = FixedEmbeddingComposite(MockDWaveSampler(), {})

        dimod.testing.assert_sampler_api(sampler)  # checks adj consistent with nodelist/edgelist

        self.assertEqual(sampler.edgelist, [])

        self.assertTrue(hasattr(sampler, 'embedding'))
        self.assertIn('embedding', sampler.properties)

    def test_instantiation_empty_adjacency(self):
        with self.assertWarns(DeprecationWarning):
            sampler = FixedEmbeddingComposite(MockDWaveSampler(), source_adjacency={})

        dimod.testing.assert_sampler_api(sampler)  # checks for attributes needed in a sampler

        self.assertEqual(sampler.edgelist, [])

        self.assertTrue(hasattr(sampler, 'embedding'))
        self.assertIn('embedding', sampler.properties)

    def test_instantiation_triangle(self):
        embedding = {'a': (0, 4), 'b': (1, 5), 'c': (2, 6)}
        sampler = FixedEmbeddingComposite(MockDWaveSampler(), embedding)
        self.assertEqual(embedding, sampler.embedding)

        dimod.testing.assert_sampler_api(sampler)  # checks adj consistent with nodelist/edgelist

        self.assertEqual(sampler.nodelist, ['a', 'b', 'c'])
        self.assertEqual(sampler.edgelist, [('a', 'b'), ('a', 'c'), ('b', 'c')])

    def test_sample_bqm_triangle(self):
        sampler = FixedEmbeddingComposite(MockDWaveSampler(), {'a': [0, 4], 'b': [1, 5], 'c': [2, 6]})

        resp = sampler.sample_ising({'a': 1, 'b': 1, 'c': 0}, {})

        self.assertEqual(set(resp.variables), {'a', 'b', 'c'})

    def test_adjacency(self):
        square_adj = {1: [2, 3], 2: [1, 4], 3: [1, 4], 4: [2, 3]}
        with self.assertWarns(DeprecationWarning):
            sampler = FixedEmbeddingComposite(MockDWaveSampler(), source_adjacency=square_adj)

        self.assertTrue(hasattr(sampler, 'adjacency'))
        self.assertTrue(hasattr(sampler, 'embedding'))
        self.assertIn('embedding', sampler.properties)

        self.assertEqual(sampler.nodelist, [1, 2, 3, 4])
        self.assertEqual(sampler.edgelist, [(1, 2), (1, 3), (2, 4), (3, 4)])

    def test_chain_break_method_customization(self):
        sampler = FixedEmbeddingComposite(MockDWaveSampler(), {'a': [0]})

        def mock_unembed(*args, **kwargs):
            self.assertIn('chain_break_method', kwargs)
            self.assertEqual(kwargs['chain_break_method'], chain_breaks.discard)
            mock_unembed.call_count += 1
            return dwave.embedding.unembed_sampleset(*args, **kwargs)

        mock_unembed.call_count = 0

        with mock.patch('dwave.system.composites.embedding.unembed_sampleset', mock_unembed):
            sampler.sample_ising({'a': 1}, {}, chain_break_method=chain_breaks.discard).resolve()

        self.assertEqual(mock_unembed.call_count, 1)

    def test_keyer(self):
        C4 = dnx.chimera_graph(4)
        nodelist = sorted(C4.nodes)
        edgelist = sorted(sorted(edge) for edge in C4.edges)

        child = dimod.StructureComposite(dimod.NullSampler(), nodelist, edgelist)

        embedding = {0: [49, 53], 1: [52], 2: [50]}

        sampler = FixedEmbeddingComposite(child, embedding)

        with self.assertRaises(dwave.embedding.exceptions.MissingChainError):
            sampler.sample_qubo({(1, 4): 1})

        sampler.sample_qubo({(1, 2): 1}).record  # sample and resolve future
        sampler.sample_qubo({(1, 1): 1}).record  # sample and resolve future

    def test_minimize_energy_chain_break_method(self):
        # bug report https://github.com/dwavesystems/dwave-system/issues/206
        Q = {(0, 1): 1, (0, 2): 1, (0, 3): 1, (1, 2): 1, (1, 3): 1, (2, 3): 1,
             (0, 0): 1, (1, 1): 1, (2, 2): 1, (3, 3): 1}
        S = {(0, 1): 1, (0, 2): 1, (0, 3): 1, (1, 2): 1, (1, 3): 1, (2, 3): 1}
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

        G = dnx.chimera_graph(4)
        sampler = dimod.StructureComposite(dimod.ExactSolver(), G.nodes, G.edges)
        embedding = {0: [55], 1: [48], 2: [50, 53], 3: [52, 51]}
        composite = FixedEmbeddingComposite(sampler, embedding)

        cbm = chain_breaks.MinimizeEnergy(bqm, embedding)
        composite.sample(bqm, chain_break_method=cbm).resolve()

    def test_subgraph_relabeling(self):
        Z12 = dnx.zephyr_graph(12)
        nodelist = sorted(Z12.nodes)
        edgelist = sorted(sorted(edge) for edge in Z12.edges)

        child = dimod.StructureComposite(dimod.NullSampler(), nodelist, edgelist)

        # source bqm on shifted nodes
        bqm = dimod.BQM({n+1: 1 for n in nodelist}, {(u+1, v+1): 1 for u, v in edgelist}, 'SPIN')

        # 1-1 mapping
        embedding = {n+1: (n, ) for n in nodelist}

        sampler = FixedEmbeddingComposite(child, embedding)
        ss = sampler.sample(bqm)

        self.assertEqual(set(ss.variables), bqm.variables)

    def test_relabeling_performance_gain(self):
        with self.subTest('native subgraph Z6 -> Z6'):
            graph = dnx.zephyr_graph(6)
            nodelist = sorted(graph.nodes)
            edgelist = sorted(sorted(edge) for edge in graph.edges)

            bqm = dimod.BQM.from_qubo({tuple(e): 1 for e in edgelist})
            child = dimod.StructureComposite(dimod.NullSampler(), nodelist, edgelist)
            embedding = {n: [n] for n in nodelist}

            sampler = FixedEmbeddingComposite(child, embedding)
            ss = sampler.sample(bqm, return_embedding=True)
            self.assertEqual(set(ss.variables), bqm.variables)
            t1 = ss.info['embedding_context']['timing']

        with self.subTest('embedding with a single chain'):
            extra_node = nodelist[-1] + 1
            graph.add_edge(nodelist[-1], extra_node)
            nodelist = sorted(graph.nodes)
            edgelist = sorted(sorted(edge) for edge in graph.edges)

            child = dimod.StructureComposite(dimod.NullSampler(), nodelist, edgelist)
            embedding[max(embedding)] = [extra_node-1, extra_node]

            sampler = FixedEmbeddingComposite(child, embedding)
            ss = sampler.sample(bqm, return_embedding=True)
            self.assertEqual(set(ss.variables), bqm.variables)
            t2 = ss.info['embedding_context']['timing']

        with self.subTest('relabeling is faster than embedding'):
            self.assertLess(t1['embedding'], t2['embedding'])
            self.assertLess(t1['unembedding'], t2['unembedding'])


@dimod.testing.load_sampler_bqm_tests(lambda: LazyFixedEmbeddingComposite(MockDWaveSampler()))
class TestLazyFixedEmbeddingComposite(unittest.TestCase):
    def test_sample_instantiation(self):
        # Check that graph related values have not been instantiated
        sampler = LazyFixedEmbeddingComposite(MockDWaveSampler())
        self.assertIsNone(sampler.embedding)
        self.assertIsNone(sampler.nodelist)
        self.assertIsNone(sampler.edgelist)
        self.assertIsNone(sampler.adjacency)

        # Set up an and_gate BQM and sample
        Q = {('a', 'a'): 0.0, ('c', 'c'): 6.0, ('b', 'b'): 0.0, ('b', 'a'): 2.0, ('c', 'a'): -4.0, ('c', 'b'): -4.0}
        sampler.sample_qubo(Q)

        # Check that values have been populated
        self.assertIsNotNone(sampler.embedding)
        self.assertEqual(sampler.nodelist, ['a', 'b', 'c'])
        self.assertEqual(sampler.edgelist, [('a', 'b'), ('a', 'c'), ('b', 'c')])
        self.assertEqual(sampler.adjacency, {'a': {'b', 'c'}, 'b': {'a', 'c'}, 'c': {'a', 'b'}})

    def test_same_embedding(self):
        sampler = LazyFixedEmbeddingComposite(MockDWaveSampler())

        # Set up Ising and sample
        h = {'a': 1, 'b': 1, 'c': 1}
        J = {('a', 'b'): 3, ('b', 'c'): -2, ('a', 'c'): 1}
        sampler.sample_ising(h, J)

        # Store embedding
        prev_embedding = sampler.embedding

        # Set up QUBO of an or_gate
        Q = {('a', 'a'): 2.0, ('c', 'c'): 2.0, ('b', 'b'): 2.0, ('b', 'a'): 2.0, ('c', 'a'): -4.0, ('c', 'b'): -4.0}
        sampler.sample_qubo(Q)

        # Check that the same embedding is used
        self.assertEqual(sampler.embedding, prev_embedding)

    def test_ising(self):
        h = {0: 11, 5: 2}
        J = {(0, 5): -8}
        sampler = LazyFixedEmbeddingComposite(MockDWaveSampler())
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
        sampler = LazyFixedEmbeddingComposite(MockDWaveSampler())
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
        sampler = LazyFixedEmbeddingComposite(MockDWaveSampler())
        response = sampler.sample_qubo(Q)

        # Check embedding
        self.assertIsNotNone(sampler.embedding)
        self.assertEqual(sampler.nodelist, [1, 2, 3])
        self.assertEqual(sampler.edgelist, [(1, 2), (1, 3)])

        # Check that at least one response was found
        self.assertGreaterEqual(len(response), 1)

    def test_chain_break_method_customization(self):
        sampler = LazyFixedEmbeddingComposite(MockDWaveSampler())

        def mock_unembed(*args, **kwargs):
            self.assertIn('chain_break_method', kwargs)
            self.assertEqual(kwargs['chain_break_method'], chain_breaks.discard)
            mock_unembed.call_count += 1
            return dwave.embedding.unembed_sampleset(*args, **kwargs)

        mock_unembed.call_count = 0

        with mock.patch('dwave.system.composites.embedding.unembed_sampleset', mock_unembed):
            sampler.sample_ising({'a': 1}, {}, chain_break_method=chain_breaks.discard).resolve()

        self.assertEqual(mock_unembed.call_count, 1)


class TestLazyEmbeddingComposite(unittest.TestCase):
    def test_deprecation_raise(self):
        # Temporarily mutate warnings filter
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Cause all warnings to always be triggered.
            LazyEmbeddingComposite(MockDWaveSampler())  # Trigger warning

            # Verify deprecation warning
            assert len(w) == 1
            assert issubclass(w[-1].category, DeprecationWarning)
            assert "renamed" in str(w[-1].message)

    def test_ising_sample(self):
        h = {'a': 1, 'b': -2}
        J = {('a', 'b'): -3}
        sampler = LazyFixedEmbeddingComposite(MockDWaveSampler())
        response = sampler.sample_ising(h, J)

        # Check that at least one response was found
        self.assertGreaterEqual(len(response), 1)


@dimod.testing.load_sampler_bqm_tests(AutoEmbeddingComposite(MockDWaveSampler()))
class TestAutoEmbeddingComposite(unittest.TestCase):
    def test_broken_find_embedding(self):
        nodelist = [0, 1, 2, 3]
        edgelist = [(0, 1), (1, 2), (2, 3), (0, 3)]
        child = dimod.StructureComposite(dimod.NullSampler(), nodelist, edgelist)

        def find_embedding(*args, **kwargs):
            raise NotImplementedError

        sampler = AutoEmbeddingComposite(child, find_embedding=find_embedding)

        sampler.sample_ising({}, {(0, 1): -1})

        with self.assertRaises(NotImplementedError):
            sampler.sample_ising({}, {('a', 0): -1})

    def test_smoke(self):
        nodelist = [0, 1, 2, 3]
        edgelist = [(0, 1), (1, 2), (2, 3), (0, 3)]
        child = dimod.StructureComposite(dimod.NullSampler(), nodelist, edgelist)

        sampler = AutoEmbeddingComposite(child)

        sampler.sample_ising({}, {(0, 1): -1})

        sampler.sample_ising({}, {('a', 0): -1})

    def test_unstructured(self):
        child = dimod.NullSampler()

        sampler = AutoEmbeddingComposite(child)

        sampler.sample_ising({}, {(0, 1): -1})

        sampler.sample_ising({}, {('a', 0): -1}, embedding_parameters={})


@parameterized_class([
   dict(embedding_composite_class=EmbeddingComposite),
   dict(embedding_composite_class=AutoEmbeddingComposite),
])
class TestProblemLabelPropagation(unittest.TestCase):

    def test_problem_labelling(self):
        sampler = self.embedding_composite_class(MockDWaveSampler())

        self.assertIn('label', sampler.parameters)

        ss = sampler.sample_ising({}, {(0, 1): -1})
        self.assertNotIn('problem_label', ss.info)

        label = 'problem label'
        ss = sampler.sample_ising({}, {(0, 1): -1}, label=label)
        self.assertIn('problem_label', ss.info)
        self.assertEqual(ss.info['problem_label'], label)
