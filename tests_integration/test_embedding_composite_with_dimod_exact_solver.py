import unittest
import random
import itertools

import dimod
import dwave_networkx as dnx

import dimod.testing as dtest

from dwave.system.composites import EmbeddingComposite


class TestEmbeddingCompositeExactSolver(unittest.TestCase):
    def test_bug60(self):

        for __ in range(100):

            # make a small K5 bqm
            bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
            for v in range(5):
                bqm.add_variable(v, random.uniform(-1, 1))
            for u, v in itertools.combinations(range(5), 2):
                bqm.add_interaction(u, v, random.uniform(-1, 1))

            sampler_exact = dimod.ExactSolver()

            # get the structure of a C1,2 chimera lattice
            C12 = dnx.chimera_graph(2, 2, 3)
            nodelist = sorted(C12)
            edgelist = sorted(sorted(edge) for edge in C12.edges)

            sampler_structured = dimod.StructureComposite(sampler_exact, nodelist=nodelist, edgelist=edgelist)

            sampler_embedding = EmbeddingComposite(sampler_structured)

            dtest.assert_sampler_api(sampler_embedding)

            resp_exact = sampler_exact.sample(bqm)
            resp_emb = sampler_embedding.sample(bqm)

            for sample, energy in resp_emb.data(['sample', 'energy']):
                self.assertEqual(bqm.energy(sample), energy)

            ground_exact = dict(next(iter(resp_exact)))
            ground_embed = dict(next(iter(resp_emb)))

            self.assertEqual(ground_embed, ground_exact)
