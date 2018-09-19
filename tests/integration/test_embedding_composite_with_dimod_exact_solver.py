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
import random
import itertools

import dimod
import dwave_networkx as dnx

import dimod.testing as dtest

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite

try:
    DWaveSampler()
    _config_found = True
except ValueError:
    _config_found = False


class TestEmbeddingCompositeExactSolver(unittest.TestCase):

    def test_initial_state_kwarg_simple(self):
        sampler = EmbeddingComposite(DWaveSampler())

        if 'initial_state' not in sampler.parameters:
            raise unittest.SkipTest

        bqm = dimod.BinaryQuadraticModel.from_ising({'a': 2.0, 'b': -2.0}, {('a', 'b'): -1})

        kwargs = {'initial_state': {'a': 1, 'b': 1},
                  'anneal_schedule': [(0, 1), (55.0, 0.45), (155.0, 0.45), (210.0, 1)]}

        sampler.sample(bqm, **kwargs).record  # sample and resolve future

    def test_initial_state_kwarg_complicated(self):
        sampler = EmbeddingComposite(DWaveSampler())

        if 'initial_state' not in sampler.parameters:
            raise unittest.SkipTest

        h = {v: 0.0 for v in range(10)}
        J = {(u, v): 1 for u, v in itertools.combinations(range(10), 2)}

        state = {v: 2*bool(v % 2) - 1 for v in h}

        bqm = dimod.BinaryQuadraticModel.from_ising(h, J)

        kwargs = {'initial_state': state,
                  'anneal_schedule': [(0, 1), (55.0, 0.45), (155.0, 0.45), (210.0, 1)]}

        sampler.sample(bqm, **kwargs).record  # sample and resolve future


@unittest.skipUnless(_config_found, "no configuration found to connect to a system")
class TestFixedEmbeddingComposite(unittest.TestCase):
    def test_keyerror(self):
        C16 = dnx.chimera_graph(16)
        child_sampler = dimod.RandomSampler()
        nodelist = sorted(C16.nodes)
        edgelist = sorted(sorted(edge) for edge in C16.edges)
        struc_sampler = dimod.StructureComposite(child_sampler, nodelist, edgelist)

        embedding = {0: [391, 386], 1: [390], 2: [385]}

        sampler = FixedEmbeddingComposite(struc_sampler, embedding)

        with self.assertRaises(ValueError):
            sampler.sample_qubo({(1, 4): 1})

        sampler.sample_qubo({(1, 2): 1}).record  # sample and resolve future
        sampler.sample_qubo({(1, 1): 1}).record  # sample and resolve future
