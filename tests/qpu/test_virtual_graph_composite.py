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
import itertools
import unittest

import minorminer
import dimod.testing as dtest

from dwave.cloud.exceptions import ConfigFileError

from dwave.system.composites import VirtualGraphComposite
from dwave.system.samplers import DWaveSampler

try:
    DWaveSampler()
    _config_found = True
except (ValueError, ConfigFileError):
    _config_found = False


@unittest.skipUnless(_config_found, "no configuration found to connect to a system")
class TestVirtualGraphComposite(unittest.TestCase):
    def test_construction(self):
        child_sampler = DWaveSampler()

        # get an embedding
        K10_edges = list(itertools.combinations(range(10), 2))
        embedding = minorminer.find_embedding(K10_edges, child_sampler.edgelist)

        sampler = VirtualGraphComposite(child_sampler, embedding)

        dtest.assert_sampler_api(sampler)

        h = {}
        J = {edge: -1 for edge in K10_edges}
        sampler.sample_ising(h, J)

        sampler.sample_ising(h, J, apply_flux_bias_offsets=False)

    def test_reverse_annealing(self):
        child_sampler = DWaveSampler()

        # get an embedding
        K10_edges = list(itertools.combinations(range(10), 2))
        embedding = minorminer.find_embedding(K10_edges, child_sampler.edgelist)

        sampler = VirtualGraphComposite(child_sampler, embedding)

        h = {}
        J = {edge: -1 for edge in K10_edges}

        kwargs = {'initial_state': {v: 0 for v in set().union(*J)},
                  'anneal_schedule': [(0, 1), (55.0, 0.45), (155.0, 0.45), (210.0, 1)]}

        # sample and resolve
        sampler.sample_ising(h, J, **kwargs).samples()
