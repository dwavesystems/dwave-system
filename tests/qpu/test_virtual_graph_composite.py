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

import os
import itertools
import unittest

import minorminer
import dimod.testing as dtest

from dwave.cloud.exceptions import ConfigFileError, SolverNotFoundError

from dwave.system.composites import VirtualGraphComposite
from dwave.system.samplers import DWaveSampler


@unittest.skipIf(os.getenv('SKIP_INT_TESTS'), "Skipping integration test.")
class TestVirtualGraphComposite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.qpu = DWaveSampler(solver=dict(flux_biases=True))
        except (ValueError, ConfigFileError, SolverNotFoundError):
            raise unittest.SkipTest("no qpu available")

    @classmethod
    def tearDownClass(cls):
        cls.qpu.client.close()

    def test_construction(self):
        child_sampler = self.qpu

        # get an embedding
        K10_edges = list(itertools.combinations(range(10), 2))
        embedding = minorminer.find_embedding(K10_edges, child_sampler.edgelist)

        sampler = VirtualGraphComposite(child_sampler, embedding,
                                        flux_bias_num_reads=10)

        dtest.assert_sampler_api(sampler)

        h = {}
        J = {edge: -1 for edge in K10_edges}

        # run with fbo
        sampler.sample_ising(h, J).resolve()

        # and again without
        sampler.sample_ising(h, J, apply_flux_bias_offsets=False).resolve()
