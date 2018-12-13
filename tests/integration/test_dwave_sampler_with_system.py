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
import itertools
import random

import numpy as np
import dimod

from dwave.cloud.exceptions import ConfigFileError

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

try:
    DWaveSampler()
    _config_found = True
except (ValueError, ConfigFileError):
    _config_found = False


@unittest.skipUnless(_config_found, "no configuration found to connect to a system")
class TestDWaveSamplerSystem(unittest.TestCase):
    def test_typical_small(self):
        h = [0, 0, 0, 0, 0]
        J = {(0, 4): 1}
        bqm = dimod.BinaryQuadraticModel.from_ising(h, J)

        response = DWaveSampler(solver={'qpu': True}).sample(bqm)

        self.assertFalse(np.any(response.samples_matrix == 0))
        self.assertIs(response.vartype, dimod.SPIN)

        rows, cols = response.samples_matrix.shape

        self.assertEqual(cols, 5)

    def test_with_software_exact_solver(self):

        sampler = DWaveSampler(solver={'software': True})

        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)

        # plant a solution

        for v in sampler.nodelist:
            bqm.add_variable(v, .001)

        for u, v in sampler.edgelist:
            bqm.add_interaction(u, v, -1)

        resp = sampler.sample(bqm, num_reads=100)

        # the ground solution should be all spin down
        ground = dict(next(iter(resp)))

        self.assertEqual(ground, {v: -1 for v in bqm})
