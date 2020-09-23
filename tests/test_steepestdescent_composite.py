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

import unittest
from collections import Mapping

import dimod
import numpy as np

from dwave.system.testing import MockDWaveSampler, mock
from dwave.system.composites import SteepestDescentComposite


@dimod.testing.load_sampler_bqm_tests(SteepestDescentComposite(dimod.ExactSolver()))
class TestSteepestDescendComposite(unittest.TestCase):
    def test_instantiation_smoketest(self):
        sampler = SteepestDescentComposite(MockDWaveSampler())

        dimod.testing.assert_sampler_api(sampler)

    def test_sample_ising(self):
        sampler = SteepestDescentComposite(MockDWaveSampler())

        h = {0: -1., 4: 2}
        J = {(0, 4): 1.5}
        num_reads = 100

        response = sampler.sample_ising(h, J, num_reads=num_reads)

        self.assertEqual(len(response), num_reads)

        for sample in response.samples():
            self.assertIsInstance(sample, Mapping)
            self.assertEqual(set(sample), set(h))

        for sample, energy in response.data(['sample', 'energy']):
            self.assertIsInstance(sample, Mapping)
            self.assertEqual(set(sample), set(h))
            self.assertAlmostEqual(dimod.ising_energy(sample, h, J), energy)

    def test_convex(self):
        # a convex section of hyperbolic paraboloid in the Ising space,
        # with global minimum at (-1,-1)
        bqm = dimod.BQM.from_ising({0: 2, 1: 2}, {(0, 1): -1})
        ground = dimod.SampleSet.from_samples_bqm({0: -1, 1: -1}, bqm)

        sampler = SteepestDescentComposite(dimod.ExactSolver())
        sampleset = sampler.sample(bqm).aggregate()

        np.testing.assert_array_almost_equal(
            sampleset.record.sample, ground.record.sample)
