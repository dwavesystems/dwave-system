# Copyright 2019 D-Wave Systems Inc.
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

import dimod
import dimod.testing as dtest

from dwave.system import CutOffComposite


class CutoffChecker(dimod.Sampler):
    def __init__(self, child_sampler, expected_bqm):

        self.child = child_sampler
        self.bqm = expected_bqm

    def sample(self, bqm, **parameters):
        assert self.bqm == bqm, '{} != {}'.format(self.bqm, bqm)
        return self.child.sample(bqm, **parameters)

    def parameters(self):
        return self.child.parameters()

    def properties(self):
        return self.child.properties()


class TestConstruction(unittest.TestCase):
    def test_instantiation_smoketest(self):
        sampler = CutOffComposite(dimod.ExactSolver(), 0)
        dtest.assert_sampler_api(sampler)


class TestCutoffIsing(unittest.TestCase):
    def test_no_cutoff(self):
        h = {'a': -4.0, 'b': -4.0}
        J = {('a', 'b'): 3.2, ('b', 'c'): 0.1}
        cutoff = -1

        bqm = dimod.BinaryQuadraticModel.from_ising(h, J)

        checker = CutoffChecker(dimod.ExactSolver(), bqm)
        samples = CutOffComposite(checker, cutoff).sample_ising(h, J)

        dimod.testing.assert_response_energies(samples, bqm)

    def test_triange_to_3path(self):
        h = {}
        J = {'ab': -1, 'bc': -1, 'ac': .5}
        cutoff = .75
        cut = dimod.BinaryQuadraticModel.from_ising({}, {'ab': -1, 'bc': -1})

        checker = CutoffChecker(dimod.ExactSolver(), cut)
        samples = CutOffComposite(checker, cutoff).sample_ising(h, J)

        dimod.testing.assert_response_energies(samples, dimod.BinaryQuadraticModel.from_ising(h, J))

    def test_triangle_to_2path(self):
        h = {'a': 1.2, 'b': 1, 'c': .5}
        J = {'ab': -1, 'bc': -.5, 'ac': -.5}
        cutoff = .75
        cut = dimod.BinaryQuadraticModel.from_ising({'a': 1.2, 'b': 1}, {'ab': -1})

        checker = CutoffChecker(dimod.ExactSolver(), cut)
        samples = CutOffComposite(checker, cutoff).sample_ising(h, J)

        dimod.testing.assert_response_energies(samples, dimod.BinaryQuadraticModel.from_ising(h, J))

        # check that we picked the right minimizing value for the isolated
        self.assertEqual(samples.first.sample['c'], -1)

    def test_triangle_to_empty(self):
        h = {'a': 1.2, 'b': 1, 'c': .5}
        J = {'ab': -.5, 'bc': -.5, 'ac': -.5}
        cutoff = .75
        cut = dimod.BinaryQuadraticModel.from_ising({}, {})

        # we cannot check in this case because all variables are isolated
        # this results in exactly one variable being sent to ExactSolver and
        # we don't know which one it will be, so we just check the correctness
        # of the output
        samples = CutOffComposite(dimod.ExactSolver(), cutoff).sample_ising(h, J)

        dimod.testing.assert_response_energies(samples, dimod.BinaryQuadraticModel.from_ising(h, J))

        # check that we picked the right minimizing value for the isolated
        self.assertEqual(samples.first.sample['a'], -1)
        self.assertEqual(samples.first.sample['b'], -1)
        self.assertEqual(samples.first.sample['c'], -1)

    def test_4_path_isolated_tail(self):
        h = {}
        J = {'ab': -1, 'bc': -.5, 'cd': -.5, 'de': -.5}
        cutoff = .75
        cut = dimod.BinaryQuadraticModel.from_ising({}, {'ab': -1})

        checker = CutoffChecker(dimod.ExactSolver(), cut)
        samples = CutOffComposite(checker, cutoff).sample_ising(h, J)

        dimod.testing.assert_response_energies(samples, dimod.BinaryQuadraticModel.from_ising(h, J))
