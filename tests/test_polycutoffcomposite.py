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

from dwave.system import PolyCutOffComposite


class CutoffChecker(dimod.PolySampler):
    def __init__(self, child_sampler, expected_poly):

        self.child = child_sampler
        self.poly = expected_poly

    def sample_poly(self, poly, **parameters):
        assert self.poly == poly, '{} != {}'.format(self.poly, poly)
        return self.child.sample_poly(poly, **parameters)

    def parameters(self):
        return self.child.parameters()

    def properties(self):
        return self.child.properties()


class TestConstruction(unittest.TestCase):
    def test_instantiation_smoketest(self):
        sampler = PolyCutOffComposite(dimod.HigherOrderComposite(dimod.ExactSolver()), 0)

        self.assertTrue(hasattr(sampler, 'sample_poly'))
        self.assertTrue(hasattr(sampler, 'sample_hising'))
        self.assertTrue(hasattr(sampler, 'sample_hubo'))

    def test_wrap_bqm(self):
        with self.assertRaises(TypeError):
            PolyCutOffComposite(dimod.ExactSolver(), -1)


class TestSampleHising(unittest.TestCase):
    def setUp(self):
        self.child = dimod.HigherOrderComposite(dimod.ExactSolver())

    def test_empty(self):
        h = {}
        J = {}
        cutoff = 1
        expected = dimod.BinaryPolynomial({}, dimod.SPIN)

        checker = CutoffChecker(self.child, expected)
        samples = PolyCutOffComposite(checker, cutoff).sample_hising(h, J)

        self.assertEqual(samples.record.sample.shape[1], 0)  # no variables

    def test_linear(self):
        # they are all isolated
        h = {'a': -1, 'b': .5}
        J = {}
        cutoff = 1

        # we cannot check in this case because all variables are isolated
        # this results in exactly one variable being sent to ExactSolver and
        # we don't know which one it will be, so we just check the correctness
        # of the output
        samples = PolyCutOffComposite(self.child, cutoff).sample_hising(h, J)

        poly = dimod.BinaryPolynomial.from_hising(h, J)
        for sample, energy in samples.data(['sample', 'energy']):
            self.assertAlmostEqual(energy, poly.energy(sample))

    def test_4_path_isolated_tail(self):
        h = {}
        J = {'ab': -1, 'bc': -.5, 'cd': -.5, 'de': -.5}
        cutoff = .75
        expected = dimod.BinaryPolynomial({'ab': -1}, dimod.SPIN)

        checker = CutoffChecker(self.child, expected)
        samples = PolyCutOffComposite(checker, cutoff).sample_hising(h, J)

        poly = dimod.BinaryPolynomial.from_hising(h, J)
        for sample, energy in samples.data(['sample', 'energy']):
            self.assertAlmostEqual(energy, poly.energy(sample))

    def test_triangle(self):
        h = {'a': -1}
        J = {'abde': -1, 'bc': -.5, 'ca': -.5}
        cutoff = .75
        expected = dimod.BinaryPolynomial({'a': -1, 'abde': -1}, dimod.SPIN)

        checker = CutoffChecker(self.child, expected)
        samples = PolyCutOffComposite(checker, cutoff).sample_hising(h, J)

        poly = dimod.BinaryPolynomial.from_hising(h, J)
        for sample, energy in samples.data(['sample', 'energy']):
            self.assertAlmostEqual(energy, poly.energy(sample))

        # 'c' was isolated, should be 1 when restored with the ground state
        self.assertEqual(samples.first.sample['c'], 1)


class TestSamplePoly(unittest.TestCase):
    def test_isolated(self):
        poly = dimod.BinaryPolynomial({'a': 3, 'abc': 4, 'ac': 0.2}, dimod.SPIN)
        sampler = dimod.HigherOrderComposite(dimod.ExactSolver())
        sampleset = PolyCutOffComposite(sampler, 4.1).sample_poly(poly)
