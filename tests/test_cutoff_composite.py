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
#
# ================================================================================================
import unittest

import dimod.testing as dtest
from dimod import BinaryQuadraticModel, Sampler, ExactSolver, \
    HigherOrderComposite

from dwave.system.composites.cutoffcomposite import CutOffComposite, \
    cutoff_ising, cutoff_bqm


class CutoffChecker(Sampler):
    def __init__(self, child_sampler, bqm=None, h=None, J=None,
                 offset=None, cutoff=None, **other_params):

        self.child = child_sampler

        if bqm is not None:
            if cutoff is None:
                self.bqm = bqm.copy()
            else:
                self.bqm, removed = cutoff_bqm(bqm, cutoff)

        elif h is not None and J is not None:
            if cutoff is None:
                self.h = dict(h)
                self.J = dict(J)
            else:
                if max(map(len, J.keys())) == 2:
                    bqm = BinaryQuadraticModel.from_ising(h, J, offset=offset)
                    self.bqm, removed = cutoff_bqm(bqm, cutoff)

                else:
                    h_sc, J_sc, removed = cutoff_ising(h, J, cutoff)
                    self.h = h_sc
                    self.J = J_sc

    def sample(self, bqm, **parameters):
        assert self.bqm == bqm
        return self.child.sample(bqm, **parameters)

    def sample_ising(self, h, J, offset=0, **parameters):
        assert self.h == h
        assert self.J == J
        return self.child.sample_ising(h, J, offset=offset, **parameters)

    def parameters(self):
        return self.child.parameters()

    def properties(self):
        return self.child.properties()


class TestCutoffIsing(unittest.TestCase):

    def test_instantiation_smoketest(self):
        sampler = CutOffComposite(ExactSolver())
        dtest.assert_sampler_api(sampler)

    def test_cutoff_none(self):
        linear = {'a': -4.0, 'b': -4.0}
        quadratic = {('a', 'b', 'c'): 3.2, ('b', 'd'): 0.1}
        offset = 2
        cutoff = None
        sampler = CutOffComposite(
            CutoffChecker(HigherOrderComposite(ExactSolver()),
                          h=linear,
                          J=quadratic,
                          cutoff=cutoff))

        response = sampler.sample_ising(linear, quadratic, offset=offset,
                                        cutoff=cutoff)
        self.assertEqual(set(response.variables), {'a', 'b', 'c', 'd'})
        self.assertAlmostEqual(response.first.energy, -9.3)
        self.assertEqual(response.info.get('cutoff_resolved', 0), 0)

    def test_cutoff(self):
        linear = {'a': -4.0, 'b': -4.0}
        quadratic = {('a', 'b', 'c'): 3.2, ('b', 'd'): 0.1}
        offset = 2
        cutoff = 1.0
        sampler = CutOffComposite(
            CutoffChecker(HigherOrderComposite(ExactSolver()),
                          h=linear,
                          J=quadratic,
                          cutoff=cutoff))
        response = sampler.sample_ising(linear, quadratic, offset=offset,
                                        cutoff=cutoff)
        self.assertEqual(set(response.variables), {'a', 'b', 'c', 'd'})
        self.assertAlmostEqual(response.first.energy, -9.3)
        self.assertEqual(set(response.info['cutoff_resolved']), {'d'})

    def test_weak_cutoff(self):
        linear = {'a': -4.0, 'b': -4.0}
        quadratic = {('a', 'b', 'c'): 3.2, ('b', 'd'): 0.1}
        offset = 2
        cutoff = 0.01
        sampler = CutOffComposite(
            CutoffChecker(HigherOrderComposite(ExactSolver()),
                          h=linear,
                          J=quadratic,
                          cutoff=cutoff))

        response = sampler.sample_ising(linear, quadratic, offset=offset,
                                        cutoff=cutoff)
        self.assertEqual(set(response.variables), {'a', 'b', 'c', 'd'})
        self.assertAlmostEqual(response.first.energy, -9.3)
        self.assertEqual(len(response.info['cutoff_resolved']), 0)

    def test_qubo(self):
        linear = {'a': -4.0, 'b': -4.0}
        quadratic = {('a', 'b'): 3.2, ('b', 'd'): 0.1}
        offset = 0
        cutoff = 1.0
        sampler = CutOffComposite(HigherOrderComposite(ExactSolver()))

        response = sampler.sample_ising(linear, quadratic, offset=offset,
                                        cutoff=cutoff)

        self.assertEqual(set(response.variables), {'a', 'b', 'd'})
        self.assertAlmostEqual(response.first.energy, -4.9)
        self.assertEqual(response.first.penalty_satisfaction, 1)
        self.assertEqual(set(response.info['cutoff_resolved']), {'d'})


class TestCutoffBqm(unittest.TestCase):

    def test_cutoff_none(self):
        linear = {'a': -4.0, 'b': -4.0}
        quadratic = {('a', 'b'): 3.2, ('b', 'd'): 0.1}
        offset = 2
        bqm = BinaryQuadraticModel.from_ising(linear, quadratic, offset=offset)
        cutoff = None
        sampler = CutOffComposite(
            CutoffChecker(HigherOrderComposite(ExactSolver()),
                          bqm=bqm, cutoff=cutoff))
        response = sampler.sample(bqm, cutoff=cutoff)

        self.assertEqual(set(response.variables), {'a', 'b', 'd'})
        self.assertAlmostEqual(response.first.energy, -2.9)
        self.assertEqual(response.info.get('cutoff_resolved', 0), 0)

    def test_cutoff(self):
        linear = {'a': -4.0, 'b': -4.0}
        quadratic = {('a', 'b'): 3.2, ('b', 'd'): 0.1}
        offset = 2
        bqm = BinaryQuadraticModel.from_ising(linear, quadratic, offset=offset)
        cutoff = 1.0
        sampler = CutOffComposite(
            CutoffChecker(HigherOrderComposite(ExactSolver()),
                          bqm=bqm, cutoff=cutoff))
        response = sampler.sample(bqm, cutoff=cutoff)

        self.assertEqual(set(response.variables), {'a', 'b', 'd'})
        self.assertAlmostEqual(response.first.energy, -2.9)
        self.assertEqual(set(response.info['cutoff_resolved']), {'d'})

    def test_weak_cutoff(self):
        linear = {'a': -4.0, 'b': -4.0}
        quadratic = {('a', 'b'): 3.2, ('b', 'd'): 0.1}
        offset = 2
        bqm = BinaryQuadraticModel.from_ising(linear, quadratic, offset=offset)
        cutoff = 0.01
        sampler = CutOffComposite(
            CutoffChecker(HigherOrderComposite(ExactSolver()),
                          bqm=bqm, cutoff=cutoff))
        response = sampler.sample(bqm, cutoff=cutoff)

        self.assertEqual(set(response.variables), {'a', 'b', 'd'})
        self.assertAlmostEqual(response.first.energy, -2.9)
        self.assertEqual(len(response.info['cutoff_resolved']), 0)
