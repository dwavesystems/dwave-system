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

import os
import unittest

import numpy as np

import dwave_networkx as dnx

from dwave.cloud.testing import isolated_environ

from dwave.system import (anneal_schedule_with_offset, common_working_graph,
    energy_scales_custom_schedule)
from dwave.system.utilities import FeatureFlags


class TestCommonWorkingGraph(unittest.TestCase):
    def test_single_tile(self):

        G1 = dnx.chimera_graph(1)
        with self.assertWarns(DeprecationWarning):
            G = common_working_graph(G1, G1)

        # should have 8 nodes
        self.assertEqual(len(G), 8)

        # nodes 0,...,7 should be in the tile
        for n in range(8):
            self.assertIn(n, G)

        # check bipartite
        for i in range(4):
            for j in range(4, 8):
                self.assertTrue((i, j) in G.edges() or (j, i) in G.edges())

    def test_c1_c2_tiles(self):
        G1 = dnx.chimera_graph(1)
        G2 = dnx.chimera_graph(2)

        with self.assertWarns(DeprecationWarning):
            G = common_working_graph(G1, G1)

        self.assertEqual(len(G), 8)

    def test_missing_node(self):
        G1 = dnx.chimera_graph(1)
        G1.remove_node(2)
        G2 = dnx.chimera_graph(2)

        with self.assertWarns(DeprecationWarning):
            G = common_working_graph(G1, G1)

        self.assertNotIn(2, G)
        self.assertNotIn((2, 4), G.edges())

    def test_sampler_adjacency(self):
        adj = {0: {1, 2}, 1: {2}, 2: {0, 1}}
        G = dnx.chimera_graph(1)

        with self.assertWarns(DeprecationWarning):
            H = common_working_graph(adj, G)

        self.assertEqual(set(H.nodes), {0, 1, 2})
        self.assertEqual(set(H.edges), set())


class TestFeatureFlagSupport(unittest.TestCase):
    def test_base(self):
        with isolated_environ(add={'DWAVE_FEATURE_FLAGS': '{"x": true}'}):
            self.assertTrue(FeatureFlags.get('x'))
        with isolated_environ(add={'DWAVE_FEATURE_FLAGS': '{"x": false}'}):
            self.assertFalse(FeatureFlags.get('x'))
        with isolated_environ(add={'DWAVE_FEATURE_FLAGS': '{"x": 0}'}):
            self.assertFalse(FeatureFlags.get('x'))
        with isolated_environ(add={'DWAVE_FEATURE_FLAGS': '{"x": 1}'}):
            self.assertTrue(FeatureFlags.get('x'))
        with isolated_environ(add={'DWAVE_FEATURE_FLAGS': '{"x": error}'}):
            self.assertFalse(FeatureFlags.get('x'))
        with isolated_environ(add={'DWAVE_FEATURE_FLAGS': '{"y": true}'}):
            self.assertFalse(FeatureFlags.get('x'))
        with isolated_environ(add={'DWAVE_FEATURE_FLAGS': ''}):
            self.assertFalse(FeatureFlags.get('x'))
        with isolated_environ(remove=('DWAVE_FEATURE_FLAGS',)):
            self.assertFalse(FeatureFlags.get('x'))

    def test_hss_solver_config_override(self):
        with isolated_environ(add={'DWAVE_FEATURE_FLAGS': '{"hss_solver_config_override": true}'}):
            self.assertTrue(FeatureFlags.hss_solver_config_override)
        with isolated_environ(remove=('DWAVE_FEATURE_FLAGS',)):
            self.assertFalse(FeatureFlags.hss_solver_config_override)


class TestAnnealScheduleWithOffset(unittest.TestCase):
    def test_doc_file(self):

        anneal_schedule = [
            [0.1, 10, 1, 0.02],
            [0.2, 6, 3, 0.25],
            [0.3, 4, 7, 0.34],
            [0.4, 2, 12, 0.399]
            ]

        offset=0.05

        # Schedule with  offset=0.05
        expected_schedule = [
            [0.1, 9.13043478, 1.43478261, 0.07],
            [0.2, 4.88888889, 5.22222222, 0.3],
            [0.3, 2.30508475, 11.23728814, 0.39],
            [0.4, 2, 12, 0.449]
            ]

        # Anneal schedule as a list
        schedule_offset = anneal_schedule_with_offset(
            offset,
            anneal_schedule,
            )

        np.testing.assert_allclose(schedule_offset,
                                   np.asarray(expected_schedule),
                                   atol=0.1,
                                   )

        # Anneal schedule as an array
        schedule_offset = anneal_schedule_with_offset(
            offset,
            np.asarray(anneal_schedule),
            )

        np.testing.assert_allclose(schedule_offset,
                                   np.asarray(expected_schedule),
                                   atol=0.1,
                                   )

        # Vector inputs as lists
        schedule_offset = anneal_schedule_with_offset(
            offset,
            s= [a[:1][0] for a in anneal_schedule],
            A = [a[1:2][0] for a in anneal_schedule],
            B = [a[2:3][0] for a in anneal_schedule],
            c = [a[3:][0] for a in anneal_schedule],
            )

        np.testing.assert_allclose(schedule_offset,
                                   np.asarray(expected_schedule),
                                   atol=0.1,
                                   )

        # Vector inputs as 1D arrays
        schedule_offset = anneal_schedule_with_offset(
            offset,
            s= np.asarray(anneal_schedule)[:,0],
            A = np.asarray(anneal_schedule)[:,1],
            B = np.asarray(anneal_schedule)[:,2],
            c = np.asarray(anneal_schedule)[:,3],
            )

        np.testing.assert_allclose(schedule_offset,
                                   np.asarray(expected_schedule),
                                   atol=0.1,
                                   )

        # Error case: missing vector
        with self.assertRaises(ValueError):
            schedule_offset = anneal_schedule_with_offset(
                offset,
                s= np.asarray(anneal_schedule)[:,0],
                B = np.asarray(anneal_schedule)[:,2],
                c = np.asarray(anneal_schedule)[:,3],
                )

        # Error case: schedule and vectors
        with self.assertRaises(ValueError):
            schedule_offset = anneal_schedule_with_offset(
                offset,
                anneal_schedule,
                s= np.asarray(anneal_schedule)[:,0],
                A = np.asarray(anneal_schedule)[:,1],
                B = np.asarray(anneal_schedule)[:,2],
                c = np.asarray(anneal_schedule)[:,3],
                )

        # Error case: schedule not 4D
        with self.assertRaises(ValueError):
            schedule_offset = anneal_schedule_with_offset(
                offset,
                anneal_schedule=np.asarray(anneal_schedule)[:,0],
                )

        # Error case: vector not 1D
        with self.assertRaises(ValueError):
            schedule_offset = anneal_schedule_with_offset(
                offset,
                s= np.asarray(anneal_schedule)[:,0],
                A = np.asarray(anneal_schedule)[:,1:3],
                B = np.asarray(anneal_schedule)[:,2],
                c = np.asarray(anneal_schedule)[:,3],
                )


class TestEnergyScalesCustomSchedule(unittest.TestCase):

    schedule_default = [
        [ 0. ,  9.3,  0. ,  0.1],
        [ 0.1,  9. ,  0.5,  0.2],
        [ 0.2,  8. ,  1.2,  0.4],
        [ 0.3,  7. ,  1.7,  0.5],
        [ 0.4,  6. ,  3. ,  0.7],
        [ 0.5,  5. ,  5. ,  0.8],
        [ 0.6,  4. ,  7. ,  1. ],
        [ 0.7,  3. ,  9. ,  2. ],
        [ 0.8,  2. , 12. ,  3. ],
        [ 0.9,  1. , 14. ,  4. ],
        [ 1. ,  0. , 16. ,  8. ]]

    def test_input_verification(self):

        simple_forward_schedule = [
            [0, 0],
            [20, 0.3],
            [100, 1]]

        # Schedules for both, as lists
        out= energy_scales_custom_schedule(
            default_schedule=self.schedule_default,
            custom_schedule=simple_forward_schedule)
        self.assertEqual(len(out), len(self.schedule_default))

        # Schedule for one, vectors for the other, as lists
        out= energy_scales_custom_schedule(
            s=[s/10 for s in range(11)],
            A=[a for a in range(10,-1,-1)],
            B=[b for b in range(11)],
            c=[c/8 for c in range(11)],
            custom_schedule=simple_forward_schedule)

        # Numpy schedule and mix of Numpy and list for vectors
        out= energy_scales_custom_schedule(
            s=[s/10 for s in range(11)],
            A=np.asarray([a for a in range(10,-1,-1)]),
            B=[b for b in range(11)],
            c=np.asarray([c/8 for c in range(11)]),
            custom_schedule=np.asarray(simple_forward_schedule))

        # Both schedule and vector
        with self.assertRaises(ValueError):
            energy_scales_custom_schedule(
                default_schedule=self.schedule_default,
                s=[s/10 for s in range(11)],
                custom_schedule=simple_forward_schedule)

        # Both schedule and vector
        with self.assertRaises(ValueError):
            energy_scales_custom_schedule(
                default_schedule=self.schedule_default,
                s=[s/10 for s in range(11)],
                custom_schedule=simple_forward_schedule)

        # Missing schedule
        with self.assertRaises(ValueError):
            energy_scales_custom_schedule(
                default_schedule=self.schedule_default)

        # Missing vector
        with self.assertRaises(ValueError):
            energy_scales_custom_schedule(
                s=[s/10 for s in range(11)],
                A=np.asarray([a for a in range(10,-1,-1)]),
                B=[b for b in range(11)],
                custom_schedule=simple_forward_schedule)

    def test_various_schedules(self):

        # Simple forward schedule that coincides with default schedule's s intervals
        schedule = [
            [0, 0],
            [20, 0.3],
            [100, 1]]
        out= energy_scales_custom_schedule(
            default_schedule=self.schedule_default,
            custom_schedule=schedule)
        self.assertEqual(np.shape(out), (11,5))

        # Simple forward schedule with noisy s intervals
        schedule = np.asarray([
            [0, 0],
            [20, 0.31],
            [50, 0.69],
            [80, 0.9],
            [100, 1]])
        out= energy_scales_custom_schedule(
            default_schedule=self.schedule_default,
            custom_schedule=schedule)
        self.assertEqual(np.shape(out), (11,5))

        # Test that at the interval seams, the interpolated time is closest
        np.testing.assert_equal(
            [np.argmin(np.abs(out[:,1] - s)) for s in np.asarray(schedule)[:,1]],
            [np.argmin(np.abs(out[:,0] - t)) for t in np.asarray(schedule)[:,0]])

        # Forward schedule with pauses
        schedule = np.asarray([
            [0, 0],
            [20, 0.3],
            [50, 0.3],
            [80, 0.8],
            [90, 0.8],
            [100, 0.8],
            [200, 1]])
        out= energy_scales_custom_schedule(
            default_schedule=self.schedule_default,
            custom_schedule=schedule)
        self.assertEqual(np.shape(out), (11 + 3,5))    #Three pauses

        # Forward schedule with pauses and noisy intervals
        schedule = np.asarray([
            [0, 0],
            [20, 0.31],
            [50, 0.31],
            [80, 0.78],
            [90, 0.78],
            [100, 0.78],
            [200, 1]])
        out= energy_scales_custom_schedule(
            default_schedule=self.schedule_default,
            custom_schedule=schedule)
        self.assertEqual(np.shape(out), (11 + 3,5))    #Three pauses
        # Where interpolated time is closest to custom time, custom s is closest to s
        min_t_inx = [np.argmin(np.abs(out[:,0] - t)) for t in np.asarray(schedule)[:,0]]
        np.testing.assert_allclose(
            out[min_t_inx][:,1],
            schedule[:,1],
            atol=0.1)

        # Simple reverse schedule that coincides with default schedule's s intervals
        schedule = [
            [0, 1],
            [20, 0.3],
            [100, 1]]
        out= energy_scales_custom_schedule(
            default_schedule=self.schedule_default,
            custom_schedule=schedule)
        self.assertEqual(np.shape(out), (2*0.7*10 + 1,5))

        # Messy reverse anneal schedule
        t_expected = np.asarray(
            [0, 6.66, 13.33, 20, 35, 50, 80, 83.33, 86.6, 90, 100, 150, 170, 200])
        B_expected = np.asarray(
            [16., 14., 12., 9.,  7.,  5.,  5.,  7.,  9., 12., 12., 12., 14., 16.])
        schedule = np.asarray([
            [0, 1],
            [20, 0.7],
            [50, 0.51],
            [80, 0.51],
            [90, 0.78],
            [100, 0.78],
            [150, 0.78],
            [170, 0.9],
            [200, 1]])
        out= energy_scales_custom_schedule(
            default_schedule=self.schedule_default,
            custom_schedule=schedule)
        np.testing.assert_allclose(out[:,0], t_expected, atol=1)
        np.testing.assert_allclose(out[:,3], B_expected, atol=1)