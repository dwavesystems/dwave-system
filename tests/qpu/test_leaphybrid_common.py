# Copyright 2021 D-Wave Systems Inc.
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

import os
import json
import threading
import unittest
from functools import partial

import numpy
from packaging.specifiers import SpecifierSet
from parameterized import parameterized_class

import dimod
from dwave.cloud.exceptions import ConfigFileError, SolverNotFoundError
from dwave.cloud.package_info import __version__ as cc_version
from dwave.cloud.testing import isolated_environ

from dwave.optimization import Model
from dwave.system import (
    LeapHybridSampler, LeapHybridDQMSampler, LeapHybridCQMSampler, LeapHybridNLSampler)


@parameterized_class(
    ("problem_type", "sampler_cls"), [
        ("bqm", LeapHybridSampler),
        ("dqm", LeapHybridDQMSampler),
        ("cqm", LeapHybridCQMSampler),
    ])
@unittest.skipIf(os.getenv('SKIP_INT_TESTS'), "Skipping integration test.")
class TestLegacySolverSelection(unittest.TestCase):

    def test_baseline(self):
        # kwarg-level override should always work
        try:
            sampler = self.sampler_cls(solver=self.sampler_cls.default_solver)
        except (ValueError, ConfigFileError):
            raise unittest.SkipTest("config error")

        self.assertIn(self.problem_type, sampler.solver.supported_problem_types)

        sampler.close()

    def test_new_positive(self):
        # given valid solver spec in env, init succeeds
        solver = json.dumps(self.sampler_cls.default_solver)
        with isolated_environ(add={'DWAVE_API_SOLVER': solver},
                              remove=('DWAVE_FEATURE_FLAGS',)):
            try:
                sampler = self.sampler_cls()
            except (ValueError, ConfigFileError):
                raise unittest.SkipTest("config error")

            self.assertIn(self.problem_type, sampler.solver.supported_problem_types)

        sampler.close()

    def test_new_negative(self):
        # given invalid solver spec in env, init fails
        solver = json.dumps(dict(qpu=True))
        with isolated_environ(add={'DWAVE_API_SOLVER': solver},
                              remove=('DWAVE_FEATURE_FLAGS',)):
            with self.assertRaises(SolverNotFoundError):
                try:
                    sampler = self.sampler_cls()
                except (ValueError, ConfigFileError):
                    raise unittest.SkipTest("config error")

    def test_old(self):
        # given invalid solver spec in env, init succeeds
        solver = json.dumps(dict(qpu=True))
        flags = json.dumps(dict(hss_solver_config_override=True))
        with isolated_environ(add={'DWAVE_API_SOLVER': solver,
                                   'DWAVE_FEATURE_FLAGS': flags}):
            try:
                sampler = self.sampler_cls()
            except (ValueError, ConfigFileError):
                raise unittest.SkipTest("config error")

            self.assertIn(self.problem_type, sampler.solver.supported_problem_types)

        sampler.close()


def _get_toy_nl_model():
    # create a simple model
    model = Model()
    x = model.list(5)
    W = model.constant(numpy.arange(25).reshape((5, 5)))
    model.minimize(W[x, :][:, x].sum())
    return model


@parameterized_class(
    ("sampler_cls", "sample_meth", "resolve_meth", "result_meth", "problem_gen"), [
        (LeapHybridSampler, "sample", "resolve", lambda self, fut: fut,
         lambda self: dimod.BQM.from_qubo({})),
        (LeapHybridDQMSampler, "sample_dqm", "resolve", lambda self, fut: fut,
         lambda self: dimod.DQM.from_numpy_vectors([0], [0], ([], [], []))),
        (LeapHybridCQMSampler, "sample_cqm", "resolve", lambda self, fut: fut,
         lambda self: dimod.CQM.from_bqm(dimod.BQM.from_qubo({'ab': 1}))),
        (LeapHybridNLSampler, "sample", "result", lambda self, fut: fut.result(),
         lambda self: _get_toy_nl_model()),
    ])
@unittest.skipIf(os.getenv('SKIP_INT_TESTS'), "Skipping integration test.")
class TestSamplerInterface(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.sampler = cls.sampler_cls()
        except (ValueError, ConfigFileError, SolverNotFoundError):
            raise unittest.SkipTest(f"{cls.sampler_cls} not available")

    @classmethod
    def tearDownClass(cls):
        cls.sampler.close()

    def test_wait_id_availability(self):
        # verify https://github.com/dwavesystems/dwave-system/issues/540 is fixed
        problem = self.problem_gen()
        fut = getattr(self.sampler, self.sample_meth)(problem)
        resolve = getattr(fut, self.resolve_meth)
        result = partial(self.result_meth, fut)

        with self.subTest("wait_id() exists"):
            pid = fut.wait_id()
            self.assertIsInstance(pid, str)

        with self.subTest("wait_id() exists post-resolve"):
            resolve()
            pid_post = fut.wait_id()
            self.assertEqual(pid, pid_post)

        # verify https://github.com/dwavesystems/dwave-system/issues/602
        with self.subTest("wait_id() result equal to info.problem_id"):
            pid = fut.wait_id()
            resolve()
            self.assertEqual(pid, result().info['problem_id'])

    @unittest.skipUnless(cc_version in SpecifierSet('>=0.13.5', prereleases=True),
                         "'dwave-cloud-client>=0.13.5' required")
    def test_problem_data_id_available(self):
        problem = self.problem_gen()
        fut = getattr(self.sampler, self.sample_meth)(problem)
        result = partial(self.result_meth, fut)

        self.assertIn('problem_id', result().info)
        self.assertIn('problem_data_id', result().info)

    def test_close(self):
        n_initial = threading.active_count()
        sampler = self.sampler_cls()
        n_active = threading.active_count()
        sampler.close()
        n_closed = threading.active_count()

        with self.subTest('verify all client threads shutdown'):
            self.assertGreater(n_active, n_initial)
            self.assertEqual(n_closed, n_initial)

        try:
            # requires `dwave-cloud-client>=0.13.3`
            from dwave.cloud.exceptions import UseAfterCloseError
        except:
            pass
        else:
            with self.subTest('verify use after close disallowed'):
                with self.assertRaises(UseAfterCloseError):
                    problem = self.problem_gen()
                    getattr(sampler, self.sample_meth)(problem)

        with self.subTest('verify context manager calls close'):
            n_initial = threading.active_count()
            with self.sampler_cls():
                n_active = threading.active_count()
            n_closed = threading.active_count()

            self.assertGreater(n_active, n_initial)
            self.assertEqual(n_closed, n_initial)
