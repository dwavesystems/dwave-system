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

from parameterized import parameterized_class

import dimod
from dwave.cloud.exceptions import ConfigFileError, SolverNotFoundError
from dwave.cloud.package_info import __version__ as cc_version
from dwave.cloud.testing import isolated_environ

from dwave.system import LeapHybridSampler, LeapHybridDQMSampler, LeapHybridCQMSampler


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


@parameterized_class(
    ("sampler_cls", "sample_meth", "problem_gen"), [
        (LeapHybridSampler, "sample", lambda self: dimod.BQM.from_qubo({})),
        (LeapHybridDQMSampler, "sample_dqm", lambda self: dimod.DQM.from_numpy_vectors([0], [0], ([], [], []))),
        (LeapHybridCQMSampler, "sample_cqm", lambda self: dimod.CQM.from_bqm(dimod.BQM.from_qubo({'ab': 1}))),
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

    def test_sampleset_wait_id_availability(self):
        # verify https://github.com/dwavesystems/dwave-system/issues/540 is fixed
        problem = self.problem_gen()
        ss = getattr(self.sampler, self.sample_meth)(problem)

        with self.subTest("sampleset.wait_id() exists"):
            pid = ss.wait_id()
            self.assertIsInstance(pid, str)

        with self.subTest("sampleset.wait_id() exists post-resolve"):
            ss.resolve()
            pid_post = ss.wait_id()
            self.assertEqual(pid, pid_post)

    @unittest.skipIf(tuple(map(int, cc_version.split('.'))) < (0, 13, 5),
                     "'dwave-cloud-client>=0.13.5' required")
    def test_problem_data_id_available(self):
        problem = self.problem_gen()
        ss = getattr(self.sampler, self.sample_meth)(problem)

        self.assertIn('problem_id', ss.info)
        self.assertIn('problem_data_id', ss.info)

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
