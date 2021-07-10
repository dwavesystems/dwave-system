# Copyright 2020 D-Wave Systems Inc.
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

import dimod

from dwave.cloud.exceptions import ConfigFileError, SolverNotFoundError
from dwave.system import LeapHybridCQMSampler

try:
    sampler = LeapHybridCQMSampler()
except (ValueError, ConfigFileError, SolverNotFoundError):
    sampler = None


@unittest.skipIf(os.getenv('SKIP_INT_TESTS'), "Skipping integration test.")
@unittest.skipIf(sampler is None, "no hybrid sampler available")
class TestLeapHybridSampler(unittest.TestCase):

    def assertConsistent(self, cqm, sampleset):
        self.assertEqual(cqm.variables, sampleset.variables)

        # todo: check the vartypes etc

    def test_small(self):
        # submit 3-variable CQMs of binary, spin, integer and mixed
        objectives = dict()

        objectives['binary'] = dimod.BQM({'a': 1, 'b': 1, 'c': 1}, {}, 0, 'BINARY')
        objectives['spin'] = dimod.BQM({'a': 1, 'b': 1, 'c': 1}, {}, 0, 'SPIN')

        objectives['integer'] = integer = dimod.QM()
        integer.add_variables_from('INTEGER', 'abc')

        objectives['mixed'] = mixed = dimod.QM()
        mixed.add_variable('BINARY', 'a')
        mixed.add_variable('SPIN', 'b')
        mixed.add_variable('INTEGER', 'c')

        for vartype, model in objectives.items():
            with self.subTest(f"one constraint, vartype={vartype}"):
                cqm = dimod.ConstrainedQuadraticModel()
                cqm.set_objective(model)
                cqm.add_constraint(model, rhs=1, sense='==')
                sampleset = sampler.sample_cqm(cqm)
                self.assertConsistent(cqm, sampleset)

            with self.subTest(f"no constraints, vartype={vartype}"):
                cqm = dimod.ConstrainedQuadraticModel()
                cqm.set_objective(model)
                sampleset = sampler.sample_cqm(cqm)
                self.assertConsistent(cqm, sampleset)

    def test_large(self):
        # submit large CQMs of binary, spin, integer and mixed
        num_variables = 5000

        objectives = dict()

        objectives['binary'] = dimod.BQM(num_variables, 'BINARY')
        objectives['spin'] = dimod.BQM(num_variables, 'SPIN')

        objectives['integer'] = integer = dimod.QM()
        integer.add_variables_from('INTEGER', range(num_variables))

        objectives['mixed'] = mixed = dimod.QM()
        mixed.add_variables_from('INTEGER', range(num_variables // 3))
        mixed.add_variables_from(
            'SPIN', range(mixed.num_variables, mixed.num_variables + (num_variables // 3)))
        mixed.add_variables_from('BINARY', range(mixed.num_variables, num_variables))

        for vartype, model in objectives.items():
            with self.subTest(f"one constraint, vartype={vartype}"):
                cqm = dimod.ConstrainedQuadraticModel()
                cqm.set_objective(model)
                cqm.add_constraint(model, rhs=1, sense='==')
                sampleset = sampler.sample_cqm(cqm)
                self.assertConsistent(cqm, sampleset)

            with self.subTest(f"no constraints, vartype={vartype}"):
                cqm = dimod.ConstrainedQuadraticModel()
                cqm.set_objective(model)
                sampleset = sampler.sample_cqm(cqm)
                self.assertConsistent(cqm, sampleset)

    def test_constraints(self):
        num_constraints = 100000

        x = dimod.Binary('x')

        cqm = dimod.ConstrainedQuadraticModel()
        for _ in range(num_constraints):
            cqm.add_constraint(x == 1)

        sampler.sample_cqm(cqm).resolve()  # smoke test
