# Copyright 2022 D-Wave Systems Inc.
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
import unittest.mock

import dimod

from dwave.system import LeapHybridCQMSampler


class TestTimeLimit(unittest.TestCase):
    class MockClient:
        @classmethod
        def from_config(cls, *args, **kwargs):
            return cls()

        def get_solver(self, *args, **kwargs):
            class MockSolver:
                properties = dict(
                    category='hybrid',
                    minimum_time_limit_s=0,
                    maximum_number_of_constraints=100,
                    maximum_number_of_variables=500,
                    maximum_number_of_biases=2000,
                    maximum_number_of_quadratic_variables=200,
                    )
                supported_problem_types = ['cqm']

                def sample_cqm(self, cqm, time_limit):
                    # return the time_limit rather than a sampleset
                    ret = unittest.mock.Mock()
                    ret.sampleset = time_limit
                    return ret

                def upload_problem(self, *args, **kwargs):
                    class MockResult:
                        @staticmethod
                        def result():
                            return

                    return MockResult

            return MockSolver()

    @unittest.mock.patch('dwave.system.samplers.leap_hybrid_sampler.Client', MockClient)
    def test_time_limit_serialization(self):
        """Check that the same time_limit is generated for the model before and
        after serialization.

        See https://github.com/dwavesystems/dwave-system/issues/482
        """
        sampler = LeapHybridCQMSampler()

        cqm = dimod.ConstrainedQuadraticModel()
        cqm.add_variables('INTEGER', 500)
        cqm.add_constraint([(0, 1, 1)], '==', 0)

        with cqm.to_file() as f:
            new = dimod.ConstrainedQuadraticModel().from_file(f)

        self.assertEqual(sampler.sample_cqm(new), sampler.sample_cqm(cqm))

    @unittest.mock.patch('dwave.system.samplers.leap_hybrid_sampler.Client')
    def test_close(self, mock_client):
        mock_solver = mock_client.from_config.return_value.get_solver.return_value
        mock_solver.properties = {'category': 'hybrid'}
        mock_solver.supported_problem_types = ['cqm']

        with self.subTest('manual close'):
            sampler = LeapHybridCQMSampler()
            sampler.close()
            mock_client.from_config.return_value.close.assert_called_once()

        mock_client.reset_mock()

        with self.subTest('context manager'):
            with LeapHybridCQMSampler():
                ...
            mock_client.from_config.return_value.close.assert_called_once()
