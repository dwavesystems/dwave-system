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

import io
import concurrent.futures
import unittest

from dwave.optimization import Model
from dwave.cloud.client import Client
from dwave.cloud.computation import Future
from dwave.cloud.solver import StructuredSolver, NLSolver
from dwave.cloud.testing.mocks import qpu_pegasus_solver_data, hybrid_nl_solver_data

from dwave.system import LeapHybridNLSampler


class TestNLSampler(unittest.TestCase):

    class mock_client_factory:
        @classmethod
        def from_config(cls, **kwargs):
            # keep instantiation local, so we can later mock BaseUnstructuredSolver
            mock_nl_solver = NLSolver(client=None, data=hybrid_nl_solver_data())
            mock_qpu_solver = StructuredSolver(client=None, data=qpu_pegasus_solver_data(2))
            kwargs.setdefault('endpoint', 'mock')
            kwargs.setdefault('token', 'mock')
            client = Client(**kwargs)
            client._fetch_solvers = lambda **kwargs: [mock_qpu_solver, mock_nl_solver]
            return client

    @unittest.mock.patch('dwave.system.samplers.leap_hybrid_sampler.Client', mock_client_factory)
    def test_solver_selection(self):
        sampler = LeapHybridNLSampler()
        self.assertIn("nl", sampler.properties.get('supported_problem_types', []))

    @unittest.mock.patch('dwave.system.samplers.leap_hybrid_sampler.Client', mock_client_factory)
    @unittest.mock.patch('dwave.cloud.solver.BaseUnstructuredSolver.sample_problem')
    @unittest.mock.patch('dwave.cloud.solver.NLSolver.upload_nlm')
    @unittest.mock.patch('dwave.cloud.solver.NLSolver.decode_response')
    def test_state_resolve(self, decode_response, upload_nlm, base_sample_problem):
        sampler = LeapHybridNLSampler()

        # create model
        model = Model()
        x = model.list(10)
        model.minimize(x.sum())
        num_states = 5
        model.states.resize(num_states)
        model.lock()

        # save states
        self.assertEqual(model.states.size(), num_states)
        states_file = io.BytesIO()
        model.states.into_file(states_file)
        states_file.seek(0)

        # reset states
        # (mock upload a smaller number of states)
        model.states.resize(2)
        self.assertEqual(model.states.size(), 2)

        # upload is tested in dwave-cloud-client, we can just mock it here
        mock_problem_data_id = '123'
        mock_timing_info = {'qpu_access_time': 1, 'run_time': 2}

        # note: instead of simply mocking `sampler.solver`, we mock a set of
        # solver methods minimally required to fully test `NLSolver.sample_problem`

        upload_nlm.return_value.result.return_value = mock_problem_data_id

        base_sample_problem.return_value = Future(solver=sampler.solver, id_="x")
        base_sample_problem.return_value._set_message({"answer": {}})

        def mock_decode_response(msg, answer_data: io.IOBase):
            # copy model states to the "received" answer_data
            answer_data.write(states_file.read())
            answer_data.seek(0)
            return {
                'problem_type': 'nl',
                'timing': mock_timing_info,
                'shape': {},
                'answer': answer_data
            }
        decode_response.side_effect = mock_decode_response

        time_limit = 5

        result = sampler.sample(model, time_limit=time_limit)

        with self.subTest('low-level sample_nlm called'):
            base_sample_problem.assert_called_with(
                mock_problem_data_id, label=None, upload_params=None, time_limit=time_limit)

        with self.subTest('max_num_states is respected on upload'):
            upload_nlm.assert_called_with(
                model, max_num_states=sampler.properties['maximum_number_of_states'])

        with self.subTest('timing returned in sample future'):
            self.assertIsInstance(result, concurrent.futures.Future)
            self.assertIsInstance(result.result(), LeapHybridNLSampler.SampleResult)
            self.assertEqual(result.result().timing, mock_timing_info)

        with self.subTest('model states updated'):
            self.assertEqual(result.result().model.states.size(), num_states)
            self.assertEqual(model.states.size(), num_states)

        with self.subTest('warnings raised'):
            # add a warning to timing info (inplace)
            msg = 'solved by classical presolve techniques'
            mock_timing_info['warnings'] = [msg]

            with self.assertWarns(UserWarning, msg=msg):
                result = sampler.sample(model, time_limit=time_limit)
                result.result()
