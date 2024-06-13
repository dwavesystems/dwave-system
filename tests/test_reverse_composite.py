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
import dwave_networkx
from dimod import ExactSolver

from dwave.system import ReverseBatchStatesComposite, ReverseAdvanceComposite

C4 = dwave_networkx.chimera_graph(4, 4, 4)


class MockReverseSampler(dimod.Sampler, dimod.Structured):
    nodelist = None
    edgelist = None
    properties = None
    parameters = None

    def __init__(self, broken_nodes=None):
        if broken_nodes is None:
            self.nodelist = sorted(C4.nodes)
            self.edgelist = sorted(sorted(edge) for edge in C4.edges)
        else:
            self.nodelist = sorted(v for v in C4.nodes if v not in broken_nodes)
            self.edgelist = sorted(sorted((u, v)) for u, v in C4.edges
                                   if u not in broken_nodes and v not in broken_nodes)

        # mark the sample kwargs
        self.parameters = parameters = {}
        parameters['num_reads'] = []
        parameters['initial_state'] = []
        parameters['reinitialize_state'] = []
        parameters['anneal_schedule'] = []

        # add the interesting properties manually
        self.properties = properties = {}
        properties['j_range'] = [-2.0, 1.0]
        properties['h_range'] = [-2.0, 2.0]
        properties['num_reads_range'] = [1, 10000]
        properties['num_qubits'] = len(C4)

    @dimod.bqm_structured
    def sample(self, bqm, **parameters):
        # we are altering the bqm
        return ExactSolver().sample(bqm)


class TestConstruction(unittest.TestCase):
    def test_instantiation_smoketest_advance(self):
        sampler = ReverseAdvanceComposite(dimod.ExactSolver())
        dtest.assert_sampler_api(sampler)

    def test_instantiation_smoketest_batch(self):
        sampler = ReverseBatchStatesComposite(dimod.ExactSolver())
        dtest.assert_sampler_api(sampler)


class TestReverseIsing(unittest.TestCase):
    def test_sample_ising_batch(self):
        sampler = ReverseBatchStatesComposite(MockReverseSampler())

        h = {0: -1., 4: 2}
        J = {(0, 4): 1.5}

        response = sampler.sample_ising(h, J, initial_states=[{0: 1, 4: 1}, {0: -1, 4: -1}, {0: 1, 4: -1}])

        # nothing failed and we got at least three responses back for each initial state
        self.assertGreaterEqual(len(response), 3)

    def test_sample_ising_advance(self):
        sampler = ReverseAdvanceComposite(MockReverseSampler())

        h = {0: -1., 4: 2}
        J = {(0, 4): 1.5}
        schedules = [[[0, 1], [1, 0.5], [2, 0.5], [3, 1]], [[0, 1], [1, 0.5], [2, 0.5], [3, 1]]]
        response = sampler.sample_ising(h, J, anneal_schedules=schedules)

        # nothing failed and we got at least two response back for each schedule point
        self.assertGreaterEqual(len(response), 2)

    def test_batch_correct_states(self):
        sampler = ReverseBatchStatesComposite(MockReverseSampler())

        h = {0: -1., 4: 2}
        J = {(0, 4): 1.5}
        initial_states = [{0: 1, 4: 1}, {0: -1, 4: -1}, {0: 1, 4: -1}]
        response = sampler.sample_ising(h, J, initial_states=initial_states)

        variables = response.variables
        initial_state_list = response.record.initial_state

        for state in initial_states:
            state = [state[var] for var in variables]
            self.assertIn(state, initial_state_list)

    def test_batch_no_initial_states(self):
        sampler = ReverseBatchStatesComposite(MockReverseSampler())
        
        h = {0: -1., 4: 2}
        J = {(0, 4): 1.5}

        # check default generates an initial state
        response = sampler.sample_ising(h, J)
        self.assertIn('initial_state', response.record.dtype.names)

    def test_batch_generate_more_initial_states(self):
        sampler = ReverseBatchStatesComposite(MockReverseSampler())
        
        h = {0: -1., 4: 2}
        J = {(0, 4): 1.5}

        num_reads = 2
        initial_states = [{0:1, 4:1}]

        response = sampler.sample_ising(h, J, initial_states=initial_states, num_reads=num_reads)
        self.assertGreaterEqual(len(response), 4)

        response = sampler.sample_ising(h, J, initial_states=initial_states, initial_states_generator='tile', num_reads=num_reads)
        self.assertEqual(len(response), 8)  

        with self.assertRaises(ValueError):
            response = sampler.sample_ising(h, J, initial_states=initial_states, initial_states_generator='none', num_reads=num_reads)

    def test_batch_truncate_initial_states(self):
        sampler = ReverseBatchStatesComposite(MockReverseSampler())
        
        h = {0: -1., 4: 2}
        J = {(0, 4): 1.5}

        num_reads = 1
        initial_states = [{0:1, 4:1}, {0:-1, 4:1}, {0:-1, 4:-1}]

        response = sampler.sample_ising(h, J, initial_states=initial_states, num_reads=num_reads)
        self.assertEqual(len(response), 4)

        response = sampler.sample_ising(h, J, initial_states=initial_states, initial_states_generator='tile', num_reads=num_reads)
        self.assertEqual(len(response), 4)

        response = sampler.sample_ising(h, J, initial_states=initial_states, initial_states_generator='none', num_reads=num_reads)
        self.assertEqual(len(response), 4)

    def test_advance_no_schedules(self):
        sampler = ReverseAdvanceComposite(MockReverseSampler())

        h = {0: -1., 4: 2}
        J = {(0, 4): 1.5}

        response = sampler.sample_ising(h, J)
        self.assertIn('schedule_index', response.record.dtype.names)

    def test_advance_correct_schedules(self):
        sampler = ReverseAdvanceComposite(MockReverseSampler())

        h = {0: -1., 4: 2}
        J = {(0, 4): 1.5}
        anneal_schedules = [[[0, 1], [1, 0.5], [2, 0.5], [3, 1]], [[0, 1], [1, 0.5], [2, 0.5], [3, 1]]]
        response = sampler.sample_ising(h, J, anneal_schedules=anneal_schedules)

        anneal_schedule_list = response.record.schedule_index

        for schedule_idx in range(len(anneal_schedules)):
            self.assertIn(schedule_idx, anneal_schedule_list)

    def test_correct_initial_state_input(self):
        sampler = ReverseAdvanceComposite(MockReverseSampler())

        h = {0: -1., 4: 2}
        J = {(0, 4): 1.5}
        anneal_schedules = [[[0, 1], [1, 0.5], [2, 0.5], [3, 1]], [[0, 1], [1, 0.5], [2, 0.5], [3, 1]]]

        with self.assertRaises(TypeError):
            response = sampler.sample_ising(h, J, anneal_schedules=anneal_schedules,
                                            initial_state=[1, 1])

    def test_correct_initial_state_used(self):
        sampler = ReverseAdvanceComposite(MockReverseSampler())

        h = {0: -1., 4: 2}
        J = {(0, 4): 1.5}
        anneal_schedules = [[[0, 1], [1, 0.5], [2, 0.5], [3, 1]], [[0, 1], [1, 0.5], [2, 0.5], [3, 1]]]
        initial = {0: -1, 4: -1}
        response = sampler.sample_ising(h, J, anneal_schedules=anneal_schedules,
                                        initial_state=initial)

        vars = response.variables
        for datum in response.data(fields=['initial_state', 'schedule_index']):
            if datum.schedule_index == 0:
                self.assertListEqual([initial[v] for v in vars], list(datum.initial_state)) # initial_state = state that was passed in
            else:
                self.assertListEqual([1, -1], list(datum.initial_state)) # initial_state = best state found in last schedule

    def test_correct_initial_state_used_reinit(self):
        sampler = ReverseAdvanceComposite(MockReverseSampler())

        h = {0: -1., 4: 2}
        J = {(0, 4): 1.5}
        anneal_schedules = [[[0, 1], [1, 0.5], [2, 0.5], [3, 1]], [[0, 1], [1, 0.5], [2, 0.5], [3, 1]]]
        initial = {0: -1, 4: -1}

        response = sampler.sample_ising(h, J, anneal_schedules=anneal_schedules,
                                        initial_state=initial, reinitialize_state=False)

        vars = response.variables

        init = [initial[v] for v in vars]  
        for datum in response.data(fields=['sample', 'initial_state'], sorted_by=None):
            self.assertListEqual(init, list(datum.initial_state))
            init = [datum.sample[v] for v in vars]  # sample should be the initial state of the next sample

    def test_combination(self):
        sampler = ReverseBatchStatesComposite(ReverseAdvanceComposite(MockReverseSampler()))

        h = {0: -1., 4: 2}
        J = {(0, 4): 1.5}
        anneal_schedules = [[[0, 1], [1, 0.5], [2, 0.5], [3, 1]], [[0, 1], [1, 0.5], [2, 0.5], [3, 1]]]
        initial_states = [{0: 1, 4: 1}, {0: -1, 4: -1}, {0: 1, 4: -1}]
        response = sampler.sample_ising(h, J, anneal_schedules=anneal_schedules, initial_states=initial_states)

        anneal_schedule_list = response.record.schedule_index
        variables = response.variables
        initial_state_list = response.record.initial_state

        for state in initial_states:
            state = [state[var] for var in variables]
            self.assertIn(state, initial_state_list)

        for state_idx in range(len(anneal_schedules)):
            self.assertIn(state_idx, anneal_schedule_list)

        self.assertGreaterEqual(len(response), 6)


class TestReverseBinary(unittest.TestCase):
    def test_sample_qubo_batch(self):
        sampler = ReverseBatchStatesComposite(MockReverseSampler())

        Q = {(0, 4): 1.5}

        response = sampler.sample_qubo(Q, initial_states=[{0: 1, 4: 1}, {0: 0, 4: 0}, {0: 1, 4: 0}])

        # nothing failed and we got at least three responses back for each initial state
        self.assertGreaterEqual(len(response), 3)

    def test_sample_qubo_advance(self):
        sampler = ReverseAdvanceComposite(MockReverseSampler())

        h = {0: -1., 4: 2}
        J = {(0, 4): 1.5}
        schedules = [[[0, 1], [1, 0.5], [2, 0.5], [3, 1]], [[0, 1], [1, 0.5], [2, 0.5], [3, 1]]]
        response = sampler.sample_ising(h, J, schedules=schedules)

        # nothing failed and we got at least two response back for each schedule point
        self.assertGreaterEqual(len(response), 2)
