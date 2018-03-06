import unittest
import random

from concurrent.futures import Future

import dimod
import dwave_networkx as dnx

import dwave.cloud.qpu as qpuclient

from dwave.system.samplers import DWaveSampler

try:
    # py3
    import unittest.mock as mock
except ImportError:
    # py2
    import mock


C16 = dnx.chimera_graph(16)

edges = set(tuple(edge) for edge in C16.edges)
edges.update([(v, u) for u, v in edges])  # solver has bi-directional


class MockSolver():
    nodes = set(range(2048))
    edges = edges
    properties = {'readout_thermalization_range': [0, 10000],
                  'annealing_time_range': [1, 2000],
                  'default_readout_thermalization': 0,
                  'parameters': {'num_spin_reversal_transforms': '',
                                 'programming_thermalization': '',
                                 'anneal_offsets': '',
                                 'num_reads': '',
                                 'max_answers': '',
                                 'readout_thermalization': '',
                                 'beta': "",
                                 'answer_mode': '',
                                 'auto_scale': '',
                                 'postprocess': "",
                                 'anneal_schedule': '',
                                 'chains': ""},
                  'chip_id': 'MockSolver'}

    def sample_ising(self, h, J, **kwargs):
        for key in kwargs:
            if key not in self.properties['parameters']:
                raise ValueError
        result = {'num_variables': 2048,
                  'format': 'qp',
                  'num_occurrences': [1],
                  'active_variables': list(range(2048)),
                  'solutions': [[random.choice((-1, +1)) for __ in range(2048)]],
                  'timing': {'total_real_time': 11511, 'anneal_time_per_run': 20,
                             'post_processing_overhead_time': 2042, 'qpu_sampling_time': 164,
                             'readout_time_per_run': 123,
                             'qpu_delay_time_per_sample': 21,
                             'qpu_anneal_time_per_sample': 20,
                             'total_post_processing_time': 2042,
                             'qpu_programming_time': 8740,
                             'run_time_chip': 164,
                             'qpu_access_time': 11511,
                             'qpu_readout_time_per_sample': 123},
                  'occurrences': [1]}
        result['samples'] = result['solutions']
        result['energies'] = [dimod.ising_energy(sample, h, J) for sample in result['samples']]
        future = Future()
        future.set_result(result)
        return future

    def sample_qubo(self, Q, **kwargs):
        for key in kwargs:
            if key not in self.properties['parameters']:
                raise ValueError
        result = {'num_variables': 2048,
                  'format': 'qp',
                  'num_occurrences': [1],
                  'active_variables': list(range(2048)),
                  'solutions': [[random.choice((0, 1)) for __ in range(2048)]],
                  'timing': {'total_real_time': 11511, 'anneal_time_per_run': 20,
                             'post_processing_overhead_time': 2042, 'qpu_sampling_time': 164,
                             'readout_time_per_run': 123,
                             'qpu_delay_time_per_sample': 21,
                             'qpu_anneal_time_per_sample': 20,
                             'total_post_processing_time': 2042,
                             'qpu_programming_time': 8740,
                             'run_time_chip': 164,
                             'qpu_access_time': 11511,
                             'qpu_readout_time_per_sample': 123},
                  'occurrences': [1]}
        result['samples'] = result['solutions']
        result['energies'] = [dimod.qubo_energy(sample, Q) for sample in result['samples']]
        future = Future()
        future.set_result(result)
        return future


class TestDwaveSampler(unittest.TestCase):
    @mock.patch('dwave.cloud.qpu.Client')
    def test_thingy(self, MockClient):
        instance = MockClient.return_value
        instance.get_solver.return_value = MockSolver()

        sampler = DWaveSampler()

        response = sampler.sample_ising({0: -1, 1: 1}, {})
