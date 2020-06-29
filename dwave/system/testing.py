# Copyright 2018 D-Wave Systems Inc.
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
import collections

import dimod
import dwave_networkx as dnx
from tabu import TabuSampler
import dwave.cloud.computation

try:
    from neal import SimulatedAnnealingSampler
except ImportError:
    from dimod import SimulatedAnnealingSampler

import concurrent.futures
import numpy as np

try:
    # py3
    import unittest.mock as mock
except ImportError:
    # py2
    import mock

C4 = dnx.chimera_graph(4, 4, 4)


class MockDWaveSampler(dimod.Sampler, dimod.Structured):
    """Mock sampler modeled after DWaveSampler that can be used for tests."""

    nodelist = None
    edgelist = None
    properties = None
    parameters = None

    def __init__(self, broken_nodes=None, **config):
        if broken_nodes is None:
            self.nodelist = sorted(C4.nodes)
            self.edgelist = sorted(tuple(sorted(edge)) for edge in C4.edges)
        else:
            self.nodelist = sorted(v for v in C4.nodes if v not in broken_nodes)
            self.edgelist = sorted(tuple(sorted((u, v))) for u, v in C4.edges
                                   if u not in broken_nodes and v not in broken_nodes)

        # mark the sample kwargs
        self.parameters = parameters = {}
        parameters['num_reads'] = ['num_reads_range']
        parameters['flux_biases'] = ['j_range']

        # add the interesting properties manually
        self.properties = properties = {}
        properties['j_range'] = [-2.0, 1.0]
        properties['h_range'] = [-2.0, 2.0]
        properties['num_reads_range'] = [1, 10000]
        properties['num_qubits'] = len(C4)
        properties['category'] = 'qpu'
        properties['quota_conversion_rate'] = 1
        properties['topology'] = {'type': 'chimera', 'shape': [4, 4, 4]}
        properties['chip_id'] = 'MockDWaveSampler'
        properties['annealing_time_range'] = [1, 2000]
        properties['num_qubits'] = len(self.nodelist)
        properties['extended_j_range'] = [-2.0, 1.0]

    @dimod.bqm_structured
    def sample(self, bqm, num_reads=10, flux_biases=[]):
        # we are altering the bqm

        if not flux_biases:
            return SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads)

        new_bqm = bqm.copy()

        for v, fbo in enumerate(flux_biases):
            self.flux_biases_flag = True
            new_bqm.add_variable(v, 1000. * fbo)  # add the bias

        response = SimulatedAnnealingSampler().sample(new_bqm, num_reads=num_reads)

        # recalculate the energies with the old bqm
        return dimod.SampleSet.from_samples_bqm([{v: sample[v] for v in bqm}
                                                 for sample in response.samples()],
                                                bqm)

class MockLeapHybridSolver:

    properties = {'supported_problem_types': ['bqm'],
                  'minimum_time_limit': [[1, 1.0], [1024, 1.0],
                                         [4096, 10.0], [10000, 40.0]],
                  'parameters': {'time_limit': None},
                  'category': 'hybrid',
                  'quota_conversion_rate': 1}

    supported_problem_types = ['bqm']

    def upload_bqm(self, bqm, **parameters):
        bqm_adjarray = dimod.serialization.fileview.load(bqm)
        future = concurrent.futures.Future()
        future.set_result(bqm_adjarray)
        return future

    def sample_bqm(self, sapi_problem_id, time_limit):
        #Workaround until TabuSampler supports C BQMs
        bqm = dimod.BQM(sapi_problem_id.linear,
                                    sapi_problem_id.quadratic,
                                    sapi_problem_id.offset,
                                    sapi_problem_id.vartype)
        result = TabuSampler().sample(bqm, timeout=1000*int(time_limit))
        future = dwave.cloud.computation.Future('fake_solver', None)
        future._result = {'sampleset': result, 'problem_type': 'bqm'}
        return future
