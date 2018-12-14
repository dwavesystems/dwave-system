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

try:
    from neal import SimulatedAnnealingSampler
except ImportError:
    from dimod import SimulatedAnnealingSampler

C4 = dnx.chimera_graph(4, 4, 4)


class MockDWaveSampler(dimod.Sampler, dimod.Structured):
    """Mock sampler modeled after DWaveSampler that can be used for tests."""

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
        parameters['num_reads'] = ['num_reads_range']
        parameters['flux_biases'] = ['j_range']

        # add the interesting properties manually
        self.properties = properties = {}
        properties['j_range'] = [-2.0, 1.0]
        properties['h_range'] = [-2.0, 2.0]
        properties['num_reads_range'] = [1, 10000]
        properties['num_qubits'] = len(C4)

    @dimod.bqm_structured
    def sample(self, bqm, num_reads=10, flux_biases=[]):
        # we are altering the bqm
        new_bqm = bqm.copy()

        for v, fbo in enumerate(flux_biases):
            self.flux_biases_flag = True
            new_bqm.add_variable(v, 1000. * fbo)  # add the bias

        response = SimulatedAnnealingSampler().sample(new_bqm, num_reads=num_reads)

        energies = [bqm.energy(sample) for sample in response.samples(sorted_by=None)]

        return dimod.Response.from_samples(response.samples(sorted_by=None), {'energy': energies}, {}, bqm.vartype)
