import collections

import dimod
import dwave_networkx as dnx

C4 = dnx.chimera_graph(4, 4, 4)


class MockSampler(dimod.Sampler, dimod.Structured):
    """Create a mock sampler that can be used for tests."""

    nodelist = None
    edgelist = None
    properties = None
    parameters = None

    def __init__(self):
        self.nodelist = sorted(C4.nodes)
        self.edgelist = sorted(sorted(edge) for edge in C4.edges)

        # mark the sample kwargs
        self.parameters = parameters = {}
        parameters['num_reads'] = ['num_reads_range']
        parameters['flux_biases'] = ['j_range']

        # add the interesting properties manually
        self.properties = properties = {}
        properties['j_range'] = [-2.0, 1.0]
        properties['h_range'] = [-2.0, 2.0]
        properties['num_reads_range'] = [1, 10000]

    @dimod.bqm_structured
    def sample(self, bqm, num_reads=10, flux_biases=[]):
        # we are altering the bqm
        new_bqm = bqm.copy()

        for v, fbo in flux_biases:
            new_bqm.add_variable(v, 1000. * fbo)  # add the bias

        response = dimod.SimulatedAnnealingSampler().sample(new_bqm, num_reads=num_reads)

        energies = [bqm.energy(sample) for sample in response.samples(sorted_by=None)]

        return dimod.Response.from_dicts(response.samples(sorted_by=None), {'energy': energies})
