import collections

import dimod
import dwave_networkx as dnx

import dwave_virtual_graph as vg


class MockSampler(dimod.Sampler):
    """Create a mock sampler that can be used for tests."""
    def __init__(self, ):
        dimod.Sampler.__init__(self)

        # set up a sampler graph and sampler's structure
        G = dnx.chimera_graph(4, 4, 4)
        self.structure = (sorted(G), sorted(sorted(edge) for edge in G.edges), {v: set(G[v]) for v in G})

        self.flux_biases_flag = False

        Solver = collections.namedtuple('Solver', ['properties', 'parameters'])
        self.solver = Solver({'chip_id': 'mock_solver', 'j_range': [-2, 1]}, {})

        self.properties = self.solver.properties

    def sample_ising(self, h, J, num_reads=10, x_flux_bias=[]):
        # NB: need to change x_flux_bias later

        new_h = h.copy()
        for v, fbo in x_flux_bias:
            self.flux_biases_flag = True
            if v in new_h:
                new_h[v] += 1000 * fbo
            else:
                new_h[v] = 1000 * fbo

        response = dimod.SimulatedAnnealingSampler().sample_ising(new_h, J, num_reads=num_reads)

        new_response = dimod.Response(dimod.SPIN)
        for sample, in response.data(['sample']):
            new_response.add_sample(sample, dimod.ising_energy(sample, h, J))

        return new_response
