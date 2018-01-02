import dimod
import dwave_networkx as dnx


class MockSampler(dimod.TemplateSampler):
    """Create a mock sampler that can be used for tests."""
    def __init__(self, test_anti_flux_biases=None):
        dimod.TemplateSampler.__init__(self)

        # set up a sampler graph and sampler's structure
        G = dnx.chimera_graph(4, 4, 4)
        self.structure = (list(G), list(G.edges), {v: set(G[v]) for v in G})

    def sample_ising(self, h, J, num_samples=10, flux_biases=None):
        """

        """

        if flux_biases is None:
            return dimod.SimulatedAnnealingSampler().sample_ising(h, J, num_samples=num_samples)
        else:
            # we would like to replace this
            return dimod.SimulatedAnnealingSampler().sample_ising(h, J, num_samples=num_samples)
