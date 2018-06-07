import itertools
import unittest

import minorminer
import dimod.testing as dtest

from dwave.system.composites import VirtualGraphComposite
from dwave.system.samplers import DWaveSampler

try:
    DWaveSampler()
    _config_found = True
except ValueError:
    _config_found = False


@unittest.skipUnless(_config_found, "no configuration found to connect to a system")
class TestVirtualGraphComposite(unittest.TestCase):
    def test_construction(self):
        child_sampler = DWaveSampler()

        # get an embedding
        K10_edges = list(itertools.combinations(range(10), 2))
        embedding = minorminer.find_embedding(K10_edges, child_sampler.edgelist)

        sampler = VirtualGraphComposite(child_sampler, embedding)

        dtest.assert_sampler_api(sampler)

        h = {}
        J = {edge: -1 for edge in K10_edges}
        sampler.sample_ising(h, J)

        sampler.sample_ising(h, J, apply_flux_bias_offsets=False)
