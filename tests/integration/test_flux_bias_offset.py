import unittest

import minorminer

import dwave.system.flux_bias_offsets as fb

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

try:
    DWaveSampler(profile='QPU')
    _config_found = True
except ValueError:
    _config_found = False


@unittest.skipUnless(_config_found, "no configuration found to connect to a system")
class TestGetFluxBiases(unittest.TestCase):

    def step1(self, sampler, embedding):
        """get new flux bias"""
        return fb.get_flux_biases(sampler, embedding, chain_strength=1, max_age=0)

    def step2(self, sampler, embedding):
        """get flux biases from cache"""
        return fb.get_flux_biases(sampler, embedding, chain_strength=1)

    def test_simple(self):
        sampler = DWaveSampler(profile='QPU')

        embedding = minorminer.find_embedding([[0, 1], [1, 2], [0, 2]], sampler.edgelist)

        fbo1 = self.step1(sampler, embedding)
        fbo2 = self.step2(sampler, embedding)

        self.assertEqual(fbo1, fbo2)
