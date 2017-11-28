import unittest

import dwave_virtual_graph as dvg


class TestCalculator(unittest.TestCase):
    def test_get_flux_biases_output_typing(self):
        sampler = ''  # to be replaced later
        embedding = {'a': {0, 5}, 'b': {1, 4}}

        flux_biases = dvg.get_flux_biases(sampler, embedding)

        for chain in embedding.values():
            for s in chain:
                self.assertIn(s, flux_biases)
                self.assertIsInstance(flux_biases[s], float)
