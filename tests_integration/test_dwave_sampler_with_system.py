import unittest

import numpy as np
import dimod

from dwave.system.samplers import DWaveSampler

try:
    DWaveSampler()
    _config_found = True
except ValueError:
    _config_found = False


@unittest.skipUnless(_config_found, "no configuration found to connect to a system")
class TestDWaveSamplerSystem(unittest.TestCase):
    def test_typical_small(self):
        h = [0, 0, 0, 0, 0]
        J = {(0, 4): 1}
        bqm = dimod.BinaryQuadraticModel.from_ising(h, J)

        response = DWaveSampler().sample(bqm)

        self.assertFalse(np.any(response.samples_matrix == 0))
        self.assertIs(response.vartype, dimod.SPIN)

        rows, cols = response.samples_matrix.shape

        self.assertEqual(cols, 5)
