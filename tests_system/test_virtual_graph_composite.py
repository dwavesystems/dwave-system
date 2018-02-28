import unittest

import dimod
import dwave.system as system

from tests.mock_sampler import MockSampler

####################################################################################################
# Test with the system if available
####################################################################################################

try:
    system.DWaveSampler(url='flux_bias_test', permissive_ssl=True)
    _sampler_connection = True
except Exception as e:
    # no sapi credentials are stored on the path or credentials are out of date
    _sampler_connection = False
_sampler_connection = False


@unittest.skipUnless(_sampler_connection, "No sampler to connect to")
class TestVirtualGraphWithSystem(unittest.TestCase):
    def test_smoke(self):
        child_sampler = system.DWaveSampler(url='flux_bias_test', permissive_ssl=True)

        # NB: this should be removed later
        child_sampler.solver.parameters['x_flux_bias'] = ''

        sampler = system.VirtualGraph(child_sampler, {'a': [0]})

        # the structure should be very simple
        self.assertEqual(sampler.structure, (['a'], [], {'a': set()}))

        response = sampler.sample_ising({'a': -.5}, {})