import unittest

import dwave_micro_client_dimod as micro


####################################################################################################
# Test with the system if available
####################################################################################################

try:
    micro.DWaveSampler(url='flux_bias_test', permissive_ssl=True)
    _sampler_connection = True
except Exception as e:
    # no sapi credentials are stored on the path or credentials are out of date
    _sampler_connection = False


class TestREADME(unittest.TestCase):
    def test_readme(self):
        import dwave_micro_client_dimod as micro
        import dwave_virtual_graph as vg

        # get the D-Wave sampler (see configuration_ for setting up credentials)
        dwave_sampler = micro.DWaveSampler()

        # get the dwave_sampler's structure
        nodelist, edgelist, adj = dwave_sampler.structure

        # create and load an embedding
        embedding = {0: [8, 12], 1: [9, 13], 2: [10, 14], 3: [11, 15]}
        vg.load_embedding(nodelist, edgelist, embedding, 'K4')

        # create virtual graph
        sampler = vg.VirtualGraph(dwave_sampler, 'K4')
