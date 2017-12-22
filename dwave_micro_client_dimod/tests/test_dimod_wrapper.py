import unittest

try:
    import unittest.mock as mock
except ImportError:
    import mock

import dwave_micro_client as micro

import dwave_micro_client_dimod as microdimod

try:
    microdimod.DWaveMicroClient()
    _sapi_connection = True
except OSError:
    _sapi_connection = False


class TestDWaveMicroClientWithMock(unittest.TestCase):
    @mock.patch("dwave_micro_client_dimod.dimod_wrapper.micro.Connection")
    def test_instantation(self, mock_Connection):
        # set up a mock connection and a mock solver to be returned to make
        # sure the args are propgating properly

        mock_connection = mock.Mock(name='instantiated connection')
        mock_Connection.return_value = mock_connection
        solver = mock.Mock()
        solver.nodes = set([0, 1, 2, 3])
        solver.edges = set([(0, 1), (1, 0), (2, 3), (3, 2)])
        mock_connection.get_solver.return_value = solver

        sampler = microdimod.DWaveMicroClient('solvername', 'url', 'token')

        # check that all of the args properly propogated
        mock_Connection.assert_called_with('url', 'token', None, False)

        mock_connection.get_solver.assert_called_with('solvername')

        self.assertSetEqual(sampler.structure[0], set([0, 1, 2, 3]))
        self.assertSetEqual(sampler.structure[1], set([(0, 1), (1, 0), (2, 3), (3, 2)]))
        self.assertDictEqual(sampler.structure[2], {0: {1}, 1: {0}, 2: {3}, 3: {2}})

        # @mock.patch("dwave_micro_client_dimod.dimod_wrapper.micro.Connection")
        # def test_sample_ising(self, mock_Connection):

        #     sampler = microdimod.DWaveMicroClient('solvername', 'url', 'token')

        #     # just overwrite the solver parameter
        #     sampler.solver = mock.Mock()

        #     h = {0: -1., 1: 2}
        #     J = {(0, 1): 1.5}

        #     sampler.sample_ising(h, J)

        #     sampler.sample_ising(h, J, kwrd='hello')

        #     h = {'a': -1., 1: 2}
        #     J = {('a', 1): 1.5}

        #     sampler.sample_ising(h, J)

        #     sampler.sample_ising(h, J, kwrd='hello')


@unittest.skipUnless(_sapi_connection, "no connection to sapi web services")
class TestDWaveMicroClient(unittest.TestCase):
    """Tests that require an actual connection. Basically just a sanity
    check, everything else should be handled by mock."""

    def setUp(self):
        self.sampler = microdimod.DWaveMicroClient('c4-sw_optimize')

    def test_sample_ising(self):
        sampler = self.sampler

        h = {0: -1., 4: 2}
        J = {(0, 4): 1.5}

        response = sampler.sample_ising(h, J).get_response()

        # nothing failed and we got at least one response back
        self.assertGreaterEqual(len(response), 1)

        for sample in response.samples():
            for v in h:
                self.assertIn(v, sample)

        # nothing failed and we got at least one response back
        self.assertGreaterEqual(len(response), 1)

        for sample in response.samples():
            for v in h:
                self.assertIn(v, sample)

    def test_sample_qubo(self):
        sampler = self.sampler

        Q = {(0, 0): .1, (0, 4): -.8, (4, 4): 1}

        response = sampler.sample_qubo(Q).get_qubo_response()

        # nothing failed and we got at least one response back
        self.assertGreaterEqual(len(response), 1)

        microdimod.dimod_wrapper._numpy = False
        response = sampler.sample_qubo(Q).get_qubo_response()
        microdimod.dimod_wrapper._numpy = True

        # nothing failed and we got at least one response back
        self.assertGreaterEqual(len(response), 1)
