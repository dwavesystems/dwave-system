import unittest

import dwave_micro_client_dimod as micro

try:
    # py3
    import unittest.mock as mock
except ImportError:
    # py2
    import mock


class TestDWaveMicroClientWithMock(unittest.TestCase):
    @mock.patch("dwave_micro_client_dimod.sampler.microclient.Connection")
    def test_instantation(self, mock_Connection):
        # set up a mock connection and a mock solver to be returned to make
        # sure the args are propgating properly

        mock_connection = mock.Mock(name='instantiated connection')
        mock_Connection.return_value = mock_connection
        solver = mock.Mock()
        solver.nodes = set([0, 1, 2, 3])
        solver.edges = set([(0, 1), (1, 0), (2, 3), (3, 2)])
        solver.properties = {}
        solver.parameters = {}
        mock_connection.get_solver.return_value = solver

        sampler = micro.DWaveSampler('solvername', 'url', 'token')

        # check that all of the args properly propogated
        mock_Connection.assert_called_with('url', 'token', None, False)

        mock_connection.get_solver.assert_called_with('solvername')

        self.assertEqual(sampler.structure[0], [0, 1, 2, 3])
        self.assertEqual(sampler.structure[1], [(0, 1), (2, 3)])
        self.assertDictEqual(sampler.structure[2], {0: {1}, 1: {0}, 2: {3}, 3: {2}})

        self.assertEqual(sampler.structure.nodelist, [0, 1, 2, 3])
        self.assertEqual(sampler.structure.edgelist, [(0, 1), (2, 3)])
        self.assertDictEqual(sampler.structure.adjacency, {0: {1}, 1: {0}, 2: {3}, 3: {2}})
