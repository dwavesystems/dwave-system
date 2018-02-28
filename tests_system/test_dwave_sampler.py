import unittest

import dimod
import dwave_micro_client as microclient

import dwave.system as system


try:
    microclient.Connection()
    _sapi_connection = True
except (OSError, IOError):
    # no sapi credentials are stored on the path
    _sapi_connection = False


@unittest.skipUnless(_sapi_connection, "no connection to sapi web services")
class TestSampler(unittest.TestCase, dimod.test.SamplerAPITest):
    """These tests require a connection to the D-Wave sapi web services."""
    def setUp(self):
        self.sampler_factory = system.DWaveSampler
        self.sampler = system.DWaveSampler()

    def test_sample_response_form(self):
        # overwrite
        pass

    def test_instantiation_structure(self):
        """check that the correct structure was assigned to the dimod sampler"""

        # these should refer to the same thing
        sampler = system.DWaveSampler()
        solver = microclient.Connection().get_solver()  # the solver the is wrapped by dimod

        nodelist, edgelist, adj = sampler.structure
        nodes = set(nodelist)
        edges = set(edgelist)

        for u, v in solver.edges:
            self.assertTrue((u, v) in edges or (v, u) in edges)
            self.assertIn(u, nodes)
            self.assertIn(v, nodes)
            self.assertIn(v, adj)
            self.assertIn(v, adj[u])
            self.assertIn(u, adj)
            self.assertIn(u, adj[v])

    def test_instantiation_keyword_arguments(self):
        conn = microclient.Connection()

        for solver_name in conn.solver_names():
            solver = microclient.Connection().get_solver(solver_name)  # the solver the is wrapped by dimod
            sampler = system.DWaveSampler(solver_name=solver_name)

            for param in solver.parameters:
                self.assertIn(param, sampler.sample_kwargs)

    def test_sample_ising(self):
        sampler = system.DWaveSampler()

        h = {0: -1., 4: 2}
        J = {(0, 4): 1.5}

        response = sampler.sample_ising(h, J)

        # nothing failed and we got at least one response back
        self.assertGreaterEqual(len(response), 1)

        for sample in response.samples():
            for v in h:
                self.assertIn(v, sample)

        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(dimod.ising_energy(sample, h, J),
                                   energy)

    def test_sample_qubo(self):
        sampler = system.DWaveSampler()

        Q = {(0, 0): .1, (0, 4): -.8, (4, 4): 1}

        response = sampler.sample_qubo(Q)

        # nothing failed and we got at least one response back
        self.assertGreaterEqual(len(response), 1)

        for sample in response.samples():
            for u, v in Q:
                self.assertIn(v, sample)
                self.assertIn(u, sample)

        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(dimod.qubo_energy(sample, Q),
                                   energy)
