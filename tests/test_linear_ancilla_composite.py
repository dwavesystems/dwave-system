#  Copyright 2024 D-Wave Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import collections
import unittest

import dimod
import networkx as nx
import numpy as np

from dwave.system.testing import MockDWaveSampler
from dwave.system import LinearAncillaComposite


class TestLinearAncillaComposite(unittest.TestCase):
    def setUp(self):
        self.qpu = MockDWaveSampler(properties=dict(extended_j_range=[-2, 1]))
        self.tracked_qpu = dimod.TrackingComposite(self.qpu)

        self.sampler = LinearAncillaComposite(
            dimod.StructureComposite(
                self.tracked_qpu,
                nodelist=self.qpu.nodelist,
                edgelist=self.qpu.edgelist,
            )
        )

        self.submask = nx.subgraph(
            self.sampler.to_networkx_graph(),
            list(self.sampler.nodelist)[::2],
        )

        # this problem should run
        self.linear_problem = dimod.BinaryQuadraticModel.from_ising(
            {i: (-1) ** i for i in self.submask.nodes()},
            {},
        )

        # this problem shouldn't run
        self.linear_problem_full_graph = dimod.BinaryQuadraticModel.from_ising(
            {i: (-1) ** i for i in self.qpu.nodelist},
            {},
        )

    def test_only_quadratic(self):
        """if no linear biases, the bqm remains intact"""

        bqm = dimod.generators.ran_r(1, self.submask, seed=1)
        self.sampler.sample(bqm)
        self.assertEqual(bqm, self.tracked_qpu.input["bqm"])

    def test_h_tolerance_too_large(self):
        """if h tolerance is larger than the linear biases,
        the bqm remains intact
        """

        self.sampler.sample(self.linear_problem, h_tolerance=1.01)
        self.assertEqual(self.linear_problem, self.tracked_qpu.input["bqm"])

    def test_intermediate_h_tolerance(self):
        """check the desired h-tolerance is left in the qubit bias"""

        h_tolerance = 0.5
        self.sampler.sample(self.linear_problem, h_tolerance=h_tolerance)
        for variable, bias in self.tracked_qpu.input["bqm"].linear.items():
            if variable in self.linear_problem.variables:  # skip the ancillas
                self.assertEqual(
                    bias,
                    np.sign(self.linear_problem.get_linear(variable)) * h_tolerance,
                )

    def test_no_ancillas_available(self):
        """send a problem that uses all the qubits, not leaving any ancillas available"""

        with self.assertRaises(ValueError):
            ss = self.sampler.sample(self.linear_problem_full_graph)

    def test_ancillas_present(self):
        """check the solver used ancillas"""

        self.sampler.sample(self.linear_problem)
        self.assertGreater(
            len(self.tracked_qpu.input["bqm"].variables),
            len(self.linear_problem.variables),
        )

    def test_ancilla_cleanup(self):
        """check the problem returned has no additional variables"""

        sampleset = self.sampler.sample(self.linear_problem)
        self.assertEqual(
            len(self.linear_problem.variables),
            len(sampleset.variables),
        )

    def test_flux_biases_present(self):
        """check flux biases are applied to non-data qubits"""

        self.sampler.sample(self.linear_problem)
        flux_biases = np.array(self.tracked_qpu.input["flux_biases"])

        # flux biases are used
        self.assertGreater(sum(flux_biases != 0), 0)

        # the qubits with flux biases are not data qubits
        for qubit, flux_bias in enumerate(flux_biases):
            if flux_bias != 0:
                self.assertNotIn(qubit, self.linear_problem.variables)
