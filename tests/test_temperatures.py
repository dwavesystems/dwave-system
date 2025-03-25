# Copyright 2018 D-Wave Systems Inc.
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

import unittest
import numpy as np
import dimod
import warnings
from itertools import product

from dwave.system.temperatures import (
    maximum_pseudolikelihood,
    maximum_pseudolikelihood_temperature,
    effective_field,
    freezeout_effective_temperature,
    fast_effective_temperature,
    Ip_in_units_of_B,
    h_to_fluxbias,
    fluxbias_to_h,
    background_susceptibility_ising,
    background_susceptibility_bqm,
)

from dwave.system.testing import MockDWaveSampler


class TestTemperatures(unittest.TestCase):
    def test_Ip_in_units_of_B(self):
        uBs = ["J", "GHz"]
        uIps = ["A", "uA"]
        uMAFMs = ["H", "pH"]
        for uIp, uB, uMAFM in product(uIps, uBs, uMAFMs):
            Ip_in_units_of_B(units_Ip=uIp, units_B=uB, units_MAFM=uMAFM)

    def test_fluxbias_h(self):
        phi = np.random.random()
        h = fluxbias_to_h(phi)
        phi2 = h_to_fluxbias(h)
        self.assertLess(abs(phi - phi2), 1e-15)
        phi = np.random.random(10)
        h = fluxbias_to_h(phi)
        phi2 = h_to_fluxbias(h)
        self.assertTrue(np.all(np.abs(phi - phi2) < 1e-15))

    def test_effective_field(self):
        # For a simple model of independent spins H = sum_i s_i
        # The effective field is 1 (setting a spin to 1, costs 1 unit of energy,
        # relative to its exclusion)
        num_var = 3
        num_samples = 2
        var_labels = list(range(num_var))
        bqm = dimod.BinaryQuadraticModel.from_ising({var: 1 for var in var_labels}, {})
        samples_like = (np.ones(shape=(num_samples, num_var)), var_labels)
        E = effective_field(bqm, samples_like)
        self.assertTrue(np.array_equal(np.ones(shape=(num_samples, num_var)), E[0]))
        self.assertEqual(num_var, len(E[1]))
        # energy lost in flipping from sample value (1) to -1 is H(1) - H(-1) = +2.
        E = effective_field(bqm, samples_like, current_state_energy=True)
        self.assertTrue(np.array_equal(2 * np.ones(shape=(num_samples, num_var)), E[0]))

    def test_effective_field_vartype(self):
        # Check effective fields are identical whether using bqm or ising model
        num_var = 4
        var_labels = list(range(num_var))
        bqm = dimod.BinaryQuadraticModel.from_ising(
            {var: var for var in var_labels},
            {
                (var1, var2): var1 % 2 + var2 % 2 - 1
                for var1 in var_labels
                for var2 in var_labels
            },
        )
        E_ising = effective_field(bqm, current_state_energy=True)
        bqm.change_vartype("BINARY", inplace=True)
        E_bqm = effective_field(bqm, current_state_energy=True)
        self.assertTrue(bqm.vartype == dimod.BINARY)
        self.assertTrue(np.array_equal(E_ising[0], E_bqm[0]))

    def test_background_susceptibility(self):
        # A Hamiltonian with + + + and - - - as ground states.
        # Symmetry is broken
        n = 3
        dh = 1 / 4
        h = np.array([dh, -2 * dh, dh])
        J = np.array([[0, -1, 0], [-1, 0, -1], [0, -1, 0]])
        dh, dJ, k = background_susceptibility_ising(h, J)
        # Assert expected dh and dJ values.
        # ([2+3], [1+3], [1+2])

        Jd = {
            (n1, n2): J[n1, n2] for n2 in range(n) for n1 in range(n2) if J[n1, n2] != 0
        }
        hd = {n: h[n] for n in range(n)}
        bqm = dimod.BinaryQuadraticModel("SPIN").from_ising(hd, Jd)
        dh, dJ, _ = background_susceptibility_ising(hd, Jd)
        # Assert sparse and dense method match
        dbqm = dimod.BinaryQuadraticModel("SPIN").from_ising(dh, dJ)

        chi = -1 / 2**6
        bqmPdbqm = bqm + chi * dbqm
        self.assertEqual(bqmPdbqm, background_susceptibility_bqm(bqm, chi=chi))

    def test_maximum_pseudolikelihood_bqms(self):
        """Tests for parameters beyond those applicable to maximum_pseudolikelihood_temparature."""
        # h1 s1 + h2 s2 + J12 s1 s2; coefficients to be inferred:
        x = [0.5, -0.4, 0.3]
        # bqms on space {-1,1}^2, defined without coefficients
        bqms = [
            dimod.BinaryQuadraticModel("SPIN").from_ising(
                {j: (j == i) for j in range(2)}, {}
            )
            for i in range(2)
        ] + [dimod.BinaryQuadraticModel("SPIN").from_ising({}, {(0, 1): 1})]
        bqm = sum(
            [-x * bqm for bqm, x in zip(bqms, x)]
        )  # total bqm weighted by coefficients
        ss = dimod.ExactSolver().sample(bqm)
        # In practice, by sampling - for testing exact ratios
        exact_unnormalized_weights = np.exp(
            -ss.record.energy + np.min(ss.record.energy)
        )
        # A discretized version of weights, to check reproducibility at high precision
        ss.record.num_occurrences = np.ceil(1000 * exact_unnormalized_weights)
        # Correct outcome indendent of formatting / detailed root finding spec.
        beta_by_sampling = None
        x_by_sampling = None
        for roo, df, uj, sw in product(
            [True, False],
            [True, False],
            [True, False],
            [None, exact_unnormalized_weights],
        ):
            # Note that, if sample weights are None, we use num_occurrences,
            # otherwise sample weights are exactly matched to Boltzmann
            # frequencies (limit of many fair samples).
            res = maximum_pseudolikelihood(
                bqms=[bqm],  # Recover minus Inverse temperature -1
                sampleset=ss,
                sample_weights=sw,
                return_optimize_object=roo,
                degenerate_fields=df,
                use_jacobian=uj,
            )
            if roo:
                self.assertTrue(res[0].converged)
                x_ret = res[0].root
            else:
                x_ret = res[0]
            if sw is None:
                # Sampled distribution, consistent inference:
                if beta_by_sampling is None:
                    beta_by_sampling = x_ret
                    self.assertLess(abs(beta_by_sampling + 1), 1e-2)  # Close
                else:
                    self.assertAlmostEqual(x_ret, beta_by_sampling)
            else:
                # Exact distribution, exact inference
                self.assertAlmostEqual(x_ret, -1)
            if df:
                # Multidimensional histogramming not (currently) supported
                with self.assertRaises(ValueError):
                    maximum_pseudolikelihood(
                        bqms=bqms,  # Recover h1, h2, J12
                        sampleset=ss,
                        sample_weights=exact_unnormalized_weights,
                        return_optimize_object=roo,
                        degenerate_fields=df,
                        use_jacobian=uj,
                    )
                continue

            res = maximum_pseudolikelihood(
                bqms=bqms,  # Recover h1, h2, J12
                sampleset=ss,
                sample_weights=sw,
                return_optimize_object=roo,
                degenerate_fields=df,
                use_jacobian=uj,
            )
            if roo:
                self.assertTrue(res[0].success)
                x_ret = res[0].x
            else:
                x_ret = res[0]
            if sw is None:
                # Sampled distribution, consistent inference:
                if x_by_sampling is None:
                    x_by_sampling = x_ret
                    self.assertTrue(
                        all(
                            abs(a / b - 1) < 1e-2 for a, b in zip(x_by_sampling, x_ret)
                        ),
                        1e-2,
                    )  # Close
                else:
                    self.assertTrue(
                        all(abs(v1 - v2) < 1e-4 for v1, v2 in zip(x_ret, x_by_sampling))
                    )
            else:
                # Exact weights, exact recovery:
                self.assertTrue(all(abs(v1 - v2) < 1e-4 for v1, v2 in zip(x_ret, x)))

    def test_maximum_pseudolikelihood_instructive_examples(self):
        """2 degenerate ground states, symmetry is broken by background
        susceptibility. beta=1 and beta chi=0.02 are recovered given exact
        sample weights"""

        chi = -0.02
        dh = 1 / 4
        bqm = dimod.BinaryQuadraticModel("SPIN").from_ising(
            {0: dh, 1: -2 * dh, 2: dh}, {(i, i + 1): -1 for i in range(2)}
        )

        dbqm = background_susceptibility_bqm(bqm)
        bqmPdbqm = bqm + chi * dbqm
        ss = dimod.ExactSolver().sample(bqmPdbqm)
        sample_weights = np.exp(-ss.record.energy + np.min(ss.record.energy))
        xtup, _ = maximum_pseudolikelihood(
            bqms=[bqm, dbqm], sampleset=ss, sample_weights=sample_weights
        )
        self.assertAlmostEqual(xtup[0], -1)  # unperturbed Hamiltonian
        self.assertAlmostEqual(xtup[1], -chi)  # susceptibility

    def test_maximum_pseudolikelihood_temperature(self):
        # Single variable H = s_i problem with mean energy (-15 + 5)/20 = -0.5
        # 5 measured excitations out of 20.
        # This implies an effective temperature 1/atanh(0.5)
        en1 = np.array([2] * 5 + [-2] * 15)
        for optimize_method in ["bisect", None]:
            T = maximum_pseudolikelihood_temperature(
                en1=en1[:, np.newaxis], optimize_method=optimize_method
            )
            self.assertTrue(type(T) is tuple and len(T) == 2)
            self.assertLess(
                np.abs(T[0] - 1 / np.arctanh(0.5)),
                1e-8,
                f"T={1/np.arctanh(0.5)} expected, but T={T[0]}; "
                f"optimize_method={optimize_method}",
            )

        # Single variable H = s_i problem with mean energy (-5 + 5)/10 = 0
        # This implies an infinite temperature (up to numerical tolerance
        # threshold of scipy optimize.)
        en1 = np.array([1] * 5 + [-1] * 5)
        T_bracket = [0.1, 1]
        with self.assertWarns(UserWarning):
            # Returned value should match upper bracket value and
            # throw a warning.
            # Temperature is infinite (excitations and relaxations)
            # are equally likely.
            T = maximum_pseudolikelihood_temperature(
                en1=en1[:, np.newaxis], optimize_method="bisect", T_bracket=T_bracket
            )
            self.assertTrue(type(T) is tuple and len(T) == 2)
            self.assertTrue(T[0] == T_bracket[1])

        # Single variable H = s_i problem with no sample excitations
        # This implies zero temperature
        # Any bounds on T_bracket should be ignored.
        en1 = np.array([-1] * 5)
        with warnings.catch_warnings():
            # Ignore expected 'out of T_bracket bound' warning:
            warnings.simplefilter(action="ignore", category=UserWarning)
            T = maximum_pseudolikelihood_temperature(
                en1=en1[:, np.newaxis], T_bracket=T_bracket
            )
        self.assertTrue(type(T) is tuple and len(T) == 2)
        self.assertTrue(T[0] == 0)

    def test_freezeout_effective_temperature(self):
        # 24mK and 1GHz line up conveniently for T=1.00
        BsGHz = 1
        TmK = 24
        T = freezeout_effective_temperature(BsGHz, TmK)
        self.assertTrue(np.round(T * 100) == 100)

        # https://docs.dwavesys.com/docs/latest/doc_physical_properties.html
        # Accessed November 12th, 2021
        # Advantage_system4.1 (Advantage): B(s=0.612) = 3.91GHz , T = 15.4mK
        # T_eff = 0.16
        # DW_2000Q_6 (DW2000Q-LN): B(s=0.719) = 6.54GHz , T = 13.5mK
        # T_eff = 0.086
        BsGHz = 3.91
        TmK = 15.4
        T = freezeout_effective_temperature(BsGHz, TmK)
        self.assertTrue(np.round(T * 100) == 16)

        BsGHz = 6.54
        TmK = 13.5
        T = freezeout_effective_temperature(BsGHz, TmK)
        self.assertTrue(np.round(T * 1000) == 86)

    def test_fast_effective_temperature(self):
        # Initializing in a ground state, all effective
        # fields must be non-negative.
        sampler = MockDWaveSampler()
        with warnings.catch_warnings():  # Suppress MockDWaveSampler "no auto_scale" warning
            warnings.simplefilter(action="ignore", category=UserWarning)
            T, sigma = fast_effective_temperature(sampler=sampler)
            # MockDWaveSampler() returns only local minima for high precision
            # problems (ExactSolver or SteepestDescentSolver)
            self.assertEqual(T, 0)

    def test_bootstrap_errors(self):
        en1 = np.array([2] * 25 + [-2] * 75)
        num_bootstrap_samples = 100

        T, Tb = maximum_pseudolikelihood_temperature(
            en1=en1[:, np.newaxis], num_bootstrap_samples=num_bootstrap_samples
        )

        # Add test to check bootstrap estimator implementation.
        # T = 1/np.arctanh(0.5). With high probability bootstrapped values
        # are finite and will throw no warnings.
        self.assertTrue(len(Tb) == num_bootstrap_samples)

    def test_sample_weights(self):
        n = 3
        bqm = dimod.BinaryQuadraticModel("BINARY").from_qubo(
            {(i, j): np.random.normal() for i in range(n) for j in range(i, n)}
        )
        ss = dimod.ExactSolver().sample(bqm)
        for Texact in [1, 0.1 + 0.9 * np.random.random()]:
            sample_weights = np.exp(
                -1 / Texact * (ss.record.energy + np.min(ss.record.energy))
            )
            Ttup, _ = maximum_pseudolikelihood_temperature(
                bqm=bqm, sampleset=ss, sample_weights=sample_weights
            )
            # Previously 1e-8, but less accurate in some environments
            # tested by circleCI
            self.assertLess(abs(Ttup - Texact), 1e-2)

    def test_background_susceptibility_ising(self):
        #
        n = 3
        h = np.random.normal(size=n)
        J = np.random.normal(size=(n, n))
        J = J + J.transpose() - 2 * np.diag(np.diag(J))
        dh, dJ, k = background_susceptibility_ising(h, J)
        self.assertEqual(type(dh), np.ndarray)
        self.assertEqual(type(dJ), np.ndarray)
        self.assertEqual(type(k), np.float64)
        self.assertEqual(dh.shape, h.shape)
        self.assertEqual(J.shape, dJ.shape)
        h_dict = {i: h[i] for i in range(n)}
        J_dict = {(i, j): J[i, j] for i in range(n) for j in range(i + 1, n)}
        dh_dict, dJ_dict, k = background_susceptibility_ising(h_dict, J_dict)
