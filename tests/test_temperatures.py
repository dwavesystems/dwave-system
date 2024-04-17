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
from itertools import product

from dwave.system.temperatures import (maximum_pseudolikelihood_temperature,
                                       effective_field,
                                       freezeout_effective_temperature,
                                       fast_effective_temperature,
                                       Ip_in_units_of_B,
                                       h_to_fluxbias,
                                       fluxbias_to_h)

from dwave.system.testing import MockDWaveSampler

class TestTemperatures(unittest.TestCase):
    def test_Ip_in_units_of_B(self):
        uBs = ['J', 'GHz']
        uIps = ['A', 'uA']
        uMAFMs = ['H', 'pH']
        for uIp, uB, uMAFM in product(uIps, uBs, uMAFMs):
            _ = Ip_in_units_of_B(units_Ip=uIp,
                                 units_B=uB,
                                 units_MAFM=uMAFM)

    def test_fluxbias_h(self):
        phi = np.random.random()
        h = fluxbias_to_h(phi)
        phi2 = h_to_fluxbias(h)
        self.assertLess(abs(phi-phi2), 1e-15)
        phi = np.random.random(10)
        h = fluxbias_to_h(phi)
        phi2 = h_to_fluxbias(h)
        self.assertTrue(np.all(np.abs(phi-phi2) < 1e-15))

    def test_effective_field(self):
        # For a simple model of independent spins H = sum_i s_i
        # The effective field is 1 (setting a spin to 1, costs 1 unit of energy,
        # relative to its exclusion)
        num_var = 3
        num_samples = 2
        var_labels = list(range(num_var))
        bqm = dimod.BinaryQuadraticModel.from_ising({var: 1 for var in var_labels}, {})
        samples_like = (np.ones(shape=(num_samples,num_var)),var_labels)
        E = effective_field(bqm,
                            samples_like)
        self.assertTrue(np.array_equal(np.ones(shape=(num_samples,num_var)), E[0]))
        self.assertTrue(num_var==len(E[1]))
        # energy lost in flipping from sample value (1) to -1 is H(1) - H(-1) = +2.
        E = effective_field(bqm,
                            samples_like,
                            current_state_energy=True)
        self.assertTrue(np.array_equal(2*np.ones(shape=(num_samples,num_var)), E[0]))

    def test_effective_field_vartype(self):
        # Check effective fields are identical whether using bqm or ising model
        num_var = 4
        var_labels = list(range(num_var))
        bqm = dimod.BinaryQuadraticModel.from_ising({var: var for var in var_labels}, {(var1,var2) : var1%2 + var2%2 - 1 for var1 in var_labels for var2 in var_labels})
        E_ising = effective_field(bqm,current_state_energy=True)
        bqm.change_vartype('BINARY',inplace=True)
        E_bqm = effective_field(bqm,current_state_energy=True)
        self.assertTrue(bqm.vartype==dimod.BINARY) 
        self.assertTrue(np.array_equal(E_ising[0], E_bqm[0]))

    def test_maximum_pseudolikelihood_temperature(self):
        # Single variable H = s_i problem with mean energy (-15 + 5)/20 = -0.5
        # 5 measured excitations out of 20.
        # This implies an effective temperature 1/atanh(0.5)
        site_energy = np.array([2]*5 + [-2]*15)
        site_names = ['a']
        for optimize_method in [None,'bisect']:
            T = maximum_pseudolikelihood_temperature(
                site_energy = (site_energy[:,np.newaxis],site_names),
                optimize_method = optimize_method)
            self.assertTrue(type(T) is tuple and len(T)==2)
            self.assertTrue(np.abs(T[0]-1/np.arctanh(0.5))<1e-8)
        
        # Single variable H = s_i problem with mean energy (-5 + 5)/10 = 0
        # This implies an infinite temperature (up to numerical tolerance
        # threshold of scipy optimize.)
        site_energy = np.array([1]*5 + [-1]*5)
        T_bracket = [0.1,1]
        with self.assertWarns(UserWarning) as w:
            # Returned value should match upper bracket value and
            # throw a warning.
            # Temperature is infinite (excitations and relaxations)
            # are equally likely.
            T = maximum_pseudolikelihood_temperature(
                site_energy = (site_energy[:,np.newaxis],[1]),
                T_bracket=T_bracket)
            self.assertTrue(type(T) is tuple and len(T)==2)
            self.assertTrue(T[0]==T_bracket[1])
        
        # Single variable H = s_i problem with no sample excitations 
        # This implies zero temperature 
        # Any bounds on T_bracket should be ignored. 
        site_energy = np.array([-1]*5)
        import warnings
        with warnings.catch_warnings():
            #Ignore expected 'out of T_bracket bound' warning:
            warnings.simplefilter(action='ignore', category=UserWarning)
            T = maximum_pseudolikelihood_temperature(
                site_energy = (site_energy[:,np.newaxis],[1]),
                T_bracket=T_bracket)
        self.assertTrue(type(T) is tuple and len(T)==2)
        self.assertTrue(T[0]==0)

    def test_freezeout_effective_temperature(self):
        # 24mK and 1GHz line up conveniently for T=1.00
        BsGHz=1
        TmK=24
        T = freezeout_effective_temperature(BsGHz,TmK)
        self.assertTrue(np.round(T*100)==100)
        
        # https://docs.dwavesys.com/docs/latest/doc_physical_properties.html
        # Accessed November 12th, 2021
        # Advantage_system4.1 (Advantage): B(s=0.612) = 3.91GHz , T = 15.4mK
        # T_eff = 0.16
        # DW_2000Q_6 (DW2000Q-LN): B(s=0.719) = 6.54GHz , T = 13.5mK
        # T_eff = 0.086 
        BsGHz=3.91
        TmK=15.4
        T = freezeout_effective_temperature(BsGHz,TmK)
        self.assertTrue(np.round(T*100)==16)
        
        BsGHz=6.54
        TmK=13.5
        T = freezeout_effective_temperature(BsGHz,TmK)
        self.assertTrue(np.round(T*1000)==86)
        
        
    def test_fast_effective_temperature(self):
        # Initializing in a ground state, all effective
        # fields must be non-negative.
        sampler = MockDWaveSampler()
        T, sigma = fast_effective_temperature(sampler=sampler)
        # Simulated Annealer will tend to return only ground
        # states, hence temperature 0. But exceptions are
        # possible. Hence no assertion on value.
        

    def test_bootstrap_errors(self):
        site_energy = np.array([2]*25 +  [-2]*75)
        num_bootstrap_samples = 100
        
        T,Tb = maximum_pseudolikelihood_temperature(site_energy = (site_energy[:,np.newaxis],[1]),num_bootstrap_samples = num_bootstrap_samples)
        
        # Add test to check bootstrap estimator implementation.
        # T = 1/np.arctanh(0.5). With high probability bootstrapped values
        # are finite and will throw no warnings.
        self.assertTrue(len(Tb) == num_bootstrap_samples)
