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

from dwave.system.temperatures import (maximum_pseudolikelihood_temperature,
                                       effective_field,
                                       freezeout_effective_temperature,
                                       fast_effective_temperature)

from dwave.system.testing import MockDWaveSampler

try:
    from neal import SimulatedAnnealingSampler
except ImportError:
    from dimod import SimulatedAnnealingSampler


class TestTemperatures(unittest.TestCase):
    
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
        self.assertTrue(np.array_equal(np.ones(shape=(num_samples,num_var)), E))
        # energy lost in flipping from sample value (1) to -1 is H(1) - H(-1) = +2.
        E = effective_field(bqm,
                            samples_like,
                            current_state_energy=True)
        self.assertTrue(np.array_equal(2*np.ones(shape=(num_samples,num_var)), E))

    def test_effective_field_vartype(self):
        # Check effective fields are identical whether using bqm or ising model
        num_var = 4
        var_labels = list(range(num_var))
        bqm = dimod.BinaryQuadraticModel.from_ising({var: var for var in var_labels}, {(var1,var2) : var1%2 + var2%2 - 1 for var1 in var_labels for var2 in var_labels})
        E_ising = effective_field(bqm,current_state_energy=True)
        bqm.change_vartype('BINARY',inplace=True)
        E_bqm = effective_field(bqm,current_state_energy=True)
        print(E_ising,E_bqm)
        self.assertTrue(np.array_equal(E_ising, E_bqm))
   
    def test_maximum_pseudolikelihood_temperature(self):
        # Single variable H = s_i problem with mean energy (-15 + 5)/20 = -0.5
        # 5 measured excitations out of 20.
        # This implies an effective temperature 1/atanh(0.5)
        site_energy = np.array([2]*5 + [-2]*15)
        T = maximum_pseudolikelihood_temperature(site_energy = site_energy[:,np.newaxis])
        self.assertTrue(type(T) is tuple and len(T)==2)
        self.assertTrue(np.abs(T[0]-1/np.arctanh(0.5))<1e-8)
        
        # Single variable H = s_i problem with mean energy (-5 + 5)/10 = 0
        # This implies an infinite temperature (up to numerical tolerance
        # threshold of scipy optimize.)
        site_energy = np.array([1]*5 + [-1]*5)
        T = maximum_pseudolikelihood_temperature(site_energy = site_energy[:,np.newaxis])
        self.assertTrue(type(T) is tuple and len(T)==2)

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
        T = fast_effective_temperature(sampler=sampler)
        # Simulated Annealer will tend to return only ground
        # states, hence temperature 0. But exceptions are
        # possible. Hence no assertion on value.
        
    def test_bqm_versus_ising(self):
        # Add test to check handling of BQM
        self.assertTrue(True)

    def test_bootstrap_errors(self):
        # Add test to check bootstrap estimator implementation
        self.assertTrue(True)
