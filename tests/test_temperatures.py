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


class TestTemperatures(unittest.TestCase):
    
    def test_effective_field(self):
        #A sampler at 24mK, with B(s*)=1GHz at
        #freeze-out, yields an effective temperature
        #of 1.
        num_var = 10
        num_samples = 2
        var_labels = list(range(num_var))
        bqm = dimod.BinaryQuadraticModel.from_ising({var: 1 for var in var_labels}, {})
        samples_like = (np.ones(shape=(num_samples,num_var)),var_labels)
        E = effective_field(bqm,
                            samples_like)
        self.assertTrue(True)
    
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
        #24mK and 1GHz line up conveniently for T=1.000
        BsGHz=1
        TmK=24
        T = freezeout_effective_temperature(BsGHz,TmK)
        self.assertTrue(np.round(T*1000)==1000)
        
    def test_fast_effective_temperature(self):
        #Initializing in a ground state, all effective
        #fields must be non-negative.
        sampler = MockDWaveSampler()
        T = fast_effective_temperature(sampler=sampler)
        #Simulated Annealer will tend to return only ground
        #states, hence temperature 0. But exceptions are
        #possible. Hence no assertion on value
        
    def test_bqm_versus_ising(self):
        #Add test to check handling of BQM
        self.assertTrue(True)

    def test_bootstrap_errors(self):
        #Add test to check bootstrap estimator implementation
        self.assertTrue(True)
