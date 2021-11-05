# Copyright 2021 D-Wave Systems Inc.
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

"""The maximum pseudo-likelihood estimator for temperature and auxiliary 
functions.

Maximum pseudo-likelihood is an efficient estimator for the temperature 
describing a classical Boltzmann distribution P(x) = exp(-H(x)/T)/Z(T) 
given samples from that distribution, where H(x) is the energy function."""

#import os
#import json
#import networkx as nx
import numpy as np
import dimod
from scipy import optimize

__all__ = ['effective_field', 'maximum_pseudolikelihood_temperature', 'fast_effective_temperature']

def effective_field(bqm,
                    samples_like,
                    current_state_energy=False,
                    dtype=None):
    '''Returns the effective field for all variables and all samples.
    
    The effective field with current_state_energy = False is the bias 
    attributable to a single variable conditioned on fixed values for all 
    neighboring spins (the additional energy associated to inclusion of 
    variable i in state +1, relative to its exclusion). 
    
    The effective field with current_state_energy = True is the energy gained
    by flipping the variable state against its current value (from say -1 to 1 
    in the Ising case, or 1 to 0 in the QUBO case). A positive value indicates 
    that the energy can be decreased by flipping the variable, hence the 
    variable is in a locally excited state.
    If all values are negative  (positive) within a sample, that sample is a 
    local minima (maxima).  

    In context of the Ising model formalism (any BQM can be converted to an 
    Ising model with unique values of J and h):
    For H(s) = sum_ij J_ij s_i s_j + sum_i h_i s_i
    With current_state_energy == False:
        effective_field(i,s) = sum_j (J_ij  s_j + J_ji s_i) + h_i
    With current_state_energy == True:
        effective_field(i,s) = 2 s_i [sum_j (J_ij  s_j + J_ji s_i) + h_i]
    
    Args:
        bqm (:obj:`dimod.BinaryQuadraticModel`): 
            Binary Quadratic Model Object
        samples_like (samples_like,optional):
            A collection of raw samples. `samples_like` is an extension of
            NumPy's array_like_ structure. See examples below.
            When not provided, a single sample with all +1 assignments is
            created, ordered by bqm.variables.
        dtype (data-type, optional): 
            Type casting for effective field. Should match bqm interaction
            types when not defaulted. Allows calculation of fields that 
            are not double precision floats (numpy default). 
        current_state_energy (bool, optional, default=False): 
            * False * Returns effective field, the energy cost of inclusion
                of a variable in state 1.
            * True * Returns energy gained in flipping the value of a variable
                against that determined by samples_like. 
    Returns:
        numpy.array: An array of effective fields for each variable in each sample.
            rows indicate independent samples, columns indicate independent variables
            ordered in accordance with the samples_like input.
    '''
    
    if samples_like == None:
        samples = np.ones(bqm.num_variables)
        labels = bqm.variables
    else:
        samples, labels = dimod.sampleset.as_samples(samples_like)
    h, (irow, icol, qdata), offset = bqm.to_numpy_vectors(
        variable_order=labels,
        dtype=dtype)
    
    effective_field = np.tile(h[np.newaxis,:],(samples.shape[0],1))
    for sI in range(samples.shape[0]):
        np.add.at(effective_field[sI,:],irow,qdata*samples[sI,icol])
        np.add.at(effective_field[sI,:],icol,qdata*samples[sI,irow])
    if current_state_energy == True:
        if bqm.vartype == dimod.vartypes.Vartype.SPIN:
            effective_field = 2*samples*effective_field
        elif bqm.vartype == dimod.vartypes.Vartype.BINARY:
            effective_field = (2*samples-1)*effective_field
        else:
            raise ValueError('Unknown vartype for BQM')
    return effective_field

def maximum_pseudolikelihood_temperature(bqm = None,
                                         sampleset = None,
                                         site_energy = None,
                                         bootstrap_size = 0,
                                         dtype=None,
                                         seed=None,
                                         Tguess=None):
    '''Returns a sampling-based temperature estimate.
    
    The temperature T parameterizes the Boltzmann distribution as 
    P(x) = exp(-H(x)/T)/Z(T), where P(x) is a probability over a state space, 
    H(x) is the energy function (BQM) and Z(T) is a normalization. 
    Given a sample set a temperature estimate establishes the temperature that 
    is most likely to have produced the sample set.
    An effective temperature can be derived from a sampleset by considering the 
    rate of excitations only. A maximum-pseudo-likelihood estimator considers
    local excitations only, which are sufficient to establish a temperature 
    efficiently (in compute time and number of samples). If the bqm consists
    of independent variable problems (linear bias only) then the estimator
    is equivalent to a maximum likelihood estimator.
    
    The effective temperature is defined by the solution T to 
    0 = sum_i \sum_{s in S}} nu_i(s) exp(beta nu_i(s)), 
    where nu is the energy cost of flipping spin i against its current 
    assignment (the effective field).
    The problem is a convex root solving problem, and is solved with scipy 
    optimize.

    If the distribution is not Boltzmann with respect to the BQM provided, as
    may be the case for heuristic samplers (such as annealers). The temperature
    estimate can be interpretted as characterizing only a rate of local 
    excitations. In the case of sample sets obtained from D-Wave quantum 
    annealing the temperature can be identified with a physical temperature via
    a late-anneal freeze-out phenomena. Further context is provided in this 
    paper:
    
    Args:
        bqm (:obj:`dimod.BinaryQuadraticModel`, optional):
            Binary quadratic model describing sample distribution.
            If bqm and site_energy are both None, then by default 
            100 samples are drawn from the default DWaveSolver(), 
            with bqm defaulted as described.
        sampleset (:class:`dimod.SampleSet`, optional):
            A set of samples, assumed to be fairly sampled from
            a Boltzmann distribution characterized by bqm.
        site_energy (numpy array, optional):
            The effective fields associated to a sampleset and bqm.
            An effective field distribution derived from the 
            bqm and sampleset if not provided.
        dtype=None
        bootstrap_size (int, optional, default=0):
            Number of bootstrap estimators to calculate.
        seed (int, optional)
            Seeds the bootstap method (if provided) allowing reproducibility
            of the estimators given bqm and samples_like.
        Tguess (float, optional):
            Seeding the optimize process can enable faster convergence.

    Returns:
        tuple of float and numpy array (T_estimate,T_bootstrap_estimates)
            T_estimate: a temperature estimate
            T_bootstrap_estimates: a numpy array of bootstrap estimators

    See also:
        https://doi.org/10.3389/fict.2016.00023
    '''
    
    T_estimate = 0
    T_bootstrap_estimates = np.zeros(bootstrap_size)

    #Check for largest local excitation in every sample, and over all samples
    if site_energy is None:
        if bqm == None or sampleset == None:
            raise ValueError('site_energy can only be derived if both'
                             'bqm and sampleset are provided as arguments')
        site_energy = effective_field(bqm,
                                      sampleset,
                                      dtype=dtype,
                                      current_state_energy = True)
        
    max_excitation = np.max(site_energy,axis=1)
    max_excitation_all =  np.max(max_excitation)
    if max_excitation_all < 0:
        #There are no local excitations present in the sample set, therefore
        #the temperature is estimated as 0. 
        pass
    else:
        def f(x):
            #O = sum_i sum_s log P(s,i)
            #log P(s,i) = - log(1 + exp[- 2 beta nu_i(s)])
            expFactor = np.exp(site_energy*x)
            return np.sum(site_energy/(1 + expFactor))        
        def fprime(x):
            expFactor = np.exp(site_energy*x)
            return np.sum(-site_energy*site_energy/((1 + expFactor)*(1 + 1/expFactor)))
        
        #Ensures good gradient method, except pathological cases
        if Tguess == None:
            x0 = -1/max_excitation_all
        else:
            x0 = -1/Tguess
        
        root_results = optimize.root_scalar(f=f, fprime=fprime, x0 = x0)
        T_estimate = -1/(root_results.root)
        if bootstrap_size > 0:
            #By bootstrapping with respect to samples we 
            x0 = root_results.root
            prng = np.random.RandomState(seed)
            num_samples = site_energy.shape[0]
            #print(np.mean(site_energy))
            for bs in range(bootstrap_size):
                indices = np.random.choice(
                    num_samples,
                    num_samples,
                    replace=True)
                T_bootstrap_estimates[bs],_ = maximum_pseudolikelihood_temperature(
                    site_energy = site_energy[indices],
                    bootstrap_size = 0,
                    Tguess = T_estimate)
    
    return T_estimate, T_bootstrap_estimates

def fast_effective_temperature(sampler=None,num_reads=100, seed=None, T_guess = 6):
    ''' Provides a single programming estimate to the effective temperature.
    
    A set of single qubit problems are submitted to the sampler, and excitations
    are counted - allowing an inference of the maximum-pseudolikelihood estimator
    for temperature (for special case of single spins, equivalent to maximum 
    likelihood estimator).

    For statistical efficiency, and in the case of QPU to avoid poor performance
    [due to noise and calibration non-idealities], it can be useful to submit 
    problems with biases comparable to 1/Temperature. The default value
    is based upon the single-qubit freeze-out hypothesis: B(s*)/kB T, for the 
    online system at temperature of 12mK, freeze-out value s*

    This method is closely related to a <x_i> = tanh(h/T) chi^2 fitting 
    procedure, in effect the gradient is returned by this method. 
    Maximum-likelihood however places greater weight on the rare (but 
    informative) fluctuations in the strongly biased portion of the tanh() 
    curve relative to chi^2 fitting. Both methods yield the same 
    temperature up to sampling error. Maximum pseudo-likelihood generalizes
    to temperature estimates over samples drawn from arbitrary Hamiltonians.
    
    We need to probe problems at an energy scale where excitations
    are common, but at which a sampler is not dominated by noise (precision).
    For Advantage_system1.1 the longitudinal field value at 
    freeze-out B(s=0.612) = 3.91GHz, and operational temperature of
    15.4mK, imply an effective temperature for single qubits of
    T_guess = B(s^*)/[2 k_B T] ~ 6 which is used by default; and is 
    sufficiently close for other devices. 

    Args:
        sampler (:class:`dimod.Sampler`, optional, default=\ :class:`~dwave.system.samplers.DWaveSampler`\ ``(client="qpu")``):
            A D-Wave sampler. 
        num_reads (int, optional, default = 100):
            Number of reads to use.
        seed (int, optional):
            Seeds the problem generation process. Allowing reproducibility
            from pseudo-random samplers.
        T_guess (int, optional, default = 6.1):
            Determines the range of biases probed for temperature
            inference; an accurate choice raises the efficiency
            of the estimator. A very bad choice may lead to
            pathological behaviour.
            
    See also:
        https://doi.org/10.3389/fict.2016.00023
    
    '''
    
    if sampler == None:
        from dwave.system.samplers import DWaveSampler
        sampler = DWaveSampler()
    h_range = [-2/T_guess, 2/T_guess]
    prng = np.random.RandomState(seed)
    h_values = h_range[0] + (h_range[1]-h_range[0])*prng.rand(len(sampler.nodelist))
    bqm = dimod.BinaryQuadraticModel.from_ising({var: h_values[idx] for idx,var in enumerate(sampler.nodelist)}, {})
    sampler_params = {'num_reads' : num_reads, 'auto_scale' : False}
    sampleset = sampler.sample(bqm, **sampler_params)
    T,estimators = maximum_pseudolikelihood_temperature(bqm, sampleset)
    return T
    
