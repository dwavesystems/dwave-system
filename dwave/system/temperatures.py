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

"""Effective temperature estimators

Maximum pseudo-likelihood is an efficient estimator for the temperature 
describing a classical Boltzmann distribution P(x) = exp(-H(x)/T)/Z(T) 
given samples from that distribution, where H(x) is the classical energy 
function. This estimator is implemented.
See also: https://doi.org/10.3389/fict.2016.00023 
https://www.jstor.org/stable/25464568

A temperature can also be inferred from an assumed freeze-out phenomena in 
combination with the schedule energy scale [B(s), the energy of the problem 
Hamiltonian]. This estimator is implemented. Relevant device-specific properties
are published for online solvers:
https://docs.dwavesys.com/docs/latest/doc_physical_properties.html

""" 

import warnings
import numpy as np
import dimod
from scipy import optimize

__all__ = ['effective_field', 'maximum_pseudolikelihood_temperature',
           'freezeout_effective_temperature', 'fast_effective_temperature']

def effective_field(bqm,
                    samples_like=None,
                    current_state_energy=False):
    '''Returns the effective field for all variables and all samples.
    
    The effective field with current_state_energy = False is the energy
    attributable to setting a variable to value 1, conditioned on fixed values 
    for all neighboring variables (relative to exclusion of the variable, and 
    associated energy terms, from the problem). 
    
    The effective field with current_state_energy = True is the energy gained
    by flipping the variable state against its current value (from say -1 to 1 
    in the Ising case, or 0 to 1 in the QUBO case). A positive value indicates 
    that the energy can be decreased by flipping the variable, hence the 
    variable is in a locally excited state.
    If all values are negative  (positive) within a sample, that sample is a 
    local minima (maxima).  

    In context of the Ising model formalism (any BQM can be converted to an 
    Ising model with unique values of J and h):

    For H(s) = sum_ij J_ij s_i s_j + sum_i h_i s_i

    if current_state_energy == False:
        effective_field(i,s) = sum_j (J_ij  s_j + J_ji s_i) + h_i
    else:
        effective_field(i,s) = 2 s_i [sum_j (J_ij  s_j + J_ji s_i) + h_i]
    
    Args:
        bqm (:obj:`dimod.BinaryQuadraticModel`): 
            Binary Quadratic Model Object
        samples_like (samples_like,optional):
            A collection of raw samples. `samples_like` is an extension of
            NumPy's array like structure. See examples below.
            When not provided, a single sample with all +1 assignments is
            created, ordered by bqm.variables. 
        current_state_energy (bool, optional, default=False): 
            *False*:
                Returns effective field, the energy cost (lost) by 
                inclusion of a variable in state 1.
            *True*:
                Returns energy gained in flipping the value of a variable
                against that determined by samples_like. 
    Returns:
        numpy.array: 
            An array of effective fields for each variable in each 
            sample. rows indicate independent samples, columns indicate 
            independent variables ordered in accordance with the samples_like 
            input.

    Examples:
       For a Ferromagnetic Ising chain H = - 1/2 sum_i s_i s_{i+1}
       and for a ground state sample (all +1), the energy lost when flipping
       any spin is equal to the number of couplers frustrated: -2 in the center
       of the chain, and -1 at the end. Note the value is negative (the energy 
       goes up when we flip - as expected because we are evaluating a ground 
       state).
       
       >>> import dimod
       >>> import numpy as np
       >>> from dwave.system.temperatures import effective_field
       >>> N = 5
       >>> bqm = dimod.BinaryQuadraticModel.from_ising(
                     {}, 
                     {(i,i+1) : -1 for i in range(N-1)})
       >>> var_labels = list(range(N))
       >>> samples_like = (np.ones(shape=(1,N)), var_labels)
       >>> E = effective_field(bqm,samples_like,current_state_energy=True)
       >>> print('Cost to flip spin against current assignment', E)
       
    '''
    if bqm.vartype == dimod.vartypes.Vartype.BINARY:
        #Copy and convert to spin type.
        #Not most efficient, but useful for clarity.
        bqm = bqm.change_vartype('SPIN', inplace=False)
       
    if samples_like is None:
        samples = np.ones(shape=(1,bqm.num_variables))
        labels = bqm.variables
    else:
        samples, labels = dimod.sampleset.as_samples(samples_like)
    h, (irow, icol, qdata), offset = bqm.to_numpy_vectors(
        variable_order=labels)
    # eff_field = h + J*s OR diag(Q) + (Q-diag(Q))*b
    effective_fields = np.tile(h[np.newaxis,:],(samples.shape[0],1))
    for sI in range(samples.shape[0]):
        np.add.at(effective_fields[sI,:],irow,qdata*samples[sI,icol])
        np.add.at(effective_fields[sI,:],icol,qdata*samples[sI,irow])

    if current_state_energy is True:
        #Ising: eff_field = 2*s*(h + J*s)
        effective_fields = 2*samples*effective_fields
        
    return effective_fields

def maximum_pseudolikelihood_temperature(bqm = None,
                                         sampleset = None,
                                         site_energy = None,
                                         bootstrap_size = 0,
                                         seed=None,
                                         Tguess=None):
    '''Returns a sampling-based temperature estimate.
    
    The temperature T parameterizes the Boltzmann distribution as 
    P(x) = exp(-H(x)/T)/Z(T), where P(x) is a probability over a state space, 
    H(x) is the energy function (BQM) and Z(T) is a normalization. 
    Given a sample set a temperature estimate establishes the temperature that 
    is most likely to have produced the sample set.
    An effective temperature can be derived from a sampleset by considering the 
    rate of excitations only. A maximum-pseudo-likelihood (MPL) estimator 
    considers local excitations only, which are sufficient to establish a 
    temperature efficiently (in compute time and number of samples). If the bqm
    consists of independent variable problems (no couplings between variables) 
    then the estimator is equivalent to a maximum likelihood estimator.
    
    The effective MPL temperature is defined by the solution T to 
    0 = sum_i \sum_{s in S}} nu_i(s) exp(nu_i(s)/T), 
    where nu is the energy lost in flipping spin i against its current 
    assignment (the effective field). 
    
    The problem is a convex root solving problem, and is solved with scipy 
    optimize.

    If the distribution is not Boltzmann with respect to the BQM provided, as
    may be the case for heuristic samplers (such as annealers). The temperature
    estimate can be interpretted as characterizing only a rate of local 
    excitations. In the case of sample sets obtained from D-Wave quantum 
    annealing the temperature can be identified with a physical temperature via
    a late-anneal freeze-out phenomena.
    
    Args:
        bqm (:obj:`dimod.BinaryQuadraticModel`, optional):
            Binary quadratic model describing sample distribution.
            If bqm and site_energy are both None, then by default 
            100 samples are drawn from the default DWaveSolver(), 
            with bqm defaulted as described.
        sampleset (:class:`dimod.SampleSet`, optional):
            A set of samples, assumed to be fairly sampled from
            a Boltzmann distribution characterized by bqm.
        site_energy (2d numpy array, optional):
            The effective fields associated to a sampleset and bqm. Rows
            denote spins, columns samples.
            An effective field matrix derived from the 
            bqm and sampleset if not provided.
        bootstrap_size (int, optional, default=0):
            Number of bootstrap estimators to calculate.
        seed (int, optional)
            Seeds the bootstap method (if provided) allowing reproducibility
            of the estimators given bqm and samples_like.
        Tguess (float, optional):
            Seeding the optimize process can enable faster convergence.

    Returns:
        tuple of float and numpy array:
            (T_estimate,T_bootstrap_estimates)

            *T_estimate*: a temperature estimate
            *T_bootstrap_estimates*: a numpy array of bootstrap estimators

    Examples:
       Draw samples from the default DWaveSampler() for a large spin-glass 
       problem (random couplers J, zero external field h).
       Establish a temperature estimate by maximum pseudo-likelihood. 

       Note that due to the complicated freeze-out properties of hard models,
       such as large scale spin-glasses, deviation from a classical Boltzmann 
       distribution is anticipated.
       Nevertheless, the T estimate can always be interpretted as an estimator
       of local excitations rates. For example T will be 0 if only 
       local minima are returned (even if some of the local minima are not 
       ground states).
       
       >>> import dimod
       >>> from dwave.system.temperatures import maximum_pseudolikelihood_temperature
       >>> from dwave.system.samplers import DWaveSampler
       >>> from random import random
       >>> sampler = DWaveSampler()     # doctest: +SKIP
       >>> bqm = dimod.BinaryQuadraticModel.from_ising(
                     {}, 
                     {e : 1-2*random() for e in sampler.edgelist})     # doctest: +SKIP
       >>> sampleset = sampler.sample(bqm, num_reads=100, auto_scale=False)     # doctest: +SKIP
       >>> T,T_bootstrap =  maximum_pseudolikelihood_temperature(bqm,sampleset)     # doctest: +SKIP
       >>> print('Effective temperature '
                 ,T)     # doctest: +SKIP
       
    See also:
        https://doi.org/10.3389/fict.2016.00023
        https://www.jstor.org/stable/25464568
        
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
        
        #Root finding trivial for this application, any naive root finder will
        #succeed to find the unique root:
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

def freezeout_effective_temperature(freezeout_B,temperature,units_B = 'GHz',units_T = 'mK'):
    ''' Provides an effective temperature as function of freezeout information.
    
    A D-Wave quantum annealer is assumed to implement a Hamiltonian
    H = B(s)/2 H_P - A(s)/2 H_D. H_P is the problem (classical) Hamiltonian
    and H_D is the driver Hamiltonian. B(s) is the problem energy scale and A(s)
    is the driver (transverse field) energy scale, s is the normalized anneal
    time s = t/t_a (in [0,1]).

    If a quantum annealer achieves an equilibrated distribution 
    over decohered states late in the anneal then the transverse field is 
    negligible: A(s) << B(s). 
    If in addition dynamics stop abruptly the equilibrated distribution is 
    described by a classical Boltzmann distribution for classical spin states s 
    that may be measured in accordance with a probability distribution:
    P(s) = exp(- B(s*) H_P(s) / 2 kB T)
    B(s*) is the schedule energy scale associated to the problem Hamiltonian, 
    T is the physical temperature and kB is the Boltzmann constant.
    We can define a unitless effective temperature, to complement the unitless
    Hamiltonian, definition as T_eff = 2 kB T/B(s*), returned by this function.
    P(s) = exp(-H_P(s)/T_eff)

    Single qubit freeze-out (s*) is well characterized as part of calibration 
    processes and reported alongside annealing schedules {A(s),B(s)} and 
    device temperature. This allows an effective temperature to be calculated 
    for online solvers, appropriate for some simple Hamiltonians. Values are
    typically specified in mK and GHz. More complicated systems such as chains
    may freeze-out differently (at different s, or asynchronously). Large
    problems may have slow dynamics at small s, where A(s) cannot be ignored.
    One should also be aware of known noise model properties for purposes 
    of interpretting a temperature.

    Note that for QPU solvers this temperature applies to programmed 
    Hamiltonians H_P submitted with auto_scale = False. If auto_scale=True 
    (default) an additional scaling factor is required.

    Args:
        freezeout_B (float):
            The schedule value for the problem Hamiltonian at freeze-out.

        temperature (float):
            The physical temperature of the annealer.
        
        units_B (string, optional, 'GHz'):
            Units in which the schedule is specified. Allowed values:
            'GHz' (Giga-Hertz) and 'J' (Joules).

        units_T (string, optional, 'mK'):
            Units in which the schedule is specified. Allowed values:
            'mK' and 'K'.

    
    Returns:
        float : The effective (unitless) temperature. 
    
    Examples:
       This example uses the published parameters for the Advantage_system4.1
       solver: B(s=0.612) = 3.91 GHz , T = 15.4mK.
       https://docs.dwavesys.com/docs/latest/doc_physical_properties.html 
       accessed November 22nd 2021.
       
       >>> from dwave.system.temperatures import freezeout_effective_temperature
       >>> T = freezeout_effective_temperature(freezeout_B = 3.91,
                                               temperature = 15.4)
       >>> print('Effective temperature at single qubit freeze-out is',T)
    
    '''
    
    #Convert units_B to Joules
    if units_B == 'GHz':
        h = 6.626068e-34 #J/Hz
        freezeout_B = freezeout_B *h
        freezeout_B *= 1e9
    elif units_B == 'J':
        pass
    else:
        raise ValueException("Units must be 'J' (Joules) "
                             "or 'mK' (milli-Kelvin)")
    
    if units_T == 'mK':
        temperature = temperature * 1e-3
    elif units_T == 'K':
        pass
    else:
        raise ValueException("Units must be 'K' (Kelvin) "
                             "or 'mK' (milli-Kelvin)")
    kB = 1.3806503e-23 # J/K
    
    return 2*temperature*kB/freezeout_B

def fast_effective_temperature(sampler=None, num_reads=None, seed=None, T_guess = 6, sampler_params = None):
    ''' Provides a single programming estimate to the effective temperature.
    
    A set of single qubit problems are submitted to the sampler, and excitations
    are counted - allowing an inference of the maximum-pseudolikelihood 
    estimator for temperature (for special case of single spins, equivalent to 
    maximum likelihood estimator).  

    This method is closely related to chi^2 fitting procedure for T, where
    <x_i> = tanh(h/T), as described in documentation.  
    Maximum-likelihood however places greater weight on the rare (but 
    informative) fluctuations in the strongly biased portion of the tanh 
    curve, relative to chi^2 fitting. This causes differences for non-Boltzmann
    distributions, particularly those with rare non-thermal high energy 
    excitations. When the distribution is Boltzmann, both methods yield the same
    temperature up to sampling error. Maximum likelihood estimation generalizes
    to temperature estimates over samples drawn from arbitrary Hamiltonians,
    pseudo-likelihood estimation is efficient for arbitrary Hamiltonians.
    
    The Advantage QPU has known deviations from an ideal Boltzmann sampler such,
    as flux noise, please refer to literature and documentation for more 
    details. Estimators accounting for such non-idealities may produce
    different results.

    For statistical efficiency, and in the case of QPU to avoid poor performance
    [due to noise and calibration non-idealities], it can be useful to submit 
    problems with energies comparable to 1/Temperature. The default value
    is based upon the single-qubit freeze-out hypothesis: B(s*)/kB T, for the 
    online system at temperature of 12mK, freeze-out value s*.

    We need to probe problems at an energy scale where excitations
    are common, but at which a sampler is not dominated by noise (precision).
    For Advantage_system1.1 the longitudinal field value at 
    freeze-out B(s=0.612) = 3.91GHz, and operational temperature of
    15.4mK, imply an effective temperature for single qubits of
    T_guess = B(s^*)/[2 k_B T] ~ 6 which is used by default; and is 
    sufficiently close for related online systems.

    Args:
        sampler (:class:`dimod.Sampler`, optional, default=\ :class:`~dwave.system.samplers.DWaveSampler`\ ``(client="qpu")``):
            A D-Wave sampler. 

        num_reads (int, optional):
            Number of reads to use. Default is 100 if not specified in 
            sampler_params.

        seed (int, optional):
            Seeds the problem generation process. Allowing reproducibility
            from pseudo-random samplers.

        T_guess (int, optional, default = 6.1):
            Determines the range of external fields (h_i) probed for temperature
            inference; an accurate choice raises the efficiency
            of the estimator. An inappropriate choice may lead to
            pathological behaviour. Default is based on D-Wave Advantage 
            processor temperature and energy scales, and is also suitable for 
            D-Wave 2000Q processor inference.
        
        sampler_params (dict, optional):
            Any additional non-defaulted sampler parameterization. If num_reads
            is a key, must be compatible with num_reads argument.

    Returns:
        float:
            The effective temperature describing single qubit problems in an
            external field.

    See also:
        https://doi.org/10.3389/fict.2016.00023
    
    Examples:
       Draw samples from the default DWaveSampler(), and establish the temperature
       
       >>> from dwave.system.temperatures import fast_effective_temperature
       >>> from dwave.system.samplers import DWaveSampler
       >>> sampler = DWaveSampler()
       >>> T = fast_effective_temperature(sampler)     # doctest: +SKIP
       >>> print('Effective temperature at freeze-out is',T)     # doctest: +SKIP
    '''
    
    if sampler == None:
        from dwave.system.samplers import DWaveSampler
        sampler = DWaveSampler()
    h_range = [-2/T_guess, 2/T_guess]
    if hasattr(sampler,'properties') and 'h_range' in sampler.properties:
        warn_user = False
        if h_range[0] < sampler.properties['h_range'][0]:
           h_range[0] = sampler.properties['h_range'][0]
           warn_user = True
        if h_range[1] > sampler.properties['h_range'][1]:
            h_range[1] = sampler.properties['h_range'][1]
            warn_user = True
        if warn_user:
            warnings.warn(
                'T_guess is small (relative to programmable h_range). '
                'Maximum h_range is employed, but this may be '
                'statistically inefficient.')
            
    prng = np.random.RandomState(seed)
    h_values = h_range[0] + (h_range[1]-h_range[0])*prng.rand(len(sampler.nodelist))
    bqm = dimod.BinaryQuadraticModel.from_ising({var: h_values[idx] for idx,var in enumerate(sampler.nodelist)}, {})
    if sampler_params == None:
        sampler_params = {}
    if num_reads == None:
        #Default is 100, makes efficient use of QPU access time:
        if 'num_reads' not in sampler_params:
            sampler_params['num_reads'] = 100 
    elif ('num_reads' in sampler_params
        and sampler_params['num_reads'] != num_reads):
        raise ValueError("sampler_params['num_reads'] != num_reads, "
                         "incompatible input arguments.")
    else:
        sampler_params['num_reads'] = num_reads
    sampler_params['auto_scale'] = False
    print(sampler_params)
    sampleset = sampler.sample(bqm, **sampler_params)
    T,estimators = maximum_pseudolikelihood_temperature(bqm, sampleset)
    return T
