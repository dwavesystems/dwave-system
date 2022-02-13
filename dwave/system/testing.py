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

import warnings
import collections
import dimod
import dwave.cloud.computation
import unittest.mock as mock
import dwave_networkx as dnx

from uuid import uuid4
from tabu import TabuSampler


try:
    #Simplest and efficient choice returning low energy states:
    from greedy import SteepestDescentSampler as SubstituteSampler
except ImportError:
    from dimod import SimulatedAnnealingSampler as SubstituteSampler
import concurrent.futures
import numpy as np

class MockDWaveSampler(dwave.system.samplers.DWaveSampler):
    """Mock sampler modeled after DWaveSampler that can be used for tests.

    Properties fields are populated matching a legacy device, and a 
    placeholder sampler routine based on steepest descent is instantiated.
    
    Args:
        broken_nodes (iterable of ints, optional):
            List of nodes to exclude (along with associated edges). For 
            emulation of unyielded qubits.
        
        broken_edges (iterable of (int,int) tuples, optional):
            List of edges to exclude. For emulation of unyielded edges.
        
        topology_type (string, optional, default='chimera'):
            QPU topology being emulated. Note that for Pegasus emulation the 
            fabric_only=True graph is presented. Supported options are
            'chimera', 'pegasus' or 'zephyr'
            
        topology_shape (string, optional):
            A list of three numbers [m,n,t] for Chimera, defaulted as [4,4,4]. 
            A list of one number [m] for Pegasus, defaulted as [3].
            A list of two numbers [m,t] for Zephyr, defaulted as [2,4]. 
            Defaults are chosen large enough to capture lattice-specific 
            structure, but small enough to allow rapid testing.
    
        parameter_warnings (bool, optional, default=True):
            The MockSampler is adaptive with respect to ``num_reads``,
            ``answer_mode`` and ``max_answers`` and ``label`` 
            parameters. By default ``initial_state`` can also be mocked, if
            dwave-greedy is installed. All other parameters are ignored and a 
            warning will be raised by default.

    """
    # Feature suggestion - add seed as an optional input, to allow reproducibility.
    
    nodelist = None
    edgelist = None
    properties = None
    parameters = None
    def __init__(self, broken_nodes=None, broken_edges=None,
                 topology_type='chimera',topology_shape=None,
                 parameter_warnings=True, **config):
        self.parameter_warnings = parameter_warnings
        
        shape_defaults = {'chimera' : [4,4,4],
                          'pegasus' : [3],
                          'zephyr' : [2,4]}
        if topology_type in shape_defaults:
            if topology_shape is None:
                topology_shape = shape_defaults[topology_type]
        else:
            raise ValueError("Only 'chimera', 'pegasus' and 'zephyr' "
                             "topologies are supported")
        self.properties = {
            'chip_id' : 'MockDWaveSampler',
            'topology' : {'type': topology_type, 'shape': topology_shape}
        }
        solver_graph = self.to_networkx_graph()
        
        if broken_nodes is None and broken_edges is None:
            self.nodelist = sorted(solver_graph.nodes)
            self.edgelist = sorted(tuple(sorted(edge))
                                   for edge in solver_graph.edges)
        else:
            if broken_nodes is None:
                broken_nodes = []
            self.nodelist = sorted(set(solver_graph.nodes).difference(broken_nodes))
            if broken_edges == None:
                broken_edges = []
            self.edgelist = sorted(tuple(sorted((u, v))) for
                                   u, v in solver_graph.edges
                                   if u not in broken_nodes
                                   and v not in broken_nodes
                                   and (u, v) not in broken_edges
                                   and (v, u) not in broken_edges)
        self.properties.update({
            'num_qubits' : len(solver_graph),
            'qubits' : self.nodelist.copy(),
            'couplers' : self.edgelist.copy(),
            'anneal_offset_ranges' : [[-0.5,0.5] if i in self.nodelist
                                      else [0,0] for i in range(len(self.nodelist))]})
        
        # Other properties and parameters mocked from
        # Advantage_system4.1 accessed February 10th 2022
        # 'RV7-3_P16-N1_4007890-05-C3_C5R3-device-cal-data-21-10-14-19%3a35'
        # with simplified lists for large parameters, and modified
        # topology arguments per MockSolver initialization:
        # See also:
        # https://docs.dwavesys.com/docs/latest/c_solver_parameters.html
        
        self.parameters = {
            
            'anneal_offsets': ['parameters'],
            'anneal_schedule': ['parameters'],
            'annealing_time': ['parameters'],
            'answer_mode': ['parameters'],
            'auto_scale': ['parameters'],
            'flux_biases': ['parameters'],
            'flux_drift_compensation': ['parameters'],
            'h_gain_schedule': ['parameters'],
            'initial_state': ['parameters'],
            'max_answers': ['parameters'],
            'num_reads': ['parameters'],
            'num_spin_reversal_transforms': ['parameters'],
            'programming_thermalization': ['parameters'],
            'readout_thermalization': ['parameters'],
            'reduce_intersample_correlation': ['parameters'],
            'reinitialize_state': ['parameters'],
            'warnings': [],
            'label': []}
        
        self.properties.update({
            'h_range': [-4.0, 4.0],
            'j_range': [-1.0, 1.0],
            'supported_problem_types': ['ising', 'qubo'],
            'parameters': {
                'anneal_offsets':
                'Anneal offsets for each working qubit, formatted as a list, '
                'with NaN specified for unused qubits.',
                'anneal_schedule':
                "Annealing schedule formatted as a piecewise linear list of "
                "floating-point pairs of 't' and 's'.",
                'annealing_time':
                'Quantum annealing duration, in microseconds, as a positive '
                'floating point number.',
                'answer_mode':
                "Format of returned answers, as 'histogram' or 'raw' samples.",
                'auto_scale':
                'Automatic rescaling of h and J values to their available '
                'range, as a boolean flag.',
                'flux_biases':
                'Flux biases for each working qubit, in normalized offset '
                'units, formatted as a list for all qubits.',
                'flux_drift_compensation':
                'Activation of flux drift compensation, as a boolean flag.',
                'h_gain_schedule':
                "h-gain schedule formatted as a piecewise linear list of "
                "floating-point pairs of 't' and 'g'.",
                'initial_state':
                'Initial states to use for a reverse-anneal request, as a list '
                'of qubit index and state.',
                'max_answers':
                'Maximum number of answers to return.',
                'num_reads':
                'Number of states to read (answers to return), as a positive '
                'integer.',
                'num_spin_reversal_transforms':
                'Number of spin-reversal transforms (gauge transformations) to '
                'perform.',
                'programming_thermalization':
                'Time in microseconds to wait after programming the processor '
                'in order for it to cool back to base temperature, as a '
                'positive floating point number.',
                'readout_thermalization':
                'Time in microseconds to wait after each state is read from '
                'the processor in order for it to cool back to base '
                'temperature, as a positive floating point number.',
                'reduce_intersample_correlation':
                'Addition of pauses between samples, as a boolean flag.',
                'reinitialize_state':
                'Reapplication of the initial_state for every read in reverse '
                'annealing, as a boolean flag.'},
            'vfyc': False,
            'anneal_offset_step': -0.0001500217998314891,
            'anneal_offset_step_phi0': 1.4303846404537006e-05,
            'annealing_time_range': [0.02, 83000.0],
            'default_annealing_time': 20.0,
            'default_programming_thermalization': 1000.0,
            'default_readout_thermalization': 0.0,
            'extended_j_range': [-2.0, 1.0],
            'h_gain_schedule_range': [-3.0, 3.0],
            'max_anneal_schedule_points': 50,
            'max_h_gain_schedule_points': 20,
            'num_reads_range': [1, 10000],
            'per_qubit_coupling_range': [-18.0, 15.0],
            'problem_run_duration_range': [0.0, 10000000.0],
            'programming_thermalization_range': [0.0, 10000.0],
            'readout_thermalization_range': [0.0, 10000.0],
            'tags': [],
            'category': 'qpu',
            'quota_conversion_rate': 1})
        
    @dimod.bqm_structured
    def sample(self, bqm, **kwargs):
        
        #Check kwargs compatibility with parameters and substitute sampler:
        mocked_parameters={'answer_mode',
                           'max_answers',
                           'num_reads',
                           'label'}
        if SubstituteSampler is not dimod.SimulatedAnnealingSampler:
            mocked_parameters.add('initial_state')
            # steepest greedy descent is the best substitute to
            # standardize upon. A few additional considerations as
            # dwave-greedy is modified and processor scale grows:
            # (1) For large sample sets, it would be more efficient to
            # handle ``answer_mode`` and ``max_answers`` within the
            # lower-level optimized code - just as they are handled server
            # side in QPU calls.
            # (2) We could also consider using 'large_sparse_opt' to
            # exploit fixed connectivity at large scale for current QPU
            # designs, but unlikely to be a significant inefficiency.
            
            # The fallback sampler (SA) is an inferior choice, but
            # greedy may not always be installed, unlike dimod.
        for kw in kwargs:
            if kw in self.parameters:
                if (kw not in mocked_parameters
                    and self.parameter_warnings == True):
                    
                    warnings.warn(kw + ' parameter is valid for DWaveSampler(), '
                                  'but not mocked in MockDWaveSampler().')
            else:
                raise ValueError('kwarg ' + kw + ' '
                                 'invalid for MockDWaveSampler()')

        # Timing values are for demonstration only. These could be made
        # adaptive to sampler parameters and mocked topology in principle.
        info = dict(problem_id=str(uuid4()),
                    timing={'qpu_sampling_time': 82.08,
                            'qpu_anneal_time_per_sample': 20.0,
                            'qpu_readout_time_per_sample': 41.54,
                            'qpu_access_time': 8550.28,
                            'qpu_access_overhead_time': 9340.72,
                            'qpu_programming_time': 8468.2,
                            'qpu_delay_time_per_sample': 20.54,
                            'total_post_processing_time': 1124.0,
                            'post_processing_overhead_time': 1124.0})
        label = kwargs.get('label')
        if label is not None:
            info.update(problem_label=label)

        #Special handling of flux_biases, for compatibility with virtual graphs

        flux_biases = kwargs.get('flux_biases')
        if flux_biases is not None:
            self.flux_biases_flag = True
            
            
        substitute_kwargs = {'num_reads' : kwargs.get('num_reads')}
        if substitute_kwargs['num_reads'] is None:
            substitute_kwargs['num_reads'] = 1
        
        if SubstituteSampler is not dimod.SimulatedAnnealingSampler:            
            initial_state = kwargs.get('initial_state')
            if initial_state is not None:
                # Initial state format is a list of (qubit,values)
                # value=3 denotes an unused variable (should be absent
                # from bqm). 
                # Convert to format for substitute (NB: plural key)
                substitute_kwargs['initial_states'] = (
                    np.array([pair[1] for pair in initial_state
                              if pair[1]!=3],dtype=float),
                    [pair[0] for pair in initial_state if pair[1]!=3])
                
        ss = SubstituteSampler().sample(bqm, **substitute_kwargs)
        ss.info.update(info)
        
        answer_mode = kwargs.get('answer_mode')
        if answer_mode is None or answer_mode == 'histogram':
            # Default for DWaveSampler() is 'histogram'
            ss = ss.aggregate()
        
        max_answers = kwargs.get('max_answers')
        if max_answers is not None:
            # Truncate sampleset if requested. Do not reorder (per DWaveSampler())
            ss = ss.truncate(max_answers)
        
        return ss
    
class MockLeapHybridDQMSampler:
    """Mock sampler modeled after LeapHybridDQMSampler that can be used for tests."""
    def __init__(self, **config):
        self.parameters = {'time_limit': ['parameters'],
                           'label': []}

        self.properties = {'category': 'hybrid',
                           'supported_problem_types': ['dqm'],
                           'quota_conversion_rate': 20,
                           'minimum_time_limit': [[20000, 5.0],
                                                  [100000, 6.0],
                                                  [200000, 13.0],
                                                  [500000, 34.0],
                                                  [1000000, 71.0],
                                                  [2000000, 152.0],
                                                  [5000000, 250.0],
                                                  [20000000, 400.0],
                                                  [250000000, 1200.0]],
                           'maximum_time_limit_hrs': 24.0,
                           'maximum_number_of_variables': 3000,
                           'maximum_number_of_biases': 3000000000}

    def sample_dqm(self, dqm, **kwargs):
        num_samples = 12    # min num of samples from dqm solver
        samples = np.empty((num_samples, dqm.num_variables()), dtype=int)

        for vi, v in enumerate(dqm.variables):
            samples[:, vi] = np.random.choice(dqm.num_cases(v), size=num_samples)

        return dimod.SampleSet.from_samples((samples, dqm.variables),
                                             vartype='DISCRETE',
                                             energy=dqm.energies(samples))

    def min_time_limit(self, dqm):
        # not caring about the problem, just returning the min
        return self.properties['minimum_time_limit'][0][1]

class MockLeapHybridSolver:

    properties = {'supported_problem_types': ['bqm'],
                  'minimum_time_limit': [[1, 1.0], [1024, 1.0],
                                         [4096, 10.0], [10000, 40.0]],
                  'parameters': {'time_limit': None},
                  'category': 'hybrid',
                  'quota_conversion_rate': 1}

    supported_problem_types = ['bqm']

    def upload_bqm(self, bqm, **parameters):
        bqm_adjarray = dimod.serialization.fileview.load(bqm)
        future = concurrent.futures.Future()
        future.set_result(bqm_adjarray)
        return future

    def sample_bqm(self, sapi_problem_id, time_limit):
        #Workaround until TabuSampler supports C BQMs
        bqm = dimod.BQM(sapi_problem_id.linear,
                                    sapi_problem_id.quadratic,
                                    sapi_problem_id.offset,
                                    sapi_problem_id.vartype)
        result = TabuSampler().sample(bqm, timeout=1000*int(time_limit))
        future = dwave.cloud.computation.Future('fake_solver', None)
        future._result = {'sampleset': result, 'problem_type': 'bqm'}
        return future
