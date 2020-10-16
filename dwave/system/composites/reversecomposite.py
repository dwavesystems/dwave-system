# Copyright 2019 D-Wave Systems Inc.
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
#
# =============================================================================
"""
Composites that do batch operations for reverse annealing.
"""

try:
    import collections.abc as abc
except ImportError:
    import collections as abc

import dimod
import numpy as np

__all__ = 'ReverseAdvanceComposite', 'ReverseBatchStatesComposite'


class ReverseAdvanceComposite(dimod.ComposedSampler):
    """
    Composite that reverse anneals an initial sample through a sequence of anneal
    schedules.

    If you do not specify an initial sample, a random sample is used for the first
    submission. By default, each subsequent submission selects the most-found
    lowest-energy sample as its initial state. If you set reinitialize_state to False,
    which makes each submission behave like a random walk, the subsequent submission
    selects the last returned sample as its initial state.

    Args:
       sampler (:obj:`dimod.Sampler`):
            A dimod sampler.

    Examples:
       This example runs 100 reverse anneals each for three schedules on a problem
       constructed by setting random :math:`\pm 1` values on a clique (complete
       graph) of 15 nodes, minor-embedded on a D-Wave system using the
       :class:`.DWaveCliqueSampler` sampler.

       >>> import dimod
       >>> from dwave.system import DWaveCliqueSampler, ReverseAdvanceComposite
       ...
       >>> sampler = DWaveCliqueSampler()     # doctest: +SKIP
       >>> sampler_reverse = ReverseAdvanceComposite(sampler)    # doctest: +SKIP
       >>> schedule = [[[0.0, 1.0], [t, 0.5], [20, 1.0]] for t in (5, 10, 15)]
       ...
       >>> bqm = dimod.generators.ran_r(1, 15)
       >>> init_samples = {i: -1 for i in range(15)}
       >>> sampleset = sampler_reverse.sample(bqm,
       ...                                    anneal_schedules=schedule,
       ...                                    initial_state=init_samples,
       ...                                    num_reads=100,
       ...                                    reinitialize_state=True)     # doctest: +SKIP

    """

    def __init__(self, child_sampler):
        self._children = [child_sampler]

    @property
    def children(self):
        return self._children

    @property
    def parameters(self):
        param = self.child.parameters.copy()
        param['schedules'] = []
        return param

    @property
    def properties(self):
        return {'child_properties': self.child.properties.copy()}

    def sample(self, bqm, anneal_schedules=None, **parameters):
        """Sample the binary quadratic model using reverse annealing along a given set of anneal schedules.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            anneal_schedules (list of lists): Anneal schedules in order of submission. Each schedule is
                formatted as a list of [time, s] pairs

            initial_state (dict, optional): the state to reverse anneal from. If not provided, it will
                be randomly generated

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :obj:`dimod.SampleSet` that has initial_state and schedule_index fields.

        Examples:
           This example runs 100 reverse anneals each for three schedules on a problem
           constructed by setting random :math:`\pm 1` values on a clique (complete
           graph) of 15 nodes, minor-embedded on a D-Wave system using the
           :class:`.DWaveCliqueSampler` sampler.

           >>> import dimod
           >>> from dwave.system import DWaveCliqueSampler, ReverseAdvanceComposite
           ...
           >>> sampler = DWaveCliqueSampler()         # doctest: +SKIP
           >>> sampler_reverse = ReverseAdvanceComposite(sampler)    # doctest: +SKIP
           >>> schedule = [[[0.0, 1.0], [t, 0.5], [20, 1.0]] for t in (5, 10, 15)]
           ...
           >>> bqm = dimod.generators.ran_r(1, 15)
           >>> init_samples = {i: -1 for i in range(15)}
           >>> sampleset = sampler_reverse.sample(bqm,
           ...                                    anneal_schedules=schedule,
           ...                                    initial_state=init_samples,
           ...                                    num_reads=100,
           ...                                    reinitialize_state=True)  # doctest: +SKIP


        """
        child = self.child

        if anneal_schedules is None:
            return child.sample(bqm, **parameters)

        vartype_values = list(bqm.vartype.value)
        if 'initial_state' not in parameters:
            initial_state = dict(zip(list(bqm.variables), np.random.choice(vartype_values, len(bqm))))
        else:
            initial_state = parameters.pop('initial_state')

        if not isinstance(initial_state, abc.Mapping):
            raise TypeError("initial state provided must be a dict, but received {}".format(initial_state))

        if 'reinitialize_state' not in parameters:
            parameters['reinitialize_state'] = True

            if "answer_mode" in child.parameters:
                parameters['answer_mode'] = 'histogram'

        vectors = {}
        for schedule_idx, anneal_schedule in enumerate(anneal_schedules):
            sampleset = child.sample(bqm, anneal_schedule=anneal_schedule, initial_state=initial_state,
                                     **parameters)

            # update vectors
            initial_state, _ = dimod.as_samples(initial_state)
            vectors = _update_data_vector(vectors, sampleset,
                                          {'initial_state': [initial_state[0]] * len(sampleset.record.energy),
                                           'schedule_index': [schedule_idx] * len(sampleset.record.energy)})

            if schedule_idx+1 == len(anneal_schedules):
                # no need to create the next initial state - last iteration
                break

            # prepare the initial state for the next iteration
            if parameters['reinitialize_state']:
                # if reinitialize is on, choose the lowest energy, most probable state for next iteration
                ground_state_energy = sampleset.first.energy
                lowest_energy_samples = sampleset.record[sampleset.record.energy == ground_state_energy]
                lowest_energy_samples.sort(order='num_occurrences')
                initial_state = dict(zip(sampleset.variables, lowest_energy_samples[-1].sample))
            else:
                # if not reinitialized, take the last state as the next initial state
                initial_state = dict(zip(sampleset.variables, sampleset.record.sample[-1]))

        samples = vectors.pop('sample')
        return dimod.SampleSet.from_samples((samples, bqm.variables),
                                            bqm.vartype,
                                            info={'anneal_schedules': anneal_schedules},
                                            **vectors)


class ReverseBatchStatesComposite(dimod.ComposedSampler, dimod.Initialized):
    """Composite that reverse anneals from multiple initial samples. Each submission is independent
    from one another.

    Args:
       sampler (:obj:`dimod.Sampler`):
            A dimod sampler.

    Examples:
       This example runs 100 reverse anneals each from two initial states on a problem
       constructed by setting random :math:`\pm 1` values on a clique (complete
       graph) of 15 nodes, minor-embedded on a D-Wave system using the
       :class:`.DWaveCliqueSampler` sampler.

       >>> import dimod
       >>> from dwave.system import DWaveCliqueSampler, ReverseBatchStatesComposite
       ...
       >>> sampler = DWaveCliqueSampler()      # doctest: +SKIP
       >>> sampler_reverse = ReverseBatchStatesComposite(sampler)    # doctest: +SKIP
       >>> schedule = [[0.0, 1.0], [10.0, 0.5], [20, 1.0]]
       ...
       >>> bqm = dimod.generators.ran_r(1, 15)
       >>> init_samples = [{i: -1 for i in range(15)}, {i: 1 for i in range(15)}]
       >>> sampleset = sampler_reverse.sample(bqm,
       ...                                    anneal_schedule=schedule,
       ...                                    initial_states=init_samples,
       ...                                    num_reads=100,
       ...                                    reinitialize_state=True)   # doctest: +SKIP



    """

    def __init__(self, child_sampler):
        self._children = [child_sampler]

    @property
    def children(self):
        return self._children

    @property
    def parameters(self):
        param = self.child.parameters.copy()
        param['initial_states'] = []
        return param

    @property
    def properties(self):
        return {'child_properties': self.child.properties.copy()}

    def sample(self, bqm, initial_states=None, initial_states_generator='random', num_reads=None, 
               seed=None, **parameters):
        """Sample the binary quadratic model using reverse annealing from multiple initial states.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            initial_states (samples-like, optional, default=None):
                One or more samples, each defining an initial state for all the problem variables. 
                If fewer than `num_reads` initial states are defined, additional values are 
                generated as specified by `initial_states_generator`. See :func:`dimod.as_samples` 
                for a description of "samples-like".

            initial_states_generator ({'none', 'tile', 'random'}, optional, default='random'):
                Defines the expansion of `initial_states` if fewer than
                `num_reads` are specified:

                * "none":
                    If the number of initial states specified is smaller than
                    `num_reads`, raises ValueError.

                * "tile":
                    Reuses the specified initial states if fewer than `num_reads`
                    or truncates if greater.

                * "random":
                    Expands the specified initial states with randomly generated
                    states if fewer than `num_reads` or truncates if greater.

            num_reads (int, optional, default=len(initial_states) or 1):
                Equivalent to number of desired initial states. If greater than the number of 
                provided initial states, additional states will be generated. If not provided, 
                it is selected to match the length of `initial_states`. If `initial_states` 
                is not provided, `num_reads` defaults to 1.

            seed (int (32-bit unsigned integer), optional):
                Seed to use for the PRNG. Specifying a particular seed with a
                constant set of parameters produces identical results. If not
                provided, a random seed is chosen.

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :obj:`dimod.SampleSet` that has initial_state field.

        Examples:
           This example runs 100 reverse anneals each from two initial states on a problem
           constructed by setting random :math:`\pm 1` values on a clique (complete
           graph) of 15 nodes, minor-embedded on a D-Wave system using the
           :class:`.DWaveCliqueSampler` sampler.

           >>> import dimod
           >>> from dwave.system import DWaveCliqueSampler, ReverseBatchStatesComposite
           ...
           >>> sampler = DWaveCliqueSampler()       # doctest: +SKIP
           >>> sampler_reverse = ReverseBatchStatesComposite(sampler)   # doctest: +SKIP
           >>> schedule = [[0.0, 1.0], [10.0, 0.5], [20, 1.0]]
           ...
           >>> bqm = dimod.generators.ran_r(1, 15)
           >>> init_samples = [{i: -1 for i in range(15)}, {i: 1 for i in range(15)}]
           >>> sampleset = sampler_reverse.sample(bqm,
           ...                                    anneal_schedule=schedule,
           ...                                    initial_states=init_samples,
           ...                                    num_reads=100,
           ...                                    reinitialize_state=True)  # doctest: +SKIP


        """
        child = self.child
        
        parsed = self.parse_initial_states(bqm, 
                                           initial_states=initial_states,
                                           initial_states_generator=initial_states_generator,
                                           num_reads=num_reads,
                                           seed=seed)
        
        parsed_initial_states = np.ascontiguousarray(parsed.initial_states.record.sample)

        # there is gonna be way too much data generated - better to histogram them if possible
        if 'answer_mode' in child.parameters:
            parameters['answer_mode'] = 'histogram'

        # reduce number of network calls if possible by aggregating init states
        if 'num_reads' in child.parameters:
            aggreg = parsed.initial_states.aggregate()
            parsed_initial_states = np.ascontiguousarray(aggreg.record.sample)
            parameters['num_reads'] = aggreg.record.num_occurrences

        samplesets = None
        
        unneeded = {'sample', 'energy', 'num_occurrences'}

        for initial_state in parsed_initial_states:
            sampleset = child.sample(bqm, initial_state=dict(zip(bqm.variables, initial_state)), **parameters)

            vectors = {}
            for key in sampleset.record.dtype.names:
                if key not in unneeded:
                    vectors[key] = list(sampleset.record[key])

            vectors['initial_state'] = [initial_state] * len(sampleset.record.energy)

            sampleset = dimod.SampleSet.from_samples((sampleset.record.sample, bqm.variables), 
                                                     bqm.vartype, sampleset.record.energy, info={}, **vectors)
  
            if samplesets is None:
                samplesets = sampleset
            else:
                samplesets = dimod.concatenate((samplesets, sampleset))

        return samplesets

def _update_data_vector(vectors, sampleset, additional_parameters=None):
    var_names = sampleset.record.dtype.names
    for name in var_names:
        try:
            vectors[name] = vectors[name] + list(sampleset.record[name])
        except KeyError:
            vectors[name] = list(sampleset.record[name])

    for key, val in additional_parameters.items():
        if key not in var_names:
            try:
                vectors[key] = vectors[key] + list(val)
            except KeyError:
                vectors[key] = list(val)
    return vectors
