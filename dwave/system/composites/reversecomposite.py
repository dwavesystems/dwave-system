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
Composite that do batch operations on initial samples provided for reverse annealing.
"""
import dimod
import numpy as np

__all__ = 'ReverseAdvanceComposite', 'BatchReverseComposite'


class ReverseAdvanceComposite(dimod.ComposedSampler):
    """ Composite that advances a sample using reverse annealing along a
        given set of anneal schedules.

    Args:
       sampler (:obj:`dimod.Sampler`):
            A dimod sampler

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

    def sample(self, bqm, schedules=None, **parameters):
        """ Composite that advances a sample using reverse annealing along a
        given set of anneal schedules. Always selects the most probable lowest energy sample out of
        the previous steps return

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            schedules (list): an ordered list of anneal schedules. The first element of the
                list should correspond to the first anneal schedule.

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :obj:`dimod.SampleSet` that has initial_state and schedule_index fields in addition to
            'energy' and 'num_occurrences'

        """
        child = self.child

        if schedules is None:
            return child.sample(bqm, **parameters)

        vartype_values = list(bqm.vartype.value)
        if 'initial_state' not in parameters:
            initial_state = dict(zip(list(bqm.variables), np.random.choice(vartype_values, len(bqm))))
        else:
            initial_state = parameters.pop('initial_state')

        if 'reinitialize_state' not in parameters:
            parameters['reinitialize_state'] = True

        # there is gonna be way too much data generated - better to histogram them
        parameters['answer_mode'] = 'histogram'

        # prepare data fields for the new sampleset object
        samples = []
        energy = []
        num_occurrences = []
        initial_states = []
        schedule_idxs = []
        sample, variables = dimod.as_samples(initial_state)
        datatypes = [('sample', sample.dtype, (len(variables),)),
                     ('energy', np.float16),
                     ('num_occurrences', np.int8),
                     ('initial_state', sample.dtype, (len(variables),)),
                     ('schedule_index', np.int8)]

        for schedule_idx, anneal_schedule in enumerate(schedules):
            sampleset = child.sample(bqm, anneal_schedule=anneal_schedule, initial_state=initial_state,
                                     **parameters)

            # collect data from sampleset object
            samples = [*samples, *sampleset.record.sample]
            energy = [*energy, *sampleset.record.energy]
            num_occurrences = [*num_occurrences, *sampleset.record.num_occurrences]

            initial_state, _ = dimod.as_samples(initial_state)
            initial_states = [*initial_states, *[initial_state] * len(sampleset.record.energy)]
            schedule_idxs = [*schedule_idxs, *[schedule_idx] * len(sampleset.record.energy)]

            # initialize the new initial_state as the lowest energy, most probable sample
            gse = sampleset.first.energy
            b = sampleset.record[sampleset.record.energy == gse]
            b.sort(order='num_occurrences')
            initial_state = dict(zip(sampleset.variables, b[-1].sample))

        record = np.rec.array(np.zeros(len(energy), dtype=datatypes))
        record['sample'] = samples
        record['energy'] = energy
        record['num_occurrences'] = num_occurrences
        record['schedule_index'] = schedule_idxs
        record['initial_state'] = initial_states

        return dimod.SampleSet(record, variables, {'schedules': schedules}, bqm.vartype)


class BatchReverseComposite(dimod.ComposedSampler):
    """ Composite that accepts multiple samples to initialize from.

    Args:
       sampler (:obj:`dimod.Sampler`):
            A dimod sampler

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

    def sample(self, bqm, **parameters):
        """ Composite that accepts multiple initial states to reverse annealing from

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :obj:`dimod.SampleSet` that has initial_state and schedule_index fields in addition to
            'energy' and 'num_occurrences'

        """
        child = self.child

        if 'initial_states' not in parameters:
            return child.sample(bqm, **parameters)

        initial_states = parameters.pop('initial_states')

        # there is gonna be way too much data generated - better to histogram them
        parameters['answer_mode'] = 'histogram'

        # prepare data fields for the new sampleset object
        samples = []
        energy = []
        num_occurrences = []
        initial_states_array = []

        datatypes = [('sample', np.int8, (len(bqm),)),
                     ('energy', np.float16),
                     ('num_occurrences', np.int8),
                     ('initial_state', np.int8, (len(bqm),))]

        for initial_state in initial_states:
            if not isinstance(initial_state, dict):
                initial_state = dict(zip(bqm.variables, initial_state))

            sampleset = child.sample(bqm, initial_state=initial_state, **parameters)

            # collect data from sampleset object
            samples = [*samples, *sampleset.record.sample]
            energy = [*energy, *sampleset.record.energy]
            num_occurrences = [*num_occurrences, *sampleset.record.num_occurrences]

            initial_state, _ = dimod.as_samples(initial_state)
            initial_states_array = [*initial_states_array, *[initial_state] * len(sampleset.record.energy)]

        record = np.rec.array(np.zeros(len(energy), dtype=datatypes))
        record['sample'] = samples
        record['energy'] = energy
        record['num_occurrences'] = num_occurrences
        record['initial_state'] = initial_states_array

        return dimod.SampleSet(record, bqm.variables, {}, bqm.vartype)
