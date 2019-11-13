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

    def sample(self, bqm, anneal_schedules=None, **parameters):
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

        if anneal_schedules is None:
            return child.sample(bqm, **parameters)

        vartype_values = list(bqm.vartype.value)
        if 'initial_state' not in parameters:
            initial_state = dict(zip(list(bqm.variables), np.random.choice(vartype_values, len(bqm))))
        else:
            initial_state = parameters.pop('initial_state')

        if not isinstance(initial_state,dict):
            raise TypeError("initial state provided must be a dict, received {}".format(initial_state))

        if 'reinitialize_state' not in parameters:
            parameters['reinitialize_state'] = True
            parameters['answer_mode'] = 'histogram'

        vectors = {}
        for schedule_idx, anneal_schedule in enumerate(anneal_schedules):
            sampleset = child.sample(bqm, anneal_schedule=anneal_schedule,initial_state = initial_state,
                                     **parameters)

            initial_state,_ = dimod.as_samples(initial_state)
            vectors = _update_data_vector(vectors,sampleset,
                                           {'initial_state':[initial_state[0]]*len(sampleset.record.energy),
                                            'schedule_index':[schedule_idx]*len(sampleset.record.energy)})

            if parameters['reinitialize_state']:
                # if reinitialize is on, choose the lowest energy, most probable state for next iteration
                gse = sampleset.first.energy
                b = sampleset.record[sampleset.record.energy == gse]
                b.sort(order='num_occurrences')
                initial_state = dict(zip(sampleset.variables, b[-1].sample))
            else:
                # if not reinitialized, take the last state as the next initial state
                initial_state = dict(zip(sampleset.variables, sampleset.record.sample[-1]))

        samples = vectors.pop('sample')
        return dimod.SampleSet.from_samples((samples, bqm.variables),
                                bqm.vartype,
                                info={'anneal_schedules': anneal_schedules},
                                **vectors)


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

        vectors = {}
        for initial_state in initial_states:

            if not isinstance(initial_state, dict):
                initial_state = dict(zip(bqm.variables, initial_state))

            sampleset = child.sample(bqm, initial_state=initial_state, **parameters)
            initial_state_, _ = dimod.as_samples(initial_state)
            vectors = _update_data_vector(vectors, sampleset,
                                          {'initial_state': [initial_state_[0]] * len(sampleset.record.energy)})

        samples = vectors.pop('sample')

        return dimod.SampleSet.from_samples((samples, bqm.variables),
                                bqm.vartype,
                                info={},
                                **vectors)


def _update_data_vector(vectors, sampleset,additional_parameters=None):

    var_names = sampleset.record.dtype.names
    for name in var_names:
        try:
            vectors[name] = [*vectors[name], *sampleset.record[name]]
        except KeyError:
            vectors[name] = sampleset.record[name]

    for key,val in additional_parameters.items():
        if key not in var_names:
            try:
                vectors[key] = [*vectors[key], *val]
            except KeyError:
                vectors[key] = val
    return vectors
