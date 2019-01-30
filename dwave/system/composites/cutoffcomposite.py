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
# ================================================================================================
"""
A composite that removes all variables smaller than the provided cutoff

"""

from heapq import heapify, heappop

import numpy as np
from dimod import BinaryQuadraticModel, SampleSet
from dimod.core.composite import ComposedSampler
from dimod.higherorder import _relabeled_poly, poly_energies
from dimod.vartypes import SPIN

__all__ = ['CutOffComposite']

class CutOffComposite(ComposedSampler):
    """Composite to cut off variables of a problem

    Inherits from :class:`dimod.ComposedSampler`.

    Cuts off the variables that are smaller than cutoff defined.

    Args:
       sampler (:obj:`dimod.Sampler`):
            A dimod sampler

    Examples:
       This example uses :class:`.CutOffComposite` to instantiate a
       composed sampler that submits a simple Ising problem to a sampler.
       The composed sampler introduces a cutoff to linear, higher order biases.

       >>> linear = {'a': -4.0, 'b': -4.0,'c':0.01}
       >>> quadratic = {('a', 'b'): 3.2}
       >>> sampler = dwave.system.CutOffComposite(dwave.system.DwaveSampler())
       >>> response = sampler.sample_ising(linear, quadratic, cutoff=1)

    """

    def __init__(self, child_sampler):
        self._children = [child_sampler]

    @property
    def children(self):
        return self._children

    @property
    def parameters(self):
        param = self.child.parameters.copy()
        param.update({'cutoff': []})
        return param

    @property
    def properties(self):
        return {'child_properties': self.child.properties.copy()}

    def sample(self, bqm, cutoff=None, **parameters):
        """ Cutoff and sample from the provided binary quadratic model.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            cutoff (float):
                the lower bound for variable magnitudes.Variables smaller than
                cutoff are removed before sending the problem to child sampler.

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :obj:`dimod.SampleSet`

        """

        child = self.child
        if cutoff is None:
            return child.sample(bqm, **parameters)

        bqm_new, removed = cutoff_bqm(bqm, cutoff)
        response = child.sample(bqm_new, **parameters)
        return _resolve_unknown(bqm, response, removed)

    def sample_ising(self, h, J, offset=0, cutoff=None, **parameters):
        """ cutoff and sample from the problem provided by h, J, offset

        Args:
            h (dict): linear biases

            J (dict): quadratic or higher order biases

            offset (float, optional): constant energy offset

            cutoff (float):
                the lower bound for variable magnitudes.Variables smaller than
                cutoff are removed before sending the problem to child sampler.

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :obj:`dimod.SampleSet`

        """

        child = self.child

        # if quadratic, create a bqm and send to sample
        if max(map(len, J.keys())) == 2:
            bqm = BinaryQuadraticModel.from_ising(h, J, offset=offset)
            return self.sample(bqm, cutoff=cutoff, **parameters)

        if cutoff is None:
            return child.sample_ising(h, J, offset=offset, **parameters)

        h_new, j_new, removed = cutoff_ising(h, J, cutoff)
        response = child.sample_ising(h_new, j_new, offset=offset,
                                      **parameters)

        return _resolve_unknown_hubo(h, J, response, removed, offset=offset)


def cutoff_ising(h, j, cutoff):
    """Function to cutoff a problem defined by h,j.
    Can handle HUBOs.

    Args:
        h (dict): linear biases

        j (dict): quadratic or higher order biases

        cutoff (float): the lower bound for variable magnitudes

    Returns:
        h_new (dict): the cut off linear variables
        j_new (dict): the cut off quadratic variables
        list: removed variables
    """

    h_new = {}
    j_new = {}
    removed_nodes = []
    all_variables = set(h).union(*j)

    for k, v in h.items():
        if np.abs(v) < np.abs(cutoff):
            removed_nodes.append(k)
        else:
            h_new[k] = v
    accounted_for = set(h.keys())

    for k, v in j.items():
        if np.abs(v) < np.abs(cutoff):
            continue

        # if a qubit was removed in h but exists in J,
        # it cannot be set to a default alignment.
        common = set(k) & set(removed_nodes)
        for c in common:
            removed_nodes.remove(c)
        j_new[k] = v
        accounted_for = accounted_for.union(*k)

    missing = all_variables.difference(accounted_for)
    removed_nodes = set(removed_nodes).union(*missing)

    return h_new, j_new, list(removed_nodes)


def cutoff_bqm(bqm, cutoff):
    """
    function to cutoff a bqm.

    Args:
        bqm: (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to cutoff
        cutoff (float): the lower bound for variable magnitudes

    Returns:
        cut_bqm (:obj:`dimod.BinaryQuadraticModel`): the cut off bqm
        list: removed variables

    """

    h, j, removed_nodes = cutoff_ising(bqm.linear, bqm.quadratic, cutoff)
    return BinaryQuadraticModel.from_ising(h, j, bqm.offset), removed_nodes


def _resolve_unknown(bqm, response, removed):
    """Helper function to bqm resolution. Keeps track of labels"""

    samples, variables = _add_zero_removed(response, removed)
    relabeling = {v: idx for idx, v in enumerate(variables)}
    bqm.relabel_variables(relabeling, inplace=True)
    removed_nl = [relabeling[rm] for rm in removed]

    new_samples = _minimize_energy(bqm.copy(), samples, removed_nl)
    energy_vector = bqm.energies(new_samples)

    # create new sampleset
    num_samples, num_variables = np.shape(samples)
    datatypes = [('sample', np.dtype(np.int8), (num_variables,)),
                 ('energy', energy_vector.dtype)]

    record = response.record
    datatypes.extend((name, record[name].dtype, record[name].shape[1:])
                     for name in record.dtype.names if
                     name not in {'sample',
                                  'energy'})

    data = np.rec.array(np.empty(num_samples, dtype=datatypes))
    data.sample = samples
    data.energy = energy_vector
    for name in record.dtype.names:
        if name not in {'sample', 'energy'}:
            data[name] = record[name]

    response.info['cutoff_resolved'] = removed
    sampleset = SampleSet(data, variables, response.info, response.vartype)
    return sampleset


def _resolve_unknown_hubo(h, j, response, removed, offset=0):
    """Helper function to hubo resolution. Keeps track of labels"""

    # pad samples and create new variable list
    samples, variables = _add_zero_removed(response, removed)
    relabeling = {v: idx for idx, v in enumerate(variables)}
    removed_nl = [relabeling[rm] for rm in removed]
    poly = _relabeled_poly(h, j, relabeling)

    # find removed variables that minimize energy
    samples = _minimize_energy_hubo(poly, samples, removed_nl)
    energy_vector = np.add(poly_energies(samples, poly), offset)

    # create new sampleset
    num_samples, num_variables = np.shape(samples)
    datatypes = [('sample', np.dtype(np.int8), (num_variables,)),
                 ('energy', energy_vector.dtype)]

    record = response.record
    datatypes.extend((name, record[name].dtype, record[name].shape[1:])
                     for name in record.dtype.names if
                     name not in {'sample',
                                  'energy'})

    data = np.rec.array(np.empty(num_samples, dtype=datatypes))
    data.sample = samples
    data.energy = energy_vector
    for name in record.dtype.names:
        if name not in {'sample', 'energy'}:
            data[name] = record[name]

    response.info['cutoff_resolved'] = removed
    sampleset = SampleSet(data, variables, response.info, response.vartype)
    return sampleset


def _add_zero_removed(response, removed):
    """Helper function. Adds removed as zeros into samples"""
    variables = list(response.variables) + removed
    samples = np.asarray(response.record.sample)
    samples = np.concatenate((samples, np.zeros((len(samples), len(removed)),
                                                dtype=np.int8)), axis=1)

    return samples, variables


def _minimize_energy(bqm, samples, removed):
    """Resolves the unknown variables by energy minimization"""

    if bqm.vartype is SPIN:
        ZERO = -1
    else:
        ZERO = 0

    def _minenergy(arr):

        energies = []
        for cidx in removed:
            en = bqm.linear[cidx] + sum(
                arr[idx] * bqm.adj[cidx][idx] for idx in
                bqm.adj[cidx])
            energies.append([-abs(en), en, cidx])
        heapify(energies)

        while energies:
            _, e, i = heappop(energies)

            arr[i] = val = ZERO if e > 0 else 1

            for energy_triple in energies:
                k = energy_triple[2]
                if k in bqm.adj[i]:
                    energy_triple[1] += val * bqm.adj[i][k]
                energy_triple[0] = -abs(energy_triple[1])

            heapify(energies)

        return arr

    return np.apply_along_axis(_minenergy, 1, samples)


def _minimize_energy_hubo(poly, samples, removed):
    """Hubo version of minimize energy"""

    adj = {}
    for key, val in poly.items():
        if len(key) <= 1:
            continue

        for var in key:
            keytmp = list(key)
            keytmp.remove(var)
            if var in adj:
                adj[var].update({tuple(keytmp): val})
            else:
                adj[var] = {tuple(keytmp): val}

    def _minenergy(arr):

        energies = []
        for cidx in removed:
            en = poly.get((cidx,), 0) + sum(np.prod([arr[i] for i in idx]) * val
                                            for
                                            idx, val in adj[cidx].items())
            energies.append([-abs(en), en, cidx])
        heapify(energies)

        while energies:
            _, e, i = heappop(energies)

            arr[i] = val = -1 if e > 0 else 1

            for energy_triple in energies:
                k = energy_triple[2]
                tmp_sum = 0
                for jj, intval in adj[i].items():
                    if k in jj:
                        tmp_sum += intval * val * np.prod(arr[i] for i in jj
                                                          if i != k)
                energy_triple[1] += tmp_sum
                energy_triple[0] = -abs(energy_triple[1])

            heapify(energies)

        return arr

    return np.apply_along_axis(_minenergy, 1, samples)
