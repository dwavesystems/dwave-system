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
A composite that removes all variables with bias smaller than the provided
cutoff

"""
import operator

import numpy as np

import dimod

__all__ = 'CutOffComposite', 'PolyCutOffComposite'


class CutOffComposite(dimod.ComposedSampler):
    """Composite to cut off variables of a problem

    Inherits from :class:`dimod.ComposedSampler`.

    Cuts off the variables that are smaller than cutoff defined.

    Args:
       sampler (:obj:`dimod.Sampler`):
            A dimod sampler

        cutoff (number, optional, default=-1):
            The lower bound for interaction bias magnitudes. Interactions
            with biases less than cutoff are removed. Isolated variables
            are also not sent to the child sampler.

        cutoff_vartype

        comparison

    Examples:
       This example uses :class:`.CutOffComposite` to instantiate a
       composed sampler that submits a simple Ising problem to a sampler.
       The composed sampler introduces a cutoff to linear, higher order biases.

       >>> linear = {'a': -4.0, 'b': -4.0,'c':0.01}
       >>> quadratic = {('a', 'b'): 3.2}
       >>> sampler = dwave.system.CutOffComposite(dwave.system.DwaveSampler())
       >>> response = sampler.sample_ising(linear, quadratic, cutoff=1)

    """

    @dimod.decorators.vartype_argument('cutoff_vartype')
    def __init__(self, child_sampler, cutoff, cutoff_vartype=dimod.SPIN,
                 comparison=operator.lt):
        self._children = [child_sampler]
        self._cutoff = cutoff
        self._cutoff_vartype = cutoff_vartype
        self._comparison = comparison

    @property
    def children(self):
        return self._children

    @property
    def parameters(self):
        return self.child.parameters.copy()

    @property
    def properties(self):
        return {'child_properties': self.child.properties.copy()}

    def sample(self, bqm, **parameters):
        """Cutoff and sample from the provided binary quadratic model.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :obj:`dimod.SampleSet`

        """
        child = self.child
        cutoff = self._cutoff
        cutoff_vartype = self._cutoff_vartype
        comp = self._comparison

        if cutoff_vartype is dimod.SPIN:
            original = bqm.spin
        else:
            original = bqm.binary

        # remove all of the interactions less than cutoff
        new = type(bqm)(original.linear,
                        ((u, v, bias)
                         for (u, v), bias in original.quadratic.items()
                         if not comp(abs(bias), cutoff)),
                        original.offset,
                        original.vartype)

        # next we check for isolated qubits and remove them, we could do this as
        # part of the construction but the assumption is there should not be
        # a large number in the 'typical' case
        isolated = [v for v in new if not new.adj[v]]
        new.remove_variables_from(isolated)

        if isolated and len(new) == 0:
            # in this case all variables are isolated, so we just put one back
            # to serve as the basis
            v = isolated.pop()
            new.linear[v] = original.linear[v]

        # get the samples from the child sampler and put them into the original vartype
        sampleset = child.sample(new, **parameters).change_vartype(bqm.vartype, inplace=True)

        # we now need to add the isolated back in, in a way that minimizes
        # the energy. There are lots of ways to do this but for now we'll just
        # do one
        if isolated:
            samples, variables = _restore_isolated(sampleset, bqm, isolated)
        else:
            samples = sampleset.record.sample
            variables = sampleset.variables

        vectors = sampleset.data_vectors
        vectors.pop('energy')  # we're going to recalculate the energy anyway

        return dimod.SampleSet.from_samples_bqm((samples, variables), bqm, **vectors)


def _restore_isolated(sampleset, bqm, isolated):
    """Return samples-like by adding isolated variables into sampleset in a
    way that minimizes the energy (relative to the other non-isolated variables).
    """

    low = -1 if bqm.vartype is dimod.SPIN else 0

    samples = sampleset.record.sample
    variables = sampleset.variables

    new_samples = np.empty((len(sampleset), len(isolated)), dtype=samples.dtype)

    # we don't let the isolated variables interact with eachother for now because
    # it will slow this down substantially
    for col, v in enumerate(isolated):
        try:
            neighbours, biases = zip(*((u, bias) for u, bias in bqm.adj[v].items()
                                       if u in variables))  # ignore other isolates
        except ValueError:
            # happens when only neightbours are other isolated variables
            new_samples[:, col] = bqm.linear[v] <= 0
            continue

        idxs = [variables.index[u] for u in neighbours]

        # figure out which value for v would minimize the energy
        # v(h_v + \sum_u J_uv * u)
        new_samples[:, col] = samples[:, idxs].dot(biases) < -bqm.linear[v]

    if bqm.vartype is dimod.SPIN:
        new_samples = 2*new_samples - 1

    return np.concatenate((samples, new_samples), axis=1), list(variables) + isolated


class PolyCutOffComposite(dimod.ComposedPolySampler):
    @dimod.decorators.vartype_argument('cutoff_vartype')
    def __init__(self, child_sampler, cutoff, cutoff_vartype=dimod.SPIN,
                 comparison=operator.lt):
        if not isinstance(child_sampler, dimod.PolySampler):
            raise TypeError("Child sampler must be a PolySampler")
        self._children = [child_sampler]
        self._cutoff = cutoff
        self._cutoff_vartype = cutoff_vartype
        self._comparison = comparison

    @property
    def children(self):
        return self._children

    @property
    def parameters(self):
        return self.child.parameters.copy()

    @property
    def properties(self):
        return {'child_properties': self.child.properties.copy()}

    def sample_poly(self, poly, **kwargs):
        child = self.child
        cutoff = self._cutoff
        cutoff_vartype = self._cutoff_vartype
        comp = self._comparison

        if cutoff_vartype is dimod.SPIN:
            original = poly.to_spin(copy=False)
        else:
            original = poly.to_binary(copy=False)

        # remove all of the terms of order >= 2 that have a bias less than cutoff
        new = type(poly)(((term, bias) for term, bias in original.items()
                          if len(term) > 1 and not comp(abs(bias), cutoff)),
                         cutoff_vartype)

        # next we check for isolated qubits and remove them, we could do this as
        # part of the construction but the assumption is there should not be
        # a large number in the 'typical' case. We want isolated to be a set
        # because we're going to be doing __contains__
        isolated = list(original.variables.difference(new.variables))

        if isolated and len(new) == 0:
            # in this case all variables are isolated, so we just put one back
            # to serve as the basis
            term = isolated.pop(),
            new[term] = original[term]

        # get the samples from the child sampler and put them into the original vartype
        sampleset = child.sample_poly(new, **kwargs).change_vartype(poly.vartype, inplace=True)

        # we now need to add the isolated back in, in a way that minimizes
        # the energy. There are lots of ways to do this but for now we'll just
        # do one
        if isolated:
            samples, variables = _restore_isolated_higherorder(sampleset, poly, isolated)
        else:
            samples = sampleset.record.sample
            variables = sampleset.variables

        vectors = sampleset.data_vectors
        vectors.pop('energy')  # we're going to recalculate the energy anyway

        return dimod.SampleSet.from_samples_bqm((samples, variables), poly, **vectors)


def _restore_isolated_higherorder(sampleset, poly, isolated):
    """Return samples-like by adding isolated variables into sampleset in a
    way that minimizes the energy (relative to the other non-isolated variables).

    Isolated should be ordered.
    """

    low = -1 if poly.vartype is dimod.SPIN else 0

    samples = sampleset.record.sample
    variables = sampleset.variables

    new_samples = np.empty((len(sampleset), len(isolated)), dtype=samples.dtype)

    # we don't let the isolated variables interact with eachother for now because
    # it will slow this down substantially
    energies = {v: 0 for v in isolated}
    for term, bias in poly.items():

        isolated_components = term.intersection(isolated)

        if not isolated_components:
            continue

        en = bias  # energy contribution of the term
        for v in term:
            if v in energies:
                continue
            en *= samples[:, sampleset.variables.index(v)]

        for v in isolated_components:
            energies[v] += en

    # now put those energies into new_samples
    for col, v in enumerate(isolated):
        new_samples[:, col] = energies[v] < 0

    if poly.vartype is dimod.SPIN:
        new_samples = 2*new_samples - 1

    return np.concatenate((samples, new_samples), axis=1), list(variables) + isolated
