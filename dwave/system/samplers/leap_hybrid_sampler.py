# Copyright 2020 D-Wave Systems Inc.
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
A :std:doc:`dimod sampler <oceandocs:docs_dimod/reference/samplers>` for Leap's hybrid solvers.
"""
from __future__ import division
import numpy as np
from warnings import warn
from numbers import Number
from collections import abc

import dimod
from dimod.serialization.fileview import FileView

from dwave.cloud import Client

__all__ = ['LeapHybridSampler', 'LeapHybridDQMSampler']


class LeapHybridSampler(dimod.Sampler):
    """A class for using Leap's cloud-based hybrid solvers.

    Uses parameters set in a configuration file, as environment variables, or
    explicitly as input arguments for selecting and communicating with a hybrid solver.
    For more information, see
    `D-Wave Cloud Client <https://docs.ocean.dwavesys.com/en/stable/docs_cloud/sdk_index.html>`_.

    Inherits from :class:`dimod.Sampler`.

    Args:
        **config:
            Keyword arguments passed to :meth:`dwave.cloud.client.Client.from_config`.

    Examples:
        This example builds a random sparse graph and uses a hybrid solver to find a
        maximum independent set.

        >>> import dimod
        >>> import networkx as nx
        >>> import dwave_networkx as dnx
        >>> import numpy as np
        >>> from dwave.system import LeapHybridSampler
        ...
        >>> # Create a maximum-independent set problem from a random graph
        >>> problem_node_count = 300
        >>> G = nx.random_geometric_graph(problem_node_count, radius=0.0005*problem_node_count)
        >>> qubo = dnx.algorithms.independent_set.maximum_weighted_independent_set_qubo(G)
        >>> bqm = dimod.BQM.from_qubo(qubo)
        ...
        >>> # Find a good solution
        >>> sampler = LeapHybridSampler()    # doctest: +SKIP
        >>> sampleset = sampler.sample(bqm)           # doctest: +SKIP

    """

    _INTEGER_BQM_SIZE_THRESHOLD = 10000

    def __init__(self, solver=None, connection_close=True, **config):

        # we want a Hybrid solver by default, but allow override
        config.setdefault('client', 'hybrid')

        if solver is None:
            solver = {}

        if isinstance(solver, abc.Mapping):
            # TODO: instead of solver selection, try with user's default first
            if solver.setdefault('category', 'hybrid') != 'hybrid':
                raise ValueError("the only 'category' this sampler supports is 'hybrid'")
            if solver.setdefault('supported_problem_types__contains', 'bqm') != 'bqm':
                raise ValueError("the only problem type this sampler supports is 'bqm'")

            # prefer the latest version, but allow kwarg override
            solver.setdefault('order_by', '-version')

        self.client = Client.from_config(
            solver=solver, connection_close=connection_close, **config)

        self.solver = self.client.get_solver()

        # For explicitly named solvers:
        if self.properties.get('category') != 'hybrid':
            raise ValueError("selected solver is not a hybrid solver.")
        if 'bqm' not in self.solver.supported_problem_types:
            raise ValueError("selected solver does not support the 'bqm' problem type.")

    @property
    def properties(self):
        """dict: Solver properties as returned by a SAPI query.

        `Solver properties <https://docs.dwavesys.com/docs/latest/c_solver_3.html>`_
        are dependent on the selected solver and subject to change.
        """
        try:
            return self._properties
        except AttributeError:
            self._properties = properties = self.solver.properties.copy()
            return properties

    @property
    def parameters(self):
        """dict[str, list]: Solver parameters in the form of a dict, where keys are
        keyword parameters accepted by a SAPI query and values are lists of properties in
        :attr:`~dwave.system.samplers.LeapHybridSampler.properties` for each key.

        `Solver parameters <https://docs.dwavesys.com/docs/latest/c_solver_3.html>`_
        are dependent on the selected solver and subject to change.
        """
        try:
            return self._parameters
        except AttributeError:
            parameters = {param: ['parameters']
                          for param in self.properties['parameters']}
            self._parameters = parameters
            return parameters

    def sample(self, bqm, time_limit=None, **kwargs):
        """Sample from the specified binary quadratic model.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                The binary quadratic model.

            time_limit (int):
                Maximum run time, in seconds, to allow the solver to work on the problem.
                Must be at least the minimum required for the number of problem variables,
                which is calculated and set by default.
                The minimum time for a hybrid solver is specified as a piecewise-linear
                curve defined by a set of floating-point pairs, the `minimum_time_limit`
                field under
                :attr:`~dwave.system.samplers.LeapHybridSampler.properties`.
                The first element in each
                pair is the number of problem variables; the second is the minimum
                required time. The minimum time for any particular number of variables
                is a linear interpolation calculated on two pairs that represent the
                relevant range for the given number of variables.
                For example, if `LeapHybridSampler().properties["minimum_time_limit"]`
                returns `[[1, 0.1], [100, 10.0], [1000, 20.0]]`, then the minimum time
                for a 50-variable problem is 5 seconds, the linear interpolation of the
                first two pairs that represent problems with between 1 to 100 variables.

            **kwargs:
                Optional keyword arguments for the solver, specified in
                :attr:`~dwave.system.samplers.LeapHybridSampler.parameters`.

        Returns:
            :class:`dimod.SampleSet`: A `dimod` :obj:`~dimod.SampleSet` object.

        Examples:
            This example builds a random sparse graph and uses a hybrid solver to find a
            maximum independent set.

            >>> import dimod
            >>> import networkx as nx
            >>> import dwave_networkx as dnx
            >>> import numpy as np
            ...
            >>> # Create a maximum-independent set problem from a random graph
            >>> problem_node_count = 300
            >>> G = nx.random_geometric_graph(problem_node_count, radius=0.0005*problem_node_count)
            >>> qubo = dnx.algorithms.independent_set.maximum_weighted_independent_set_qubo(G)
            >>> bqm = dimod.BQM.from_qubo(qubo)
            ...
            >>> # Find a good solution
            >>> sampler = LeapHybridSampler()    # doctest: +SKIP
            >>> sampleset = sampler.sample(bqm)           # doctest: +SKIP

        """

        num_vars = bqm.num_variables
        xx, yy = zip(*self.properties["minimum_time_limit"])
        min_time_limit = np.interp([num_vars], xx, yy)[0]

        if time_limit is None:
            time_limit = min_time_limit
        if not isinstance(time_limit, Number):
            raise TypeError("time limit must be a number")
        if time_limit < min_time_limit:
            msg = ("time limit for problem size {} must be at least {}"
                   ).format(num_vars, min_time_limit)
            raise ValueError(msg)

        # for very large BQMs, it is better to send the unlabelled version,
        # to save on serializating the labels in both directions.
        # Note that different hybrid solvers accept different numbers of
        # variables and they might be lower than this threshold
        if num_vars > self._INTEGER_BQM_SIZE_THRESHOLD:
            return self._sample_large(bqm, time_limit=time_limit, **kwargs)

        return self._sample(bqm, time_limit=time_limit, **kwargs)

    def _sample(self, bqm, **kwargs):
        """Sample from the given BQM."""
        # get a FileView-compatibile BQM
        bqm = dimod.as_bqm(bqm, cls=[dimod.AdjArrayBQM,
                                     dimod.AdjMapBQM,
                                     dimod.AdjVectorBQM])

        with FileView(bqm, version=2) as fv:
            sapi_problem_id = self.solver.upload_bqm(fv).result()

        return self.solver.sample_bqm(sapi_problem_id, **kwargs).sampleset

    def _sample_large(self, bqm, **kwargs):
        """Sample from the unlabelled version of the BQM, then apply the
        labels to the returned sampleset.
        """
        # get a FileView-compatibile BQM
        # it is also important that the BQM be ordered
        bqm = dimod.as_bqm(bqm, cls=[dimod.AdjArrayBQM,
                                     dimod.AdjMapBQM,
                                     dimod.AdjVectorBQM])

        with FileView(bqm, version=2, ignore_labels=True) as fv:
            sapi_problem_id = self.solver.upload_bqm(fv).result()

        sampleset = self.solver.sample_bqm(sapi_problem_id, **kwargs).sampleset

        # relabel, as of dimod 0.9.5+ this is not blocking
        mapping = dict(enumerate(bqm.iter_variables()))
        return sampleset.relabel_variables(mapping)


LeapHybridBQMSampler = LeapHybridSampler


class LeapHybridDQMSampler:
    """A class for using Leap's cloud-based hybrid DQM solvers."""

    def __init__(self, solver=None, connection_close=True, **config):

        # we want a Hybrid solver by default, but allow override
        config.setdefault('client', 'hybrid')

        if solver is None:
            solver = {}

        if isinstance(solver, abc.Mapping):
            # TODO: instead of solver selection, try with user's default first
            if solver.setdefault('category', 'hybrid') != 'hybrid':
                raise ValueError("the only 'category' this sampler supports is 'hybrid'")
            if solver.setdefault('supported_problem_types__contains', 'dqm') != 'dqm':
                raise ValueError("the only problem type this sampler supports is 'dqm'")

            # prefer the latest version, but allow kwarg override
            solver.setdefault('order_by', '-version')

        self.client = Client.from_config(
            solver=solver, connection_close=connection_close, **config)

        self.solver = self.client.get_solver()

        # For explicitly named solvers:
        if self.properties.get('category') != 'hybrid':
            raise ValueError("selected solver is not a hybrid solver.")
        if 'dqm' not in self.solver.supported_problem_types:
            raise ValueError("selected solver does not support the 'dqm' problem type.")

        # overwrite the (static)

    @property
    def properties(self):
        """dict: Solver properties as returned by a SAPI query.

        `Solver properties <https://docs.dwavesys.com/docs/latest/c_solver_3.html>`_
        are dependent on the selected solver and subject to change.
        """
        try:
            return self._properties
        except AttributeError:
            self._properties = properties = self.solver.properties.copy()
            return properties

    @property
    def parameters(self):
        """dict[str, list]: Solver parameters in the form of a dict, where keys
        are keyword parameters accepted by a SAPI query and values are lists of
        properties in
        :attr:`~dwave.system.samplers.LeapHybridSampler.properties` for each
        key.

        `Solver parameters <https://docs.dwavesys.com/docs/latest/c_solver_3.html>`_
        are dependent on the selected solver and subject to change.
        """
        try:
            return self._parameters
        except AttributeError:
            parameters = {param: ['parameters']
                          for param in self.properties['parameters']}
            self._parameters = parameters
            return parameters

    def sample_dqm(self, dqm, time_limit=None, compressed=False, **kwargs):
        """Sample from the specified binary quadratic model.

        Args:
            bqm (:obj:`dimod.DiscreteQuadraticModel`):
                The binary quadratic model.

            time_limit (int):
                The maximum run time in seconds.

            **kwargs:
                Optional keyword arguments for the solver, specified in
                :attr:`~dwave.system.samplers.LeapHybridSampler.parameters`.

        Returns:
            :class:`dimod.SampleSet`: A sample set.

        """
        if time_limit is None:
            time_limit = self.min_time_limit(dqm)

        # we convert to a file here rather than let the cloud-client handle
        # it because we want to strip the labels and let the user handle
        # note: SpooledTemporaryFile currently returned by DQM.to_file
        # does not implement io.BaseIO interface, so we use the underlying
        # (and internal) file-like object for now
        f = dqm.to_file(compressed=compressed, ignore_labels=True)._file
        sampleset = self.solver.sample_dqm(f, time_limit=time_limit, **kwargs).sampleset
        return sampleset.relabel_variables(dict(enumerate(dqm.variables)))

    def min_time_limit(self, dqm):
        """Return the minimum `time_limit` that will be accepted for the given
        dqm.
        """
        ec = dqm.num_variable_interactions() * dqm.num_cases() / dqm.num_variables()
        limits = np.array(self.properties['minimum_time_limit'])
        t = np.interp(ec, limits[:, 0], limits[:, 1])
        return max([5, t])
