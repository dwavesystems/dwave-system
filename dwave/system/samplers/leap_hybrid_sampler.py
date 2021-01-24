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

"""
A :std:doc:`dimod sampler <oceandocs:docs_dimod/reference/samplers>` for Leap's hybrid solvers.
"""

import numpy as np
from warnings import warn
from numbers import Number
from collections import abc

import dimod
from dimod.serialization.fileview import FileView

from dwave.cloud import Client

__all__ = ['LeapHybridSampler', 'LeapHybridDQMSampler']


class LeapHybridSampler(dimod.Sampler):
    """A class for using Leap's cloud-based hybrid BQM solvers.

    Leap’s quantum-classical hybrid BQM solvers are intended to solve arbitrary
    application problems formulated as binary quadratic models (BQM).

    You can configure your :term:`solver` selection and usage by setting parameters,
    hierarchically, in a configuration file, as environment variables, or
    explicitly as input arguments, as described in
    `D-Wave Cloud Client <https://docs.ocean.dwavesys.com/en/stable/docs_cloud/sdk_index.html>`_.

    :ref:`dwave-cloud-client <sdk_index_cloud>`'s
    :meth:`~dwave.cloud.client.Client.get_solvers` method filters solvers you have
    access to by `solver properties <https://docs.dwavesys.com/docs/latest/c_solver_3.html>`_
    ``category=hybrid`` and ``supported_problem_type=bqm``. By default, online
    hybrid BQM solvers are returned ordered by latest ``version``.

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
            solver.setdefault('order_by', '-properties.version')

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
            parameters.update(label=[])
            self._parameters = parameters
            return parameters

    def sample(self, bqm, time_limit=None, **kwargs):
        """Sample from the specified binary quadratic model.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model.

            time_limit (int):
                Maximum run time, in seconds, to allow the solver to work on the
                problem. Must be at least the minimum required for the number of
                problem variables, which is calculated and set by default.

                :meth:`~dwave.system.samplers.LeapHybridSampler.min_time_limit`
                calculates (and describes) the minimum time for your problem.

            **kwargs:
                Optional keyword arguments for the solver, specified in
                :attr:`~dwave.system.samplers.LeapHybridSampler.parameters`.

        Returns:
            :class:`dimod.SampleSet`: A `dimod` :obj:`~dimod.SampleSet` object.

        Examples:
            This example builds a random sparse graph and uses a hybrid solver to
            find a maximum independent set.

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

        if time_limit is None:
            time_limit = self.min_time_limit(bqm)
        if not isinstance(time_limit, Number):
            raise TypeError("time limit must be a number")
        if time_limit < self.min_time_limit(bqm):
            msg = ("time limit for problem size {} must be at least {}"
                   ).format(num_vars, self.min_time_limit(bqm))
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

    def min_time_limit(self, bqm):
        """Return the minimum `time_limit` accepted for the given problem.

        The minimum time for a hybrid BQM solver is specified as a piecewise-linear
        curve defined by a set of floating-point pairs, the `minimum_time_limit`
        field under :attr:`~dwave.system.samplers.LeapHybridSampler.properties`.
        The first element in each pair is the number of problem variables; the
        second is the minimum required time. The minimum time for any number of
        variables is a linear interpolation calculated on two pairs that represent
        the relevant range for the given number of variables.

        Examples:
            For a solver where
            `LeapHybridSampler().properties["minimum_time_limit"]` returns
            `[[1, 0.1], [100, 10.0], [1000, 20.0]]`, the minimum time for a
            problem 50 variales is 5 seconds (the linear interpolation of the
            first two pairs that represent problems with between 1 to 100
            variables).
        """

        xx, yy = zip(*self.properties["minimum_time_limit"])
        return np.interp([bqm.num_variables], xx, yy)[0]

LeapHybridBQMSampler = LeapHybridSampler


class LeapHybridDQMSampler:
    """A class for using Leap's cloud-based hybrid DQM solvers.

    Leap’s quantum-classical hybrid DQM solvers are intended to solve arbitrary
    application problems formulated as **discrete** quadratic models (DQM).

    You can configure your :term:`solver` selection and usage by setting parameters,
    hierarchically, in a configuration file, as environment variables, or
    explicitly as input arguments, as described in
    `D-Wave Cloud Client <https://docs.ocean.dwavesys.com/en/stable/docs_cloud/sdk_index.html>`_.

    :ref:`dwave-cloud-client <sdk_index_cloud>`'s
    :meth:`~dwave.cloud.client.Client.get_solvers` method filters solvers you have
    access to by `solver properties <https://docs.dwavesys.com/docs/latest/c_solver_3.html>`_
    ``category=hybrid`` and ``supported_problem_type=dqm``. By default, online
    hybrid DQM solvers are returned ordered by latest ``version``.

    Args:
        **config:
            Keyword arguments passed to :meth:`dwave.cloud.client.Client.from_config`.

    Examples:
        This example solves a small, illustrative problem: a game of
        rock-paper-scissors. The DQM has two variables representing two hands,
        with cases for rock, paper, scissors. Quadratic biases are set to
        produce a lower value of the DQM for cases of variable ``my_hand``
        interacting with cases of variable ``their_hand`` such that the former
        wins over the latter; for example, the interaction of ``rock-scissors`` is
        set to -1 while ``scissors-rock`` is set to +1.

        >>> import dimod
        >>> from dwave.system import LeapHybridDQMSampler
        ...
        >>> cases = ["rock", "paper", "scissors"]
        >>> win = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
        ...
        >>> dqm = dimod.DiscreteQuadraticModel()
        >>> dqm.add_variable(3, label='my_hand')
        'my_hand'
        >>> dqm.add_variable(3, label='their_hand')
        'their_hand'
        >>> for my_idx, my_case in enumerate(cases):
        ...    for their_idx, their_case in enumerate(cases):
        ...       if win[my_case] == their_case:
        ...          dqm.set_quadratic('my_hand', 'their_hand',
        ...                            {(my_idx, their_idx): -1})
        ...       if win[their_case] == my_case:
        ...          dqm.set_quadratic('my_hand', 'their_hand',
        ...                            {(my_idx, their_idx): 1})
        ...
        >>> dqm_sampler = LeapHybridDQMSampler()      # doctest: +SKIP
        ...
        >>> sampleset = dqm_sampler.sample_dqm(dqm)   # doctest: +SKIP
        >>> print("{} beats {}".format(cases[sampleset.first.sample['my_hand']],
        ...                            cases[sampleset.first.sample['their_hand']]))   # doctest: +SKIP
        rock beats scissors

    """

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
            solver.setdefault('order_by', '-properties.version')

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
        :attr:`~dwave.system.samplers.LeapHybridDQMSampler.properties` for each
        key.

        `Solver parameters <https://docs.dwavesys.com/docs/latest/c_solver_3.html>`_
        are dependent on the selected solver and subject to change.
        """
        try:
            return self._parameters
        except AttributeError:
            parameters = {param: ['parameters']
                          for param in self.properties['parameters']}
            parameters.update(label=[])
            self._parameters = parameters
            return parameters

    def sample_dqm(self, dqm, time_limit=None, compress=False, compressed=None, **kwargs):
        """Sample from the specified discrete quadratic model.

        Args:
            dqm (:obj:`dimod.DiscreteQuadraticModel`):
                Discrete quadratic model (DQM).

            time_limit (int, optional):
                Maximum run time, in seconds, to allow the solver to work on the
                problem. Must be at least the minimum required for the number of
                problem variables, which is calculated and set by default.

                :meth:`~dwave.system.samplers.LeapHybridDQMSampler.min_time_limit`
                calculates (and describes) the minimum time for your problem.

            compress (binary, optional):
                Compresses the DQM data when set to True. Use if your problem
                somewhat exceeds the maximum allowed size. Compression tends to
                be slow and more effective on homogenous data, which in this
                case means it is more likely to help on DQMs with many identical
                integer-valued biases than ones with random float-valued biases,
                for example.

            compressed (binary, optional):
                Deprecated; please use ``compress`` instead.

            **kwargs:
                Optional keyword arguments for the solver, specified in
                :attr:`~dwave.system.samplers.LeapHybridDQMSampler.parameters`.

        Returns:
            :class:`dimod.SampleSet`: A sample set.

    Examples:
        See the example in :class:`LeapHybridDQMSampler`.

        """
        if time_limit is None:
            time_limit = self.min_time_limit(dqm)
        elif time_limit < self.min_time_limit(dqm):
            raise ValueError("the minimum time limit is {}s ({}s provided)"
                             "".format(self.min_time_limit(dqm), time_limit))

        # check the max time_limit if it's available
        if 'maximum_time_limit_hrs' in self.properties:
            if time_limit > 60*60*self.properties['maximum_time_limit_hrs']:
                raise ValueError("time_limit cannot exceed the solver maximum "
                                 "of {} hours ({} seconds given)".format(
                                    self.properties['maximum_time_limit_hrs'],
                                    time_limit))

        # we convert to a file here rather than let the cloud-client handle
        # it because we want to strip the labels and let the user handle
        # note: SpooledTemporaryFile currently returned by DQM.to_file
        # does not implement io.BaseIO interface, so we use the underlying
        # (and internal) file-like object for now

        if compressed is not None:
            warn(
                "Argument 'compressed' is deprecated and in future will raise an "
                "exception; please use 'compress' instead.",
                DeprecationWarning, stacklevel=2
                )
            compress = compressed or compress

        f = dqm.to_file(compress=compress, ignore_labels=True)._file
        sampleset = self.solver.sample_dqm(f, time_limit=time_limit, **kwargs).sampleset
        return sampleset.relabel_variables(dict(enumerate(dqm.variables)))

    def min_time_limit(self, dqm):
        """Return the minimum `time_limit` accepted for the given problem.

        The minimum time for a hybrid DQM solver is specified as a
        piecewise-linear curve defined by a set of floating-point pairs,
        the `minimum_time_limit` field under
        :attr:`~dwave.system.samplers.LeapHybridDQMSampler.properties`.
        The first element in each pair is a combination of the numbers of
        interactions, variables, and cases that reflects the "density" of
        connectivity between the problem's variables;
        the second is the minimum required time. The minimum time for any
        particular problem size is a linear interpolation calculated on
        two pairs that represent the relevant range for the given problem.

        Examples:
            For a solver where
            `LeapHybridDQMSampler().properties["minimum_time_limit"]` returns
            `[[1, 0.1], [100, 10.0], [1000, 20.0]]`, the minimum time for a
            problem of "density" 50 is 5 seconds (the linear interpolation of the
            first two pairs that represent problems with "density" between 1 to
            100).
        """
        ec = (dqm.num_variable_interactions() * dqm.num_cases() /
              max(dqm.num_variables(), 1))
        limits = np.array(self.properties['minimum_time_limit'])
        t = np.interp(ec, limits[:, 0], limits[:, 1])
        return max([5, t])
