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

from typing import Any, Dict, List, Optional

import numpy as np
from warnings import warn
from numbers import Number
from collections import abc

import dimod

try:
    # dimod 0.10.x
    from dimod.binary.binary_quadratic_model import BQM

    bqm_to_file = BQM.to_file
except ImportError:
    # dimod 0.9.x
    from dimod import AdjVectorBQM as BQM
    from dimod.serialization.fileview import FileView

    bqm_to_file = FileView

from dwave.cloud import Client
from dwave.system.utilities import classproperty, FeatureFlags


__all__ = ['LeapHybridSampler',
           'LeapHybridBQMSampler',
           'LeapHybridDQMSampler',
           'LeapHybridCQMSampler',
           ]


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
    access to by `solver properties <https://docs.dwavesys.com/docs/latest/c_solver_properties.html>`_
    ``category=hybrid`` and ``supported_problem_type=bqm``. By default, online
    hybrid BQM solvers are returned ordered by latest ``version``.

    The default specification for filtering and ordering solvers by features is
    available as :attr:`.default_solver` property. Explicitly specifying a
    solver in a configuration file, an environment variable, or keyword
    arguments overrides this specification. See the example below on how to
    extend it instead.

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
        >>> sampler = LeapHybridSampler()       # doctest: +SKIP
        >>> sampleset = sampler.sample(bqm)     # doctest: +SKIP

        This example specializes the default solver selection by filtering out
        bulk BQM solvers. (Bulk solvers are throughput-optimal for heavy/batch
        workloads, have a higher start-up latency, and are not well suited for
        live workloads. Not all Leap accounts have access to bulk solvers.)

        >>> from dwave.system import LeapHybridSampler
        ...
        >>> solver = LeapHybridSampler.default_solver
        >>> solver.update(name__regex=".*(?<!bulk)$")       # name shouldn't end with "bulk"
        >>> sampler = LeapHybridSampler(solver=solver)      # doctest: +SKIP
        >>> sampler.solver        # doctest: +SKIP
        BQMSolver(id='hybrid_binary_quadratic_model_version2')

    """

    _INTEGER_BQM_SIZE_THRESHOLD = 10000

    @classproperty
    def default_solver(cls):
        """dict: Features used to select the latest accessible hybrid BQM solver."""
        return dict(supported_problem_types__contains='bqm',
                    order_by='-properties.version')

    def __init__(self, **config):
        # strongly prefer hybrid solvers; requires kwarg-level override
        config.setdefault('client', 'hybrid')

        # default to short-lived session to prevent resets on slow uploads
        config.setdefault('connection_close', True)

        if FeatureFlags.hss_solver_config_override:
            # use legacy behavior (override solver config from env/file)
            solver = config.setdefault('solver', {})
            if isinstance(solver, abc.Mapping):
                solver.update(self.default_solver)

        # prefer the latest hybrid BQM solver available, but allow for an easy
        # override on any config level above the defaults (file/env/kwarg)
        defaults = config.setdefault('defaults', {})
        if not isinstance(defaults, abc.Mapping):
            raise TypeError("mapping expected for 'defaults'")
        defaults.update(solver=self.default_solver)

        self.client = Client.from_config(**config)
        self.solver = self.client.get_solver()

        # check user-specified solver conforms to our requirements
        if self.properties.get('category') != 'hybrid':
            raise ValueError("selected solver is not a hybrid solver.")
        if 'bqm' not in self.solver.supported_problem_types:
            raise ValueError("selected solver does not support the 'bqm' problem type.")

    @property
    def properties(self) -> Dict[str, Any]:
        """Solver properties as returned by a SAPI query.

        `Solver properties <https://docs.dwavesys.com/docs/latest/c_solver_properties.html>`_
        are dependent on the selected solver and subject to change.
        """
        try:
            return self._properties
        except AttributeError:
            self._properties = properties = self.solver.properties.copy()
            return properties

    @property
    def parameters(self) -> Dict[str, list]:
        """Solver parameters in the form of a dict, where keys are
        keyword parameters accepted by a SAPI query and values are lists of properties in
        :attr:`~dwave.system.samplers.LeapHybridSampler.properties` for each key.

        `Solver parameters <https://docs.dwavesys.com/docs/latest/c_solver_parameters.html>`_
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
            :class:`~dimod.SampleSet`: Sample set constructed from a (non-blocking)
            :class:`~concurrent.futures.Future`-like object.

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
        if not isinstance(bqm, BQM):
            bqm = BQM(bqm)

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
        with bqm_to_file(bqm, version=2) as fv:
            sapi_problem_id = self.solver.upload_bqm(fv).result()

        return self.solver.sample_bqm(sapi_problem_id, **kwargs).sampleset

    def _sample_large(self, bqm, **kwargs):
        """Sample from the unlabelled version of the BQM, then apply the
        labels to the returned sampleset.
        """
        with bqm_to_file(bqm, version=2, ignore_labels=True) as fv:
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
    access to by `solver properties <https://docs.dwavesys.com/docs/latest/c_solver_properties.html>`_
    ``category=hybrid`` and ``supported_problem_type=dqm``. By default, online
    hybrid DQM solvers are returned ordered by latest ``version``.

    The default specification for filtering and ordering solvers by features is
    available as :attr:`.default_solver` property. Explicitly specifying a
    solver in a configuration file, an environment variable, or keyword
    arguments overrides this specification. See the example in :class:`.LeapHybridSampler`
    on how to extend it instead.

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

    @classproperty
    def default_solver(self):
        """dict: Features used to select the latest accessible hybrid DQM solver."""
        return dict(supported_problem_types__contains='dqm',
                    order_by='-properties.version')

    def __init__(self, **config):
        # strongly prefer hybrid solvers; requires kwarg-level override
        config.setdefault('client', 'hybrid')

        # default to short-lived session to prevent resets on slow uploads
        config.setdefault('connection_close', True)

        if FeatureFlags.hss_solver_config_override:
            # use legacy behavior (override solver config from env/file)
            solver = config.setdefault('solver', {})
            if isinstance(solver, abc.Mapping):
                solver.update(self.default_solver)

        # prefer the latest hybrid DQM solver available, but allow for an easy
        # override on any config level above the defaults (file/env/kwarg)
        defaults = config.setdefault('defaults', {})
        if not isinstance(defaults, abc.Mapping):
            raise TypeError("mapping expected for 'defaults'")
        defaults.update(solver=self.default_solver)

        self.client = Client.from_config(**config)
        self.solver = self.client.get_solver()

        # check user-specified solver conforms to our requirements
        if self.properties.get('category') != 'hybrid':
            raise ValueError("selected solver is not a hybrid solver.")
        if 'dqm' not in self.solver.supported_problem_types:
            raise ValueError("selected solver does not support the 'dqm' problem type.")

    @property
    def properties(self) -> Dict[str, Any]:
        """Solver properties as returned by a SAPI query.

        `Solver properties <https://docs.dwavesys.com/docs/latest/c_solver_properties.html>`_
        are dependent on the selected solver and subject to change.
        """
        try:
            return self._properties
        except AttributeError:
            self._properties = properties = self.solver.properties.copy()
            return properties

    @property
    def parameters(self) -> Dict[str, list]:
        """Solver parameters in the form of a dict, where keys
        are keyword parameters accepted by a SAPI query and values are lists of
        properties in
        :attr:`~dwave.system.samplers.LeapHybridDQMSampler.properties` for each
        key.

        `Solver parameters <https://docs.dwavesys.com/docs/latest/c_solver_parameters.html>`_
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

    @dimod.decorators.nonblocking_sample_method
    def sample_dqm(self, dqm, time_limit=None, compress=False, compressed=None, **kwargs):
        """Sample from the specified discrete quadratic model.

        Args:
            dqm (:obj:`dimod.DiscreteQuadraticModel`):
                Discrete quadratic model (DQM).

                Note that if `dqm` is a :class:`dimod.CaseLabelDQM`, then
                :meth:`~dimod.CaseLabelDQM.map_sample` will need to be used to
                restore the case labels in the returned sample set.

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
            :class:`~dimod.SampleSet`: Sample set constructed from a (non-blocking)
            :class:`~concurrent.futures.Future`-like object.

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

        try:
            f = dqm.to_file(compress=compress, ignore_labels=True)._file
        except NotImplementedError:
            f = dimod.DQM.to_file(dqm, compress=compress, ignore_labels=True)._file

        future = self.solver.sample_dqm(f, time_limit=time_limit, **kwargs)
        yield future

        sampleset = future.sampleset.relabel_variables(dict(enumerate(dqm.variables)))

        if hasattr(dqm, 'offset') and dqm.offset:
            # dimod 0.10+
            # some versions of HSS don't account for the offset and it's hard
            # to tell which
            sampleset.record.energy = dqm.energies(sampleset)

        yield sampleset

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


class LeapHybridCQMSampler:
    """A class for using Leap's cloud-based hybrid CQM solvers.

    Leap’s quantum-classical hybrid CQM solvers are intended to solve
    application problems formulated as
    :ref:`constrained quadratic models (CQM) <cqm_sdk>`.

    You can configure your :term:`solver` selection and usage by setting parameters,
    hierarchically, in a configuration file, as environment variables, or
    explicitly as input arguments, as described in
    `D-Wave Cloud Client <https://docs.ocean.dwavesys.com/en/stable/docs_cloud/sdk_index.html>`_.

    :ref:`dwave-cloud-client <sdk_index_cloud>`'s
    :meth:`~dwave.cloud.client.Client.get_solvers` method filters solvers you have
    access to by `solver properties <https://docs.dwavesys.com/docs/latest/c_solver_properties.html>`_
    ``category=hybrid`` and ``supported_problem_type=cqm``. By default, online
    hybrid CQM solvers are returned ordered by latest ``version``.

    Args:
        **config:
            Keyword arguments passed to :meth:`dwave.cloud.client.Client.from_config`.

    Examples:
        This example solves a simple problem of finding the rectangle with the
        greatest area when the perimeter is limited. In this example, the
        perimeter of the rectangle is set to 8 (meaning the largest area is for
        the :math:`2X2` square).

        A CQM is created that will have two integer variables, :math:`i, j`, each
        limited to half the maximum perimeter length of 8, to represent the
        lengths of the rectangle's sides:

        >>> from dimod import ConstrainedQuadraticModel, Integer
        >>> i = Integer('i', upper_bound=4)
        >>> j = Integer('j', upper_bound=4)
        >>> cqm = ConstrainedQuadraticModel()

        The area of the rectangle is given by the multiplication of side :math:`i`
        by side :math:`j`. The goal is to maximize the area, :math:`i*j`. Because
        D-Wave samplers minimize, the objective should have its lowest value when
        this goal is met. Objective :math:`-i*j` has its minimum value when
        :math:`i*j`, the area, is greatest:

        >>> cqm.set_objective(-i*j)

        Finally, the requirement that the sum of both sides must not exceed the
        perimeter is represented as constraint :math:`2i + 2j <= 8`:

        >>> cqm.add_constraint(2*i+2*j <= 8, "Max perimeter")
        'Max perimeter'

        Instantiate a hybrid CQM sampler and submit the problem for solution by
        a remote solver provided by the Leap quantum cloud service:

        >>> from dwave.system import LeapHybridCQMSampler   # doctest: +SKIP
        >>> sampler = LeapHybridCQMSampler()                # doctest: +SKIP
        >>> sampleset = sampler.sample_cqm(cqm)             # doctest: +SKIP
        >>> print(sampleset.first)                          # doctest: +SKIP
        Sample(sample={'i': 2.0, 'j': 2.0}, energy=-4.0, num_occurrences=1,
        ...            is_feasible=True, is_satisfied=array([ True]))

        The best (lowest-energy) solution found has :math:`i=j=2` as expected,
        a solution that is feasible because all the constraints (one in this
        example) are satisfied.

    """
    def __init__(self, **config):
        # strongly prefer hybrid solvers; requires kwarg-level override
        config.setdefault('client', 'hybrid')

        # default to short-lived session to prevent resets on slow uploads
        config.setdefault('connection_close', True)

        if FeatureFlags.hss_solver_config_override:
            # use legacy behavior (override solver config from env/file)
            solver = config.setdefault('solver', {})
            if isinstance(solver, abc.Mapping):
                solver.update(self.default_solver)

        # prefer the latest hybrid CQM solver available, but allow for an easy
        # override on any config level above the defaults (file/env/kwarg)
        defaults = config.setdefault('defaults', {})
        if not isinstance(defaults, abc.Mapping):
            raise TypeError("mapping expected for 'defaults'")
        defaults.update(solver=self.default_solver)

        self.client = Client.from_config(**config)
        self.solver = self.client.get_solver()

        # For explicitly named solvers:
        if self.properties.get('category') != 'hybrid':
            raise ValueError("selected solver is not a hybrid solver.")
        if 'cqm' not in self.solver.supported_problem_types:
            raise ValueError("selected solver does not support the 'cqm' problem type.")

    @classproperty
    def default_solver(cls) -> Dict[str, str]:
        """Features used to select the latest accessible hybrid CQM solver."""
        return dict(supported_problem_types__contains='cqm',
                    order_by='-properties.version')

    @property
    def properties(self) -> Dict[str, Any]:
        """Solver properties as returned by a SAPI query.

        `Solver properties <https://docs.dwavesys.com/docs/latest/c_solver_properties.html>`_
        are dependent on the selected solver and subject to change.
        """
        try:
            return self._properties
        except AttributeError:
            self._properties = properties = self.solver.properties.copy()
            return properties

    @property
    def parameters(self) -> Dict[str, List[str]]:
        """Solver parameters in the form of a dict, where keys
        are keyword parameters accepted by a SAPI query and values are lists of
        properties in
        :attr:`~dwave.system.samplers.LeapHybridCQMSampler.properties` for each
        key.

        `Solver parameters <https://docs.dwavesys.com/docs/latest/c_solver_parameters.html>`_
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

    def sample_cqm(self, cqm: dimod.ConstrainedQuadraticModel,
                   time_limit: Optional[float] = None, **kwargs):
        """Sample from the specified constrained quadratic model.

        Args:
            cqm (:obj:`dimod.ConstrainedQuadraticModel`):
                Constrained quadratic model (CQM).

            time_limit (int, optional):
                Maximum run time, in seconds, to allow the solver to work on the
                problem. Must be at least the minimum required for the problem,
                which is calculated and set by default.

                :meth:`~dwave.system.samplers.LeapHybridCQMSampler.min_time_limit`
                calculates (and describes) the minimum time for your problem.

            **kwargs:
                Optional keyword arguments for the solver, specified in
                :attr:`~dwave.system.samplers.LeapHybridCQMSampler.parameters`.

        Returns:
            :class:`~dimod.SampleSet`: Sample set constructed from a (non-blocking)
            :class:`~concurrent.futures.Future`-like object.

        Examples:
            See the example in :class:`LeapHybridCQMSampler`.

        """

        if not isinstance(cqm, dimod.ConstrainedQuadraticModel):
            raise TypeError("first argument 'cqm' must be a ConstrainedQuadraticModel, "
                            f"recieved {type(cqm).__name__}")

        if time_limit is None:
            time_limit = self.min_time_limit(cqm)
        elif time_limit < self.min_time_limit(cqm):
            raise ValueError("the minimum time limit for this problem is "
                             f"{self.min_time_limit(cqm)} seconds "
                             f"({time_limit}s provided), "
                             "see .min_time_limit method")

        if len(cqm.constraints) > self.properties['maximum_number_of_constraints']:
            raise ValueError(
                "constrained quadratic model must have "
                f"{self.properties['maximum_number_of_constraints']} or fewer "
                f"constraints, given model has {len(cqm.constraints)}")

        if len(cqm.variables) > self.properties['maximum_number_of_variables']:
            raise ValueError(
                "constrained quadratic model must have "
                f"{self.properties['maximum_number_of_variables']} or fewer "
                f"variables, given model has {len(cqm.variables)}")

        if cqm.num_biases() > self.properties['maximum_number_of_biases']:
            raise ValueError(
                "constrained quadratic model must have "
                f"{self.properties['maximum_number_of_biases']} or fewer "
                f"biases, given model has {cqm.num_biases()}")

        if cqm.num_quadratic_variables() > self.properties['maximum_number_of_quadratic_variables']:
            raise ValueError(
                "constrained quadratic model must have "
                f"{self.properties['maximum_number_of_quadratic_variables']} "
                "or fewer variables with at least one quadratic bias across "
                "all constraints, given model has "
                f"{cqm.num_quadratic_variables()}")

        return self.solver.sample_cqm(cqm, time_limit=time_limit, **kwargs).sampleset

    def min_time_limit(self, cqm: dimod.ConstrainedQuadraticModel) -> float:
        """Return the minimum `time_limit` accepted for the given problem."""

        # todo: remove the hard-coded defaults
        num_variables_multiplier = self.properties.get('num_variables_multiplier', 1.57e-04)
        num_biases_multiplier = self.properties.get('num_biases_multiplier', 4.65e-06)
        num_constraints_multiplier = self.properties.get('num_constraints_multiplier', 6.44e-09)
        minimum_time_limit = self.properties['minimum_time_limit_s']

        num_variables = len(cqm.variables)
        num_constraints = len(cqm.constraints)
        num_biases = cqm.num_biases()

        return max(
            num_variables_multiplier * num_variables +
            num_biases_multiplier * num_biases +
            num_constraints_multiplier * num_variables * num_constraints,
            minimum_time_limit
            )
