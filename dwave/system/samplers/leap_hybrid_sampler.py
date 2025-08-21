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
:ref:`dimod <index_dimod>` :term:`samplers <sampler>` for the Leap service's
:term:`hybrid` solvers.
"""

import concurrent.futures
import warnings
from collections import abc
from numbers import Number
from typing import Any, Dict, List, NamedTuple, Optional

import dimod
import dwave.optimization
import numpy
from dwave.cloud.client import Client

from dwave.system.samplers import ResultInfoDict
from dwave.system.utilities import classproperty, FeatureFlags


__all__ = ['LeapHybridSampler',
           'LeapHybridBQMSampler',
           'LeapHybridDQMSampler',
           'LeapHybridCQMSampler',
           'LeapHybridNLSampler',
           ]


class _ScopedSamplerMixin(dimod.Scoped):
    """A mixin that implements ``close`` method to close the underlying cloud
    client. A default context manager that closes resources on exit is
    inherited from :class:`~dimod.Scoped`.
    """

    def close(self):
        """Close the underlying cloud client to release system resources such as
        threads.

        .. note::

            The method blocks for all the currently scheduled work (sampling
            requests) to finish.

        See: :meth:`~dwave.cloud.client.Client.close`.
        """
        self.client.close()


class LeapHybridSampler(_ScopedSamplerMixin, dimod.Sampler):
    r"""Submits binary quadratic models to a hybrid solver in the Leap service.

    The :term:`Leap` service's quantum-classical :term:`hybrid`
    :term:`binary quadratic model` (BQM) solvers are intended to solve arbitrary
    application problems formulated as BQMs.

    You can configure your :term:`solver` selection as described in the
    :ref:`cloud_configuration` section.\ [#]_

    Args:
        **config:
            Keyword arguments passed to
            :meth:`~dwave.cloud.client.Client.from_config`.

    Examples:
        This example samples a randomly generated binary quadratic model with 10
        variables and 15 interactions.

        >>> from dimod.generators import gnm_random_bqm
        >>> from dwave.system import LeapHybridSampler
        ...
        >>> bqm = gnm_random_bqm(10, 15, 'SPIN')
        >>> with LeapHybridSampler() as sampler:    # doctest: +SKIP
        ...     sampleset = sampler.sample(bqm)

    .. [#]
        :ref:`dwave-cloud-client <index_cloud>`'s
        :meth:`~dwave.cloud.client.Client.get_solvers` method filters solvers
        you have access to by
        :ref:`solver properties <opt_solver_bqm_properties>` ``category=hybrid``
        and ``supported_problem_type=bqm``. By default, online hybrid BQM
        solvers are returned ordered by latest ``version``.

        The default specification for filtering and ordering solvers by features
        is available as :attr:`.default_solver` property. Explicitly specifying
        a solver in a configuration file, an environment variable, or keyword
        arguments overrides this specification.

    """

    _INTEGER_BQM_SIZE_THRESHOLD = 10000

    @classproperty
    def default_solver(cls):
        """dict: Features used to select the latest accessible hybrid BQM solver.
        """
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
        """Solver properties as returned by a :term:`SAPI` query.

        :ref:`Solver properties <opt_solver_bqm_properties>` are dependent on
        the selected solver and subject to change; for example, new features may
        add properties.
        """
        try:
            return self._properties
        except AttributeError:
            self._properties = properties = self.solver.properties.copy()
            return properties

    @property
    def parameters(self) -> Dict[str, list]:
        """Solver parameters as returned by a :term:`SAPI` query.

        Keys of the returned dict are keyword parameters accepted by a SAPI
        query and values are lists of properties in
        :attr:`~dwave.system.samplers.LeapHybridSampler.properties` for each key.

        :ref:`Solver parameters <opt_solver_bqm_properties>`
        are dependent on the selected solver and subject to change; for example,
        new features may add parameters.
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
            bqm (:obj:`~dimod.binary.BinaryQuadraticModel`):
                :term:`Binary quadratic model`.

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
            This example samples a randomly generated binary quadratic model
            with 10 variables and 15 interactions.

            >>> from dimod.generators import gnm_random_bqm
            >>> from dwave.system import LeapHybridSampler
            ...
            >>> bqm = gnm_random_bqm(10, 15, 'SPIN')
            >>> with LeapHybridSampler() as sampler:    # doctest: +SKIP
            ...     sampleset = sampler.sample(bqm)
        """

        if not isinstance(bqm, dimod.BQM):
            bqm = dimod.BQM(bqm)

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
        with bqm.to_file(version=2) as fv:
            sapi_problem_id = self.solver.upload_bqm(fv).result()

        return self.solver.sample_bqm(sapi_problem_id, **kwargs).sampleset

    def _sample_large(self, bqm, **kwargs):
        """Sample from the unlabelled version of the BQM, then apply the
        labels to the returned sampleset.
        """
        with bqm.to_file(version=2, ignore_labels=True) as fv:
            sapi_problem_id = self.solver.upload_bqm(fv).result()

        sampleset = self.solver.sample_bqm(sapi_problem_id, **kwargs).sampleset

        # relabel, as of dimod 0.9.5+ this is not blocking
        mapping = dict(enumerate(bqm.variables))
        return sampleset.relabel_variables(mapping)

    def min_time_limit(self, bqm):
        """Return the minimum ``time_limit`` accepted for the given problem.

        The minimum time for a hybrid BQM solver is specified as a
        piecewise-linear curve defined by a set of floating-point pairs,
        the :ref:`property_bqm_minimum_time_limit` property.

        Args:
            bqm (:class:`~dimod.binary.BinaryQuadraticModel`):
                A :term:`binary quadratic model`.

        Examples:
            For a solver where
            ``LeapHybridSampler().properties["minimum_time_limit"]`` returns
            ``[[1, 0.1], [100, 10.0], [1000, 20.0]]``, the minimum time for a
            problem of 50 variables is 5 seconds (the linear interpolation of
            the first two pairs that represent problems with between 1 to 100
            variables).
        """

        xx, yy = zip(*self.properties["minimum_time_limit"])
        return numpy.interp([bqm.num_variables], xx, yy)[0]

LeapHybridBQMSampler = LeapHybridSampler


class LeapHybridDQMSampler(_ScopedSamplerMixin):
    r"""Submits discrete quadratic models to a hybrid solver in the Leap service.

    The :term:`Leap` service's quantum-classical :term:`hybrid`
    :term:`discrete quadratic model` (DQM) solvers are intended to solve
    arbitrary application problems formulated as DQMs.

    You can configure your :term:`solver` selection as described in the
    :ref:`cloud_configuration` section.\ [#]_

    Args:
        **config:
            Keyword arguments passed to
            :meth:`~dwave.cloud.client.Client.from_config`.

    Examples:
        This example solves a small, illustrative problem: a game of
        rock-paper-scissors. The DQM has two variables representing two hands,
        with cases for rock, paper, scissors. Quadratic biases are set to
        produce a lower value of the DQM for cases of variable ``my_hand``
        interacting with cases of variable ``their_hand`` such that the former
        wins over the latter; for example, the interaction of ``rock-scissors``
        is set to -1 while ``scissors-rock`` is set to +1.

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
        >>> with LeapHybridDQMSampler() as dqm_sampler:     # doctest: +SKIP
        ...     sampleset = dqm_sampler.sample_dqm(dqm)
        ...     print(f"{} beats {}".format(cases[sampleset.first.sample['my_hand']],
        ...                                 cases[sampleset.first.sample['their_hand']]))
        rock beats scissors

    .. [#]
        :ref:`dwave-cloud-client <index_cloud>`'s
        :meth:`~dwave.cloud.client.Client.get_solvers` method filters solvers
        you have access to by
        :ref:`solver properties <opt_solver_dqm_properties>` ``category=hybrid``
        and ``supported_problem_type=dqm``. By default, online hybrid DQM
        solvers are returned ordered by latest ``version``.

        The default specification for filtering and ordering solvers by features
        is available as :attr:`.default_solver` property. Explicitly specifying
        a solver in a configuration file, an environment variable, or keyword
        arguments overrides this specification.
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
        """Solver properties as returned by a :term:`SAPI` query.

        :ref:`Solver properties <opt_solver_dqm_properties>` are dependent on
        the selected solver and subject to change; for example, new features may
        add properties.
        """
        try:
            return self._properties
        except AttributeError:
            self._properties = properties = self.solver.properties.copy()
            return properties

    @property
    def parameters(self) -> Dict[str, list]:
        """Solver parameters as returned by a :term:`SAPI` query.

        Keys of the returned dict are keyword parameters accepted by a SAPI
        query and values are lists of properties in
        :attr:`~dwave.system.samplers.LeapHybridDQMSampler.properties` for each
        key.

        :ref:`Solver parameters <opt_solver_dqm_properties>`
        are dependent on the selected solver and subject to change; for example,
        new features may add parameters.
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
            dqm (:class:`~dimod.DiscreteQuadraticModel`):
                Discrete quadratic model (:term:`DQM`).
                If ``dqm`` is a :class:`~dimod.CaseLabelDQM` class, use the
                :meth:`~dimod.CaseLabelDQM.map_sample` method to restore case
                labels in the returned sample set.

            time_limit (int, optional):
                Maximum run time, in seconds, to allow the solver to work on the
                problem. Must be at least the minimum required for the number of
                problem variables, which is calculated and set by default.
                :meth:`~dwave.system.samplers.LeapHybridDQMSampler.min_time_limit`
                calculates (and describes) the minimum time for your problem.

            compress (binary, optional):
                Compresses the DQM data when set to True. Use if your problem
                somewhat exceeds the maximum allowed size. Compression tends to
                be slow and more effective on homogenous data; for example, it
                is more likely to help on DQMs with many identical
                integer-valued biases than ones with random float-valued biases.

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
            warnings.warn(
                "Argument 'compressed' is deprecated and in future will raise an "
                "exception; please use 'compress' instead.",
                DeprecationWarning, stacklevel=2
            )
            compress = compressed or compress

        with dqm.to_file(compress=compress, ignore_labels=True) as f:
            sapi_problem_id = self.solver.upload_problem(f).result()

        future = self.solver.sample_dqm(sapi_problem_id, time_limit=time_limit, **kwargs)

        yield future

        sampleset = future.sampleset.relabel_variables(dict(enumerate(dqm.variables)))

        if hasattr(dqm, 'offset') and dqm.offset:
            # dimod 0.10+
            # some versions of HSS don't account for the offset and it's hard
            # to tell which
            sampleset.record.energy = dqm.energies(sampleset)

        yield sampleset

    def min_time_limit(self, dqm):
        """Return the minimum ``time_limit`` accepted for the given problem.

        The minimum time for a hybrid DQM solver is specified as a
        piecewise-linear curve defined by a set of floating-point pairs,
        the :ref:`property_dqm_minimum_time_limit` property.

        Args:
            dqm (:class:`~dimod.DiscreteQuadraticModel`):
                A :term:`discrete quadratic model`.

        Examples:
            For a solver where
            ``LeapHybridDQMSampler().properties["minimum_time_limit"]`` returns
            ``[[1, 0.1], [100, 10.0], [1000, 20.0]]``, the minimum time for a
            problem of "density" 50 is 5 seconds (the linear interpolation of
            the first two pairs that represent problems with "density" between 1
            to 100).
        """
        ec = (dqm.num_variable_interactions() * dqm.num_cases() /
              max(dqm.num_variables(), 1))
        limits = numpy.array(self.properties['minimum_time_limit'])
        t = numpy.interp(ec, limits[:, 0], limits[:, 1])
        return max([5, t])


class LeapHybridCQMSampler(_ScopedSamplerMixin):
    r"""Submits constrained quadratic models to a hybrid solver in the Leap service.

    The :term:`Leap` service's quantum-classical :term:`hybrid`
    :term:`constrained quadratic model` (CQM) solvers are intended to solve
    arbitrary application problems formulated as CQMs.

    You can configure your :term:`solver` selection as described in the
    :ref:`cloud_configuration` section.\ [#]_

    Args:
        **config:
            Keyword arguments passed to
            :meth:`~dwave.cloud.client.Client.from_config`.

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
        >>> with LeapHybridCQMSampler() as sampler:         # doctest: +SKIP
        ...     sampleset = sampler.sample_cqm(cqm)
        ...     print(sampleset.first)
        Sample(sample={'i': 2.0, 'j': 2.0}, energy=-4.0, num_occurrences=1,
        ...            is_feasible=True, is_satisfied=array([ True]))

        The best (lowest-energy) solution found has :math:`i=j=2` as expected,
        a solution that is feasible because all the constraints (one in this
        example) are satisfied.

    .. [#]
        :ref:`dwave-cloud-client <index_cloud>`'s
        :meth:`~dwave.cloud.client.Client.get_solvers` method filters solvers
        you have access to by
        :ref:`solver properties <opt_solver_cqm_properties>` ``category=hybrid``
        and ``supported_problem_type=cqm``. By default, online hybrid CQM
        solvers are returned ordered by latest ``version``.

        The default specification for filtering and ordering solvers by features
        is available as :attr:`.default_solver` property. Explicitly specifying
        a solver in a configuration file, an environment variable, or keyword
        arguments overrides this specification.
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
        """Solver properties as returned by a :term:`SAPI` query.

        :ref:`Solver properties <opt_solver_cqm_properties>`
        are dependent on the selected solver and subject to change; for example,
        new features may add properties.
        """
        try:
            return self._properties
        except AttributeError:
            self._properties = properties = self.solver.properties.copy()
            return properties

    @property
    def parameters(self) -> Dict[str, List[str]]:
        """Solver parameters as returned by a :term:`SAPI` query.

        Keys of the returned dict are keyword parameters accepted by a SAPI
        query and values are lists of properties in
        :attr:`~dwave.system.samplers.LeapHybridCQMSampler.properties` for each
        key.

        :ref:`Solver parameters <opt_solver_cqm_properties>` are dependent on
        the selected solver and subject to change; for example, new features may
        add parameters.
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
            cqm (:obj:`~dimod.ConstrainedQuadraticModel`):
                Constrained quadratic model (:term:`CQM`).

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

        # developer note: this is a temporary fix until
        # https://github.com/dwavesystems/dimod/issues/1303 is fixed
        # and should be reverted afterwards
        with cqm.to_file() as fcqm:
            data = dimod.serialization.fileview.read_header(
                fcqm,
                dimod.constrained.CQM_MAGIC_PREFIX,
                ).data

            class _cqm:
                # a fake CQM that has the same properties as the real thing,
                # except it reports the num_biases of the serialized model.
                # To remove once https://github.com/dwavesystems/dimod/issues/1303
                # is fixed.
                variables = cqm.variables
                constraints = cqm.constraints

                @staticmethod
                def num_biases():
                    return data['num_biases']

            if time_limit is None:
                time_limit = self.min_time_limit(_cqm)
            elif time_limit < self.min_time_limit(_cqm):
                raise ValueError("the minimum time limit for this problem is "
                                 f"{self.min_time_limit(_cqm)} seconds "
                                 f"({time_limit}s provided), "
                                 "see .min_time_limit method")

            contact_sales_str = "Contact D-Wave at sales@dwavesys.com if your " + \
                                "application requires scale or performance that " + \
                                "exceeds the currently advertised capabilities of " + \
                                "this hybrid solver."

            if len(cqm.constraints) > self.properties['maximum_number_of_constraints']:
                raise ValueError(
                    "constrained quadratic model must have "
                    f"{self.properties['maximum_number_of_constraints']} or fewer "
                    f"constraints; given model has {len(cqm.constraints)}. "
                    f"{contact_sales_str}")

            if len(cqm.variables) > self.properties['maximum_number_of_variables']:
                raise ValueError(
                    "constrained quadratic model must have "
                    f"{self.properties['maximum_number_of_variables']} or fewer "
                    f"variables; given model has {len(cqm.variables)}. "
                    f"{contact_sales_str}")

            if _cqm.num_biases() > self.properties['maximum_number_of_biases']:
                raise ValueError(
                    "constrained quadratic model must have "
                    f"{self.properties['maximum_number_of_biases']} or fewer "
                    f"biases; given model has {cqm.num_biases()}. "
                    f"{contact_sales_str}")

            if cqm.num_quadratic_variables(include_objective=False) > self.properties['maximum_number_of_quadratic_variables']:
                raise ValueError(
                    "constrained quadratic model must have "
                    f"{self.properties['maximum_number_of_quadratic_variables']} "
                    "or fewer variables with at least one quadratic bias across "
                    "all constraints; given model has "
                    f"{cqm.num_quadratic_variables()}. "
                    f"{contact_sales_str}")

            sapi_problem_id = self.solver.upload_problem(fcqm).result()

        return self.solver.sample_cqm(sapi_problem_id, time_limit=time_limit, **kwargs).sampleset

    def min_time_limit(self, cqm: dimod.ConstrainedQuadraticModel) -> float:
        """Return the minimum ``time_limit``, in seconds, accepted for the given
        problem.

        This minimum runtime is always at least the minimum specified by the CQM
        solver's :ref:`property_cqm_minimum_time_limit_s` property. As the size
        and complexity of the CQM increases, the minimum runtime may increase.
        This method calculates, for the given CQM, the minimum runtime as a
        function of its number of variables, constraints, and biases, weighted
        by solver properties such as
        :ref:`property_cqm_num_variables_multiplier` and others described in the
        :ref:`opt_solver_cqm_properties` section. See the code for the
        calculation.

        Args:
            cqm (:class:`~dimod.ConstrainedQuadraticModel`):
                A :term:`constrained quadratic model`.

        Examples:
            This example generates a small CQM that requires only the minimum
            runtime of the :ref:`property_cqm_minimum_time_limit_s` property and
            a more complex CQM that requires a larger minimum ``time_limit``.

            >>> from dimod.generators import bin_packing
            >>> from dwave.system import LeapHybridCQMSampler
            ...
            >>> cqm = bin_packing([5]*5, 10)
            >>> sampler.min_time_limit(cqm) > sampler.properties["minimum_time_limit_s"]  # doctest: +SKIP
            False
            >>> cqm = bin_packing([5]*5, 10)
            >>> sampler.min_time_limit(cqm) > sampler.properties["minimum_time_limit_s"]  # doctest: +SKIP
            True
        """

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


class LeapHybridNLSampler(_ScopedSamplerMixin):
    r"""Submits nonlinear models to a hybrid solver in the Leap service.

    The :term:`Leap` service's quantum-classical :term:`hybrid`
    :term:`nonlinear model` solvers are intended to solve arbitrary application
    problems formulated as :ref:`nonlinear models <concept_models_nonlinear>`.

    You can configure your :term:`solver` selection as described in the
    :ref:`cloud_configuration` section.\ [#]_

    Args:
        **config:
            Keyword arguments passed to
            :meth:`~dwave.cloud.client.Client.from_config`.

    Examples:
        This example submits a model for a
        :func:`flow-shop-scheduling <dwave.optimization.generators.flow_shop_scheduling>`
        problem.

        >>> from dwave.optimization.generators import flow_shop_scheduling
        >>> from dwave.system import LeapHybridNLSampler
        ...
        >>> with LeapHybridNLSampler() as sampler:      # doctest: +SKIP
        ...     processing_times = [[10, 5, 7], [20, 10, 15]]
        ...     model = flow_shop_scheduling(processing_times=processing_times)
        ...     results = sampler.sample(model, label="Small FSS problem")
        ...     job_order = next(model.iter_decisions())
        ...     print(f"State 0 of {model.objective.state_size()} has an "
        ...           f"objective value {model.objective.state(0)} for order "
        ...           f"{job_order.state(0)}.")
        State 0 of 8 has an objective value 50.0 for order [1. 2. 0.].

    .. [#]
        :ref:`dwave-cloud-client <index_cloud>`'s
        :meth:`~dwave.cloud.client.Client.get_solvers` method filters solvers
        you have access to by
        :ref:`solver properties <opt_solver_nl_properties>` ``category=hybrid``
        and ``supported_problem_type=nl``. By default, online hybrid NL
        solvers are returned ordered by latest ``version``.

        The default specification for filtering and ordering solvers by features
        is available as :attr:`.default_solver` property. Explicitly specifying
        a solver in a configuration file, an environment variable, or keyword
        arguments overrides this specification.

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

        # prefer the latest hybrid NL solver available, but allow for an easy
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
        if 'nl' not in self.solver.supported_problem_types:
            raise ValueError("selected solver does not support the 'nl' problem type.")

        self._executor = concurrent.futures.ThreadPoolExecutor()

    def close(self):
        """Close the underlying cloud client to release system resources such as
        threads.

        The method blocks for all the currently scheduled work (sampling
        requests) to finish.

        See also:
            :meth:`~dwave.cloud.client.Client.close`.
        """
        super().close()
        self._executor.shutdown()

    @classproperty
    def default_solver(cls) -> Dict[str, str]:
        """Features used to select the latest accessible hybrid nonlinear-model solver."""
        return dict(supported_problem_types__contains='nl',
                    order_by='-properties.version')

    @property
    def properties(self) -> Dict[str, Any]:
        """Solver properties as returned by a :term:`SAPI` query.

        :ref:`Solver properties <opt_solver_nl_properties>`
        are dependent on the selected solver and subject to change; for example,
        new features may add properties.
        """
        try:
            return self._properties
        except AttributeError:
            self._properties = properties = self.solver.properties.copy()
            return properties

    @property
    def parameters(self) -> Dict[str, List[str]]:
        """Solver parameters as returned by a :term:`SAPI` query.

        Keys of the returned dict are keyword parameters accepted by a SAPI
        query and values are lists of properties in
        :attr:`~dwave.system.samplers.LeapHybridNLSampler.properties` for each
        key.

        :ref:`Solver parameters <opt_solver_nl_properties>` are dependent on
        the selected solver and subject to change; for example, new features may
        add parameters.
        """
        try:
            return self._parameters
        except AttributeError:
            parameters = {param: ['parameters']
                          for param in self.properties['parameters']}
            parameters.update(label=[])
            self._parameters = parameters
            return parameters

    class SampleResult(NamedTuple):
        model: dwave.optimization.Model
        info: ResultInfoDict

    def sample(self, model: dwave.optimization.Model,
               time_limit: Optional[float] = None, **kwargs
               ) -> 'concurrent.futures.Future[SampleResult]':
        """Sample from the specified nonlinear model.

        Args:
            model (:class:`~dwave.optimization.model.Model`):
                Nonlinear model.

            time_limit (float, optional):
                Maximum runtime, in seconds, the solver should work on the
                problem. Should be at least the estimated minimum required for
                the problem, which is calculated and set by default.
                :meth:`~dwave.system.samplers.LeapHybridNLSampler.estimated_min_time_limit`
                estimates the minimum time for your problem.  For ``time_limit``
                values shorter than the estimated minimum, runtime (and charge
                time) is not guaranteed to be shorter than the estimated time.

            **kwargs:
                Optional keyword arguments for the solver, specified in
                :attr:`~dwave.system.samplers.LeapHybridNLSampler.parameters`.

        Returns:
            :class:`~concurrent.futures.Future` [SampleResult]:
                Named tuple, in a Future, containing the nonlinear model and
                general result information such as timing and the identity of
                the problem data.

        .. versionchanged:: 1.31.0
            The return value includes timing information as part of the ``info``
            field dictionary, which now replaces the previous ``timing`` field.
        """

        if not isinstance(model, dwave.optimization.Model):

            raise TypeError("first argument 'model' must be a dwave.optimization.Model, "
                            f"received {type(model).__name__}")

        if time_limit is None:
            time_limit = self.estimated_min_time_limit(model)

        num_states = len(model.states)
        max_num_states = min(
            self.solver.properties.get("maximum_number_of_states", num_states),
            num_states
        )
        problem_data_id = self.solver.upload_nlm(model, max_num_states=max_num_states).result()

        future = self.solver.sample_nlm(problem_data_id, time_limit=time_limit, **kwargs)

        def hook(model, future):
            # TODO: known dwave-optimization bug, don't check header for now
            model.states.from_file(future.answer_data, check_header=False)

        model.states.from_future(future, hook)

        def collect():
            timing = future.timing.copy()
            info = dict(
                timing=timing,
                warnings=timing.pop('warnings', []),
                # match SampleSet.info fields (see :meth:`~dwave.cloud.computation.Future._get_problem_info`)
                problem_id=future.id,
                problem_label=future.label,
                problem_data_id=problem_data_id,
            )
            for msg in info['warnings']:
                # note: no point using stacklevel, as this is a different thread
                warnings.warn(msg, category=UserWarning)

            return LeapHybridNLSampler.SampleResult(model, info)

        result = self._executor.submit(collect)

        return result

    def estimated_min_time_limit(self, nlm: dwave.optimization.Model) -> float:
        """Return the minimum required time, in seconds, estimated for the given
        problem.

        Runtime (and charge time) is not guaranteed to be shorter than this
        minimum time.

        Args:
            nlm (:class:`~dwave.optimization.model.Model`):
                A :term:`nonlinear model`.
        """

        num_nodes_multiplier = self.properties.get('num_nodes_multiplier', 8.306792043756981e-05)
        state_size_multiplier = self.properties.get('state_size_multiplier', 2.8379674360396316e-10)
        num_nodes_state_size_multiplier = self.properties.get('num_nodes_state_size_multiplier', 2.1097317822863966e-12)
        offset = self.properties.get('offset', 0.012671678446550175)
        min_time_limit = self.properties.get('min_time_limit', 5)

        nn = nlm.num_nodes()
        ss = nlm.state_size()

        return max(
            num_nodes_multiplier * nn
            + state_size_multiplier * ss
            + num_nodes_state_size_multiplier * nn * ss
            + offset,
            min_time_limit
        )
