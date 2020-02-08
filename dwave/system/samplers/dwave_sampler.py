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
#
# ================================================================================================
"""
A :std:doc:`dimod sampler <oceandocs:docs_dimod/reference/samplers>` for the D-Wave system.

See :std:doc:`Ocean Glossary <oceandocs:glossary>`
for explanations of technical terms in descriptions of Ocean tools.
"""
from __future__ import division

import functools
import time

from warnings import warn

import dimod

from dimod.exceptions import BinaryQuadraticModelStructureError
from dwave.cloud import Client
from dwave.cloud.exceptions import SolverOfflineError, SolverNotFoundError

from dwave.system.warnings import WarningHandler, WarningAction

__all__ = ['DWaveSampler']


def _failover(f):
    @functools.wraps(f)
    def wrapper(sampler, *args, **kwargs):
        while True:
            try:
                return f(sampler, *args, **kwargs)
            except SolverOfflineError as err:
                if not sampler.failover:
                    raise err

            try:
                # the requested features are saved on the client object, so
                # we just need to request a new solver
                sampler.solver = sampler.client.get_solver()

                # delete the lazily-constructed attributes
                try:
                    del sampler._edgelist
                except AttributeError:
                    pass

                try:
                    del sampler._nodelist
                except AttributeError:
                    pass

                try:
                    del sampler._parameters
                except AttributeError:
                    pass

                try:
                    del sampler._properties
                except AttributeError:
                    pass

            except SolverNotFoundError as err:
                if sampler.retry_interval < 0:
                    raise err

                time.sleep(sampler.retry_interval)
    return wrapper


class DWaveSampler(dimod.Sampler, dimod.Structured):
    """A class for using the D-Wave system as a sampler.

    Uses parameters set in a configuration file, as environment variables, or
    explicitly as input arguments for selecting and communicating with a D-Wave
    system. For more information, see
    `D-Wave Cloud Client <https://docs.ocean.dwavesys.com/projects/cloud-client/en/latest/>`_.

    Inherits from :class:`dimod.Sampler` and :class:`dimod.Structured`.

    Args:
        failover (bool, optional, default=False):
            Switch to a new QPU in the rare event that the currently connected
            system goes offline. Note that different QPUs may have different
            hardware graphs and a failover will result in a regenerated
            :attr:`.nodelist`, :attr:`.edgelist`, :attr:`.properties` and
            :attr:`.parameters`.

        retry_interval (number, optional, default=-1):
            The amount of time (in seconds) to wait to poll for a solver in
            the case that no solver is found. If `retry_interval` is negative
            then it will instead propogate the `SolverNotFoundError` to the
            user.

        config_file (str, optional):
            Path to a configuration file that identifies a D-Wave system and provides
            connection information.

        profile (str, optional):
            Profile to select from the configuration file.

        endpoint (str, optional):
            D-Wave API endpoint URL.

        token (str, optional):
            Authentication token for the D-Wave API to authenticate the client session.

        solver (dict/str, optional):
            Solver (a D-Wave system on which to run submitted problems) to select given
            as a set of required features. Supported features and values are described in
            :meth:`~dwave.cloud.client.Client.get_solvers`. For backward
            compatibility, a solver name, formatted as a string, is accepted.

        proxy (str, optional):
            Proxy URL to be used for accessing the D-Wave API.

        **config:
            Keyword arguments passed directly to :meth:`~dwave.cloud.client.Client.from_config`.

    Examples:
        This example submits a two-variable Ising problem mapped directly to qubits 0
        and 1 on a D-Wave system selected by explicitly requiring that it have these two
        active qubits. Other required parameters for communication with the system, such
        as its URL and an autentication token, are implicitly set in a configuration file
        or as environment variables, as described in
        `Configuring a D-Wave System <https://docs.ocean.dwavesys.com/en/latest/overview/dwavesys.html>`_.

        >>> from dwave.system.samplers import DWaveSampler
        >>> sampler = DWaveSampler(solver={'qubits__issuperset': {0, 1}})
        >>> sampleset = sampler.sample_ising({0: -1, 1: 1}, {})
        >>> for sample in sampleset.samples():  # doctest: +SKIP
        ...    print(sample)
        ...
        {0: 1, 1: -1}

    See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_
    for explanations of technical terms in descriptions of Ocean tools.

    """
    def __init__(self, failover=False, retry_interval=-1, **config):

        if config.get('solver_features') is not None:
            warn("'solver_features' argument has been renamed to 'solver'.", DeprecationWarning)

            if config.get('solver') is not None:
                raise ValueError("can not combine 'solver' and 'solver_features'")

            config['solver'] = config.pop('solver_features')

        self.client = Client.from_config(**config)
        self.solver = self.client.get_solver()

        self.failover = failover
        self.retry_interval = retry_interval

    warnings_default = WarningAction.IGNORE
    """Defines the default behabior for :meth:`.sample_ising`'s  and
    :meth:`sample_qubo`'s `warnings` kwarg.
    """

    @property
    def properties(self):
        """dict: D-Wave solver properties as returned by a SAPI query.

        Solver properties are dependent on the selected D-Wave solver and subject to change;
        for example, new released features may add properties.
        `D-Wave System Documentation <https://docs.dwavesys.com/docs/latest/doc_solver_ref.html>`_
        describes the parameters and properties supported on the D-Wave system.

        Examples:

            >>> from dwave.system.samplers import DWaveSampler
            >>> sampler = DWaveSampler()
            >>> sampler.properties    # doctest: +SKIP
            {u'anneal_offset_ranges': [[-0.2197463755538704, 0.03821687759418928],
              [-0.2242514597680286, 0.01718456460967399],
              [-0.20860153999435985, 0.05511969218508182],
            # Snipped above response for brevity

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_
        for explanations of technical terms in descriptions of Ocean tools.

        """
        try:
            return self._properties
        except AttributeError:
            self._properties = properties = self.solver.properties.copy()
            return properties

    @property
    def parameters(self):
        """dict[str, list]: D-Wave solver parameters in the form of a dict, where keys are
        keyword parameters accepted by a SAPI query and values are lists of properties in
        :attr:`.DWaveSampler.properties` for each key.

        Solver parameters are dependent on the selected D-Wave solver and subject to change;
        for example, new released features may add parameters.
        `D-Wave System Documentation <https://docs.dwavesys.com/docs/latest/doc_solver_ref.html>`_
        describes the parameters and properties supported on the D-Wave system.

        Examples:

            >>> from dwave.system.samplers import DWaveSampler
            >>> sampler = DWaveSampler()
            >>> sampler.parameters    # doctest: +SKIP
            {u'anneal_offsets': ['parameters'],
            u'anneal_schedule': ['parameters'],
            u'annealing_time': ['parameters'],
            u'answer_mode': ['parameters'],
            u'auto_scale': ['parameters'],
            # Snipped above response for brevity

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_
        for explanations of technical terms in descriptions of Ocean tools.

        """
        try:
            return self._parameters
        except AttributeError:
            parameters = {param: ['parameters']
                          for param in self.properties['parameters']}
            parameters.update(warnings=[])
            self._parameters = parameters
            return parameters

    @property
    def edgelist(self):
        """list: List of active couplers for the D-Wave solver.

        Examples:

            >>> from dwave.system.samplers import DWaveSampler
            >>> sampler = DWaveSampler()
            >>> sampler.edgelist    # doctest: +SKIP
            [(0, 4),
             (0, 5),
             (0, 6),
             (0, 7),
             (0, 128),
             (1, 4),
            # Snipped above response for brevity

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_
        for explanations of technical terms in descriptions of Ocean tools.

        """
        # Assumption: cloud client nodes are always integer-labelled
        try:
            edgelist = self._edgelist
        except AttributeError:
            self._edgelist = edgelist = sorted(set((u, v) if u < v else (v, u)
                                                   for u, v in self.solver.edges))
        return edgelist

    @property
    def nodelist(self):
        """list: List of active qubits for the D-Wave solver.

        Examples:

            >>> from dwave.system.samplers import DWaveSampler
            >>> sampler = DWaveSampler()
            >>> sampler.nodelist    # doctest: +SKIP
            [0,
             1,
             2,
            # Snipped above response for brevity

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_
        for explanations of technical terms in descriptions of Ocean tools.

        """
        # Assumption: cloud client nodes are always integer-labelled
        try:
            nodelist = self._nodelist
        except AttributeError:
            self._nodelist = nodelist = sorted(self.solver.nodes)
        return nodelist

    @_failover
    def sample_ising(self, h, J, warnings=None, **kwargs):
        """Sample from the specified Ising model.

        Args:
            h (dict/list):
                Linear biases of the Ising model. If a dict, should be of the
                form `{v: bias, ...}` where `v` is a spin-valued variable and
                `bias` is its associated bias. If a list, it is treated as a
                list of biases where the indices are the variable labels,
                except in the case of missing qubits in which case 0 biases are
                ignored while a non-zero bias set on a missing qubit raises an
                error.

            J (dict[(int, int): float]):
                Quadratic biases of the Ising model.

            warnings (:class:`~dwave.system.warnings.WarningAction`, optional):
                Defines what warning action to take, if any. See
                :mod:`~dwave.system.warnings`. The default behaviour is defined
                by :attr:`warnings_default`, which itself defaults to
                :class:`~dwave.system.warnings.IGNORE`

            **kwargs:
                Optional keyword arguments for the sampling method, specified per solver in
                :attr:`.DWaveSampler.parameters`. D-Wave System Documentation's
                `solver guide <https://docs.dwavesys.com/docs/latest/doc_solver_ref.html>`_
                describes the parameters and properties supported on the D-Wave system.

        Returns:
            :class:`dimod.SampleSet`: A `dimod` :obj:`~dimod.SampleSet` object.
            In it this sampler also provides timing information in the `info`
            field as described in the D-Wave System Documentation's
            `timing guide <https://docs.dwavesys.com/docs/latest/doc_timing.html>`_.

        Examples:
            This example submits a two-variable Ising problem mapped directly to qubits
            0 and 1 on a D-Wave system.

            >>> from dwave.system.samplers import DWaveSampler
            >>> sampler = DWaveSampler()
            >>> sampleset = sampler.sample_ising({0: -1, 1: 1}, {})
            >>> for sample in sampleset.samples():    # doctest: +SKIP
            ...    print(sample)
            ...
            {0: 1, 1: -1}

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_
        for explanations of technical terms in descriptions of Ocean tools.

        """
        nodes = self.solver.nodes  # set rather than .nodelist which is a list

        if isinstance(h, list):
            # to be consistent with the cloud-client, we ignore the 0 biases
            # on missing nodes.
            h = dict((v, b) for v, b in enumerate(h) if b or v in nodes)

        # developer note: in the future we should probably catch exceptions
        # from the cloud client, but for now this is simpler/cleaner. We use
        # the solver's nodes/edges because they are a set, so faster lookup
        edges = self.solver.edges
        if not (all(v in nodes for v in h) and
                all((u, v) in edges or (v, u) in edges for u, v in J)):
            msg = "Problem graph incompatible with solver."
            raise BinaryQuadraticModelStructureError(msg)

        future = self.solver.sample_ising(h, J, **kwargs)

        # do as much as possible after the future is returned

        variables = set(h).union(*J)

        if warnings is None:
            warnings = self.warnings_default
        warninghandler = WarningHandler(warnings)
        warninghandler.energy_scale((h, J))

        hook = _result_to_response_hook(variables, dimod.SPIN, warninghandler)
        return dimod.SampleSet.from_future(future, hook)

    @_failover
    def sample_qubo(self, Q, warnings=None, **kwargs):
        """Sample from the specified QUBO.

        Args:
            Q (dict):
                Coefficients of a quadratic unconstrained binary optimization (QUBO) model.

            warnings (:class:`~dwave.system.warnings.WarningAction`, optional):
                Defines what warning action to take, if any. See
                :mod:`~dwave.system.warnings`. The default behaviour is defined
                by :attr:`warnings_default`, which itself defaults to
                :class:`~dwave.system.warnings.IGNORE`

            **kwargs:
                Optional keyword arguments for the sampling method, specified per solver in
                :attr:`.DWaveSampler.parameters`. D-Wave System Documentation's
                `solver guide <https://docs.dwavesys.com/docs/latest/doc_solver_ref.html>`_
                describes the parameters and properties supported on the D-Wave system.

        Returns:
            :class:`dimod.SampleSet`: A `dimod` :obj:`~dimod.SampleSet` object.
            In it this sampler also provides timing information in the `info`
            field as described in the D-Wave System Documentation's
            `timing guide <https://docs.dwavesys.com/docs/latest/doc_timing.html>`_.

        Examples:
            This example submits a two-variable QUBO mapped directly to qubits
            0 and 4 on a D-Wave system.

            >>> from dwave.system.samplers import DWaveSampler
            >>> sampler = DWaveSampler()
            >>> Q = {(0, 0): -1, (4, 4): -1, (0, 4): 2}
            >>> sampleset = sampler.sample_qubo(Q)
            >>> for sample in sampleset.samples():    # doctest: +SKIP
            ...    print(sample)
            ...
            {0: 0, 4: 1}

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_
        for explanations of technical terms in descriptions of Ocean tools.

        """

        # developer note: in the future we should probably catch exceptions
        # from the cloud client, but for now this is simpler/cleaner. We use
        # the solver's nodes/edges because they are a set, so faster lookup
        nodes = self.solver.nodes
        edges = self.solver.edges
        if not all(u in nodes if u == v else ((u, v) in edges or (v, u) in edges)
                   for u, v in Q):
            msg = "Problem graph incompatible with solver."
            raise BinaryQuadraticModelStructureError(msg)

        future = self.solver.sample_qubo(Q, **kwargs)

        # do as much as possible after the future is returned

        variables = set().union(*Q)

        if warnings is None:
            warnings = self.warnings_default
        warninghandler = WarningHandler(warnings)
        warninghandler.energy_scale((Q,))

        hook = _result_to_response_hook(variables, dimod.BINARY, warninghandler)
        return dimod.SampleSet.from_future(future, hook)

    def validate_anneal_schedule(self, anneal_schedule):
        """Raise an exception if the specified schedule is invalid for the sampler.

        Args:
            anneal_schedule (list):
                An anneal schedule variation is defined by a series of pairs of floating-point
                numbers identifying points in the schedule at which to change slope. The first
                element in the pair is time t in microseconds; the second, normalized persistent
                current s in the range [0,1]. The resulting schedule is the piecewise-linear curve
                that connects the provided points.

        Raises:
            ValueError: If the schedule violates any of the conditions listed below.

            RuntimeError: If the sampler does not accept the `anneal_schedule` parameter or
                if it does not have `annealing_time_range` or `max_anneal_schedule_points`
                properties.

        As described in
        `D-Wave System Documentation <https://docs.dwavesys.com/docs/latest/doc_solver_ref.html>`_,
        an anneal schedule must satisfy the following conditions:

            * Time t must increase for all points in the schedule.
            * For forward annealing, the first point must be (0,0) and the anneal fraction s must
              increase monotonically.
            * For reverse annealing, the anneal fraction s must start and end at s=1.
            * In the final point, anneal fraction s must equal 1 and time t must not exceed the
              maximum  value in the `annealing_time_range` property.
            * The number of points must be >=2.
            * The upper bound is system-dependent; check the `max_anneal_schedule_points` property.
              For reverse annealing, the maximum number of points allowed is one more than the
              number given by this property.

        Examples:
            This example sets a quench schedule on a D-Wave system.

            >>> from dwave.system.samplers import DWaveSampler
            >>> sampler = DWaveSampler()
            >>> quench_schedule=[[0.0, 0.0], [12.0, 0.6], [12.8, 1.0]]
            >>> DWaveSampler().validate_anneal_schedule(quench_schedule)    # doctest: +SKIP
            >>>

        """
        if 'anneal_schedule' not in self.parameters:
            raise RuntimeError("anneal_schedule is not an accepted parameter for this sampler")

        properties = self.properties

        try:
            min_anneal_time, max_anneal_time = properties['annealing_time_range']
            max_anneal_schedule_points = properties['max_anneal_schedule_points']
        except KeyError:
            raise RuntimeError("annealing_time_range and max_anneal_schedule_points are not properties of this solver")

        # The number of points must be >= 2.
        # The upper bound is system-dependent; check the max_anneal_schedule_points property
        if not isinstance(anneal_schedule, list):
            raise TypeError("anneal_schedule should be a list")
        elif len(anneal_schedule) < 2 or len(anneal_schedule) > max_anneal_schedule_points:
            msg = ("anneal_schedule must contain between 2 and {} points (contains {})"
                   ).format(max_anneal_schedule_points, len(anneal_schedule))
            raise ValueError(msg)

        try:
            t_list, s_list = zip(*anneal_schedule)
        except ValueError:
            raise ValueError("anneal_schedule should be a list of 2-tuples")

        # Time t must increase for all points in the schedule.
        if not all(tail_t < lead_t for tail_t, lead_t in zip(t_list, t_list[1:])):
            raise ValueError("Time t must increase for all points in the schedule")

        # max t cannot exceed max_anneal_time
        if t_list[-1] > max_anneal_time:
            raise ValueError("schedule cannot be longer than the maximum anneal time of {}".format(max_anneal_time))

        start_s, end_s = s_list[0], s_list[-1]
        if end_s != 1:
            raise ValueError("In the final point, anneal fraction s must equal 1.")
        if start_s == 1:
            # reverse annealing
            pass
        elif start_s == 0:
            # forward annealing, s must monotonically increase.
            if not all(tail_s <= lead_s for tail_s, lead_s in zip(s_list, s_list[1:])):
                raise ValueError("For forward anneals, anneal fraction s must monotonically increase")
        else:
            msg = ("In the first point, anneal fraction s must equal 0 for forward annealing or "
                   "1 for reverse annealing")
            raise ValueError(msg)

        # finally check the slope abs(slope) < 1/min_anneal_time
        max_slope = 1.0 / min_anneal_time
        for (t0, s0), (t1, s1) in zip(anneal_schedule, anneal_schedule[1:]):
            if abs((s0 - s1) / (t0 - t1)) > max_slope:
                raise ValueError("the maximum slope cannot exceed {}".format(max_slope))


def _result_to_response_hook(variables, vartype, warninghandler=None):

    def _hook(computation):
        result = computation.result()

        # get the samples. The future will return all spins so filter for the ones in variables
        samples = [[sample[v] for v in variables] for sample in result.get('solutions')]

        # construct the info field (add timing, problem id)
        info = {}
        if 'timing' in result:
            info.update(timing=result['timing'])
        if hasattr(computation, 'id'):
            info.update(problem_id=computation.id)

        sampleset = dimod.SampleSet.from_samples((samples, variables), info =info, vartype=vartype,
                                                 energy=result['energies'],
                                                 num_occurrences=result.get('num_occurrences', None),
                                                 sort_labels=True)

        if warninghandler is not None:
            warninghandler.too_few_samples(sampleset)

            if warninghandler.action is WarningAction.SAVE:
                sampleset.info['warnings'] = warninghandler.saved

        return sampleset

    return _hook
