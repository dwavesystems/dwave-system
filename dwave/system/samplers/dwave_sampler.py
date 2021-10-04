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

"""
A :std:doc:`dimod sampler <oceandocs:docs_dimod/reference/samplers>` for the D-Wave system.

See :std:doc:`Ocean Glossary <oceandocs:glossary>`
for explanations of technical terms in descriptions of Ocean tools.
"""

import time
import functools
import collections.abc as abc

from warnings import warn

import dimod

from dimod.exceptions import BinaryQuadraticModelStructureError
from dwave.cloud import Client
from dwave.cloud.exceptions import (
    SolverOfflineError, SolverNotFoundError, ProblemStructureError)

from dwave.system.warnings import WarningHandler, WarningAction

import dwave_networkx as dnx

__all__ = ['DWaveSampler']


def _failover(f):
    """Decorator for methods that might raise SolverOfflineError. Assumes that
    the method is on a class with a `trigger_failover` method and a truthy
    `failover` attribute.
    """
    @functools.wraps(f)
    def wrapper(sampler, *args, **kwargs):
        while True:
            try:
                return f(sampler, *args, **kwargs)
            except SolverOfflineError as err:
                if not sampler.failover:
                    raise err

            try:
                sampler.trigger_failover()
            except SolverNotFoundError as err:
                if sampler.retry_interval < 0:
                    raise err

                time.sleep(sampler.retry_interval)

    return wrapper


class DWaveSampler(dimod.Sampler, dimod.Structured):
    """A class for using the D-Wave system as a sampler for binary quadratic models.

    You can configure your :term:`solver` selection and usage by setting parameters,
    hierarchically, in a configuration file, as environment variables, or
    explicitly as input arguments. For more information, see
    `D-Wave Cloud Client <https://docs.ocean.dwavesys.com/en/stable/docs_cloud/sdk_index.html>`_
    :meth:`~dwave.cloud.client.Client.get_solvers`. By default, online
    D-Wave systems are returned ordered by highest number of qubits.

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

        **config:
            Keyword arguments passed to :meth:`dwave.cloud.client.Client.from_config`.

    Note:
        Prior to version 1.0.0, :class:`.DWaveSampler` used the ``base`` client,
        allowing non-QPU solvers to be selected.
        To reproduce the old behavior, instantiate :class:`.DWaveSampler` with
        ``client='base'``.

    Examples:
        This example submits a two-variable Ising problem mapped directly to two
        adjacent qubits on a D-Wave system. ``qubit_a`` is the first qubit in
        the QPU's indexed list of qubits and ``qubit_b`` is one of the qubits
        coupled to it. Other required parameters for communication with the system, such
        as its URL and an autentication token, are implicitly set in a configuration file
        or as environment variables, as described in
        `Configuring Access to D-Wave Solvers <https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html>`_.
        Given sufficient reads (here 100), the quantum
        computer should return the best solution, :math:`{1, -1}` on ``qubit_a`` and
        ``qubit_b``, respectively, as its first sample (samples are ordered from
        lowest energy).

        >>> from dwave.system import DWaveSampler
        ...
        >>> sampler = DWaveSampler()
        ...
        >>> qubit_a = sampler.nodelist[0]
        >>> qubit_b = next(iter(sampler.adjacency[qubit_a]))
        >>> sampleset = sampler.sample_ising({qubit_a: -1, qubit_b: 1},
        ...                                  {},
        ...                                  num_reads=100)
        >>> sampleset.first.sample[qubit_a] == 1 and sampleset.first.sample[qubit_b] == -1
        True

    See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/stable/concepts/index.html>`_
    for explanations of technical terms in descriptions of Ocean tools.

    """
    def __init__(self, failover=False, retry_interval=-1, **config):
        # strongly prefer QPU solvers; requires kwarg-level override
        config.setdefault('client', 'qpu')

        # weakly prefer QPU solver with the highest qubit count,
        # easily overridden on any config level above defaults (file/env/kwarg)
        defaults = config.setdefault('defaults', {})
        if not isinstance(defaults, abc.Mapping):
            raise TypeError("mapping expected for 'defaults'")
        defaults.update(solver=dict(order_by='-num_active_qubits'))

        self.client = Client.from_config(**config)
        self.solver = self.client.get_solver()

        self.failover = failover
        self.retry_interval = retry_interval

    warnings_default = WarningAction.IGNORE
    """Defines the default behavior for :meth:`.sample_ising`'s  and
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

            >>> from dwave.system import DWaveSampler
            >>> sampler = DWaveSampler()
            >>> sampler.properties    # doctest: +SKIP
            {'anneal_offset_ranges': [[-0.2197463755538704, 0.03821687759418928],
              [-0.2242514597680286, 0.01718456460967399],
              [-0.20860153999435985, 0.05511969218508182],
            # Snipped above response for brevity

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/stable/concepts/index.html>`_
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
        :attr:`.properties` for each key.

        Solver parameters are dependent on the selected D-Wave solver and subject to change;
        for example, new released features may add parameters.
        `D-Wave System Documentation <https://docs.dwavesys.com/docs/latest/doc_solver_ref.html>`_
        describes the parameters and properties supported on the D-Wave system.

        Examples:

            >>> from dwave.system import DWaveSampler
            >>> sampler = DWaveSampler()
            >>> sampler.parameters    # doctest: +SKIP
            {'anneal_offsets': ['parameters'],
             'anneal_schedule': ['parameters'],
             'annealing_time': ['parameters'],
             'answer_mode': ['parameters'],
             'auto_scale': ['parameters'],
             # Snipped above response for brevity

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/stable/concepts/index.html>`_
        for explanations of technical terms in descriptions of Ocean tools.

        """
        try:
            return self._parameters
        except AttributeError:
            parameters = {param: ['parameters']
                          for param in self.properties['parameters']}
            parameters.update(warnings=[])
            parameters.update(label=[])
            self._parameters = parameters
            return parameters

    @property
    def edgelist(self):
        """list: List of active couplers for the D-Wave solver.

        Examples:
            Coupler list for one D-Wave 2000Q system (output snipped for brevity).

            >>> from dwave.system import DWaveSampler
            >>> sampler = DWaveSampler()
            >>> sampler.edgelist
            [(0, 4), (0, 5), (0, 6), (0, 7), ...

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/stable/concepts/index.html>`_
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
            Node list for one D-Wave 2000Q system (output snipped for brevity).

            >>> from dwave.system import DWaveSampler
            >>> sampler = DWaveSampler()
            >>> sampler.nodelist
            [0, 1, 2, ...

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/stable/concepts/index.html>`_
        for explanations of technical terms in descriptions of Ocean tools.

        """
        # Assumption: cloud client nodes are always integer-labelled
        try:
            nodelist = self._nodelist
        except AttributeError:
            self._nodelist = nodelist = sorted(self.solver.nodes)
        return nodelist

    def trigger_failover(self):
        """Trigger a failover and connect to a new solver."""

        # the requested features are saved on the client object, so
        # we just need to request a new solver
        self.solver = self.client.get_solver()

        # delete the lazily-constructed attributes
        try:
            del self._edgelist
        except AttributeError:
            pass

        try:
            del self._nodelist
        except AttributeError:
            pass

        try:
            del self._parameters
        except AttributeError:
            pass

        try:
            del self._properties
        except AttributeError:
            pass

    @_failover
    def sample(self, bqm, warnings=None, **kwargs):
        """Sample from the specified binary quadratic model.

        Args:
            bqm (:class:`~dimod.BinaryQuadraticModel`):
                The binary quadratic model. Must match :attr:`.nodelist` and
                :attr:`.edgelist`.

            warnings (:class:`~dwave.system.warnings.WarningAction`, optional):
                Defines what warning action to take, if any. See
                :ref:`warnings_system`. The default behaviour is to
                ignore warnings.

            **kwargs:
                Optional keyword arguments for the sampling method, specified per solver in
                :attr:`.parameters`. D-Wave System Documentation's
                `solver guide <https://docs.dwavesys.com/docs/latest/doc_solver_ref.html>`_
                describes the parameters and properties supported on the D-Wave system.

        Returns:
            :class:`~dimod.SampleSet`: Sample set constructed from a (non-blocking)
            :class:`~concurrent.futures.Future`-like object.
            In it this sampler also provides timing information in the `info`
            field as described in the D-Wave System Documentation's
            :ref:`sysdocs_gettingstarted:qpu_sapi_qpu_timing`.

        Examples:
            This example submits a two-variable Ising problem mapped directly to two
            adjacent qubits on a D-Wave system. ``qubit_a`` is the first qubit in
            the QPU's indexed list of qubits and ``qubit_b`` is one of the qubits
            coupled to it. Given sufficient reads (here 100), the quantum
            computer should return the best solution, :math:`{1, -1}` on ``qubit_a`` and
            ``qubit_b``, respectively, as its first sample (samples are ordered from
            lowest energy).

            >>> from dwave.system import DWaveSampler
            ...
            >>> sampler = DWaveSampler()
            ...
            >>> qubit_a = sampler.nodelist[0]
            >>> qubit_b = next(iter(sampler.adjacency[qubit_a]))
            >>> sampleset = sampler.sample_ising({qubit_a: -1, qubit_b: 1},
            ...                                  {},
            ...                                  num_reads=100)
            >>> sampleset.first.sample[qubit_a] == 1 and sampleset.first.sample[qubit_b] == -1
            True

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/stable/concepts/index.html>`_
        for explanations of technical terms in descriptions of Ocean tools.

        """

        solver = self.solver

        try:
            future = solver.sample_bqm(bqm, **kwargs)
        except ProblemStructureError as exc:
            msg = ("Problem graph incompatible with solver. Please use 'EmbeddingComposite' "
                   "to map the problem graph to the solver.")
            raise BinaryQuadraticModelStructureError(msg) from exc

        if warnings is None:
            warnings = self.warnings_default
        warninghandler = WarningHandler(warnings)
        warninghandler.energy_scale(bqm)

        # need a hook so that we can check the sampleset (lazily) for
        # warnings
        def _hook(computation):
            sampleset = computation.sampleset

            if warninghandler is not None:
                warninghandler.too_few_samples(sampleset)
                if warninghandler.action is WarningAction.SAVE:
                    sampleset.info['warnings'] = warninghandler.saved

            return sampleset

        return dimod.SampleSet.from_future(future, _hook)

    def sample_ising(self, h, *args, **kwargs):
        # to be consistent with the cloud-client, we ignore the 0 biases
        # on missing nodes for lists
        if isinstance(h, list):
            if len(h) > self.solver.num_qubits:
                msg = ("Problem graph incompatible with solver. Please use 'EmbeddingComposite' "
                       "to map the problem graph to the solver.")
                raise BinaryQuadraticModelStructureError(msg)
            nodes = self.solver.nodes
            h = dict((v, b) for v, b in enumerate(h) if b and v in nodes)

        return super().sample_ising(h, *args, **kwargs)

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

            >>> from dwave.system import DWaveSampler
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
            if round(abs((s0 - s1) / (t0 - t1)),10) > max_slope:
                raise ValueError("the maximum slope cannot exceed {}".format(max_slope))


    def to_networkx_graph(self):
        """Converts DWaveSampler's structure to a Chimera or Pegasus NetworkX graph.

        Returns:
            :class:`networkx.Graph`:
                Either an (m, n, t) Chimera lattice or a Pegasus lattice of size m.

        Examples:
            This example converts a selected D-Wave system solver to a graph
            and verifies it has over 2000 nodes.

            >>> from dwave.system import DWaveSampler
            ...
            >>> sampler = DWaveSampler()
            >>> g = sampler.to_networkx_graph()      # doctest: +SKIP
            >>> len(g.nodes) > 2000                  # doctest: +SKIP
            True
        """

        topology_type = self.properties['topology']['type']
        shape = self.properties['topology']['shape']

        if topology_type == 'chimera':
            G = dnx.chimera_graph(*shape,
                                  node_list=self.nodelist,
                                  edge_list=self.edgelist)

        elif topology_type == 'pegasus':
            G = dnx.pegasus_graph(shape[0],
                                  node_list=self.nodelist,
                                  edge_list=self.edgelist)

        return G
