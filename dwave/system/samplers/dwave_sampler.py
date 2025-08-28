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
A :ref:`dimod <index_dimod>` :term:`sampler` for D-Wave quantum computers.
"""

from __future__ import annotations

import copy
import collections.abc as abc
from collections import defaultdict
from typing import Optional, Dict, TYPE_CHECKING

import dimod
import dwave_networkx as dnx
from dimod.exceptions import BinaryQuadraticModelStructureError
from dwave.cloud.client import Client
from dwave.cloud.exceptions import (
    SolverError, SolverAuthenticationError, InvalidAPIResponseError,
    RequestTimeout, PollingTimeout, ProblemUploadError, ProblemStructureError,
    SolverNotFoundError,
)

from dwave.system.exceptions import FailoverCondition, RetryCondition
from dwave.system.warnings import WarningHandler, WarningAction

if TYPE_CHECKING:
    from dwave.cloud.solver import StructuredSolver

__all__ = ['DWaveSampler', 'qpu_graph']


def qpu_graph(topology_type, topology_shape, nodelist, edgelist):
    """Convert node and edge lists to a ``dwave-networkx`` graph.

    Creates a QPU topology (Chimera, Pegasus or Zephyr) graph compatible with
    Ocean software's :ref:`index_dnx`.

    Args:
        topology_type (string):
            Type of lattice. Valid strings are `chimera`, `pegasus` and
            `zephyr`.
        topology_shape(iterable of ints):
            Dimensions of the lattice.
        nodelist (list of ints):
            List of nodes in the graph. Node labeling is integer and compatible
            with the linear labeling scheme for the specified ``topology_type``.
        edgelist (list of Tuples):
            List of edges in the graph, with each edge consisting of a pair of
            nodes.

    See also:
            :func:`dwave_networkx.chimera_graph`,
            :func:`dwave_networkx.pegasus_graph`,
            :func:`dwave_networkx.zephyr_graph` for descriptions of the lattice
            parameters and indexing.
    """

    if topology_type == 'chimera':
        if not (1 <= len(topology_shape) <=3):
            raise ValueError('topology_shape is incompatible with a chimera lattice.')
        G = dnx.chimera_graph(*topology_shape,
                              node_list=nodelist,
                              edge_list=edgelist)
    elif topology_type == 'pegasus':
        if len(topology_shape) != 1:
            raise ValueError('topology_shape is incompatible with a pegasus lattice.')
        G = dnx.pegasus_graph(topology_shape[0],
                                  node_list=nodelist,
                                  edge_list=edgelist)
    elif topology_type == 'zephyr':
        if len(topology_shape) not in (1, 2):
            raise ValueError('topology_shape is incompatible with a zephyr lattice.')
        G = dnx.zephyr_graph(*topology_shape,
                                 node_list=nodelist,
                                 edge_list=edgelist)
    else:
        # Alternative could be to create a standard network graph and
        # issue a warning. Requires new dependency on networkx.
        raise ValueError('topology_type does not match a known QPU architecure')
    return G


def _get_solver_id(solver: StructuredSolver) -> str:
    """Return a unique solver string identifier, derived from solver's name or
    solver's identity (if available)."""

    # only available in cloud-client>=0.14
    if hasattr(solver, 'identity'):
        return str(solver.identity)

    # used until cloud-client==0.14, deprecated since
    return solver.id


class DWaveSampler(dimod.Sampler, dimod.Structured):
    r"""Submits binary quadratic models directly to D-Wave quantum computers.

    Linear and quadratic terms of the :term:`binary quadratic model` (BQM) must
    map directly to qubit and coupler indices of the selected :term:`QPU`.
    Typically this mapping (:term:`minor-embedding`) is handled by software
    (e.g., the :class:`.EmbeddingComposite` class) but for small problems can be
    manual.
    You can configure your :term:`solver` selection and usage by setting
    parameters, hierarchically, in a configuration file, as environment
    variables, or explicitly as input arguments. For more information, see the
    :meth:`~dwave.cloud.client.Client.get_solvers` method. By default, online
    QPUs are returned ordered by highest number of qubits.

    Args:
        failover (bool, optional, default=False):
            Signal a failover condition if a sampling error occurs. When
            ``True``, raises :exc:`~dwave.system.exceptions.FailoverCondition`
            or :exc:`~dwave.system.exceptions.RetryCondition` on sampleset
            :meth:`~dimod.SampleSet.resolve` to signal failover.
            Actual failover (i.e., selection of a new solver) has to be handled
            by the user. A convenience method :meth:`.trigger_failover` is
            available for this. Note that hardware graphs vary between QPUs, so
            triggering failover results in regenerated :attr:`.nodelist`,
            :attr:`.edgelist`, :attr:`.properties` and :attr:`.parameters`.

            .. versionchanged:: 1.16.0

               In the past, the :meth:`.sample` method was blocking and
               ``failover=True`` caused a solver failover and sampling retry.
               However, this failover implementation broke when :meth:`.sample`
               became non-blocking (asynchronous), Setting ``failover=True`` had
               no effect.

        retry_interval (number, optional, default=-1):
            Ignored, but kept for backward compatibility.

            .. versionchanged:: 1.16.0

               Ignored since 1.16.0. See note for ``failover`` parameter above.

        **config:
            Keyword arguments passed to
            :meth:`~dwave.cloud.client.Client.from_config`.

    .. versionadded:: 1.29.0
        Support for context manager protocol.

    Note:
        Prior to version 1.0.0, :class:`.DWaveSampler` used the ``base`` client,
        allowing non-QPU solvers to be selected.
        To reproduce the old behavior, instantiate :class:`.DWaveSampler` with
        ``client='base'``.

    Note:
        The recommended way to use :class:`DWaveSampler` is from a
        `runtime context <https://docs.python.org/3/reference/datamodel.html#with-statement-context-managers>`_:

        >>> with DWaveSampler() as sampler:
        ...     sampler.sample_ising(...)       # doctest: +SKIP

        Alternatively, call the :meth:`~DWaveSampler.close` method to
        terminate the sampler resources:

        >>> sampler = DWaveSampler()
        ...
        >>> sampler.close()

    Examples:
        This example submits a two-variable Ising problem mapped directly to two
        adjacent qubits on a :term:`QPU`. ``qubit_a`` is the first qubit in
        the QPU's indexed list of qubits and ``qubit_b`` is one of the qubits
        coupled to it. Other required parameters for communication with the
        system, such as its URL and an authentication token, are implicitly set
        in a configuration file or as environment variables, as described in the
        :ref:`ocean_sapi_access_basic` section. Given sufficient reads (here
        100), the quantum computer should return the best solution,
        :math:`{1, -1}` on ``qubit_a`` and ``qubit_b``, respectively, as its
        first sample (samples are ordered from lowest energy).

        >>> from dwave.system import DWaveSampler
        ...
        >>> with DWaveSampler() as sampler:
        ...     qubit_a = sampler.nodelist[0]
        ...     qubit_b = next(iter(sampler.adjacency[qubit_a]))
        ...     sampleset = sampler.sample_ising({qubit_a: -1, qubit_b: 1},
        ...                                      {},
        ...                                      num_reads=100)
        ...     print(sampleset.first.sample[qubit_a] == 1 and sampleset.first.sample[qubit_b] == -1)
        True


        For additional examples, see:

        *   :ref:`Beginner examples <qpu_index_examples_beginner>` of using
            :class:`.DWaveSampler`.
        *   :ref:`qpu_basic_config`.
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

        self.failover = failover
        self.retry_interval = retry_interval
        self._solver_penalty = defaultdict(int)

        self.client = Client.from_config(**config)
        self.solver = self._get_solver(penalty=self._solver_penalty)

    def close(self):
        """Close the underlying cloud client to release system resources such as
        threads.

        The method blocks for all the currently scheduled work (sampling
        requests) to finish.

        See also:
            :meth:`~dwave.cloud.client.Client.close`.
        """
        self.client.close()

    def _get_solver(self, *, refresh: bool = False, penalty: Optional[Dict[str, int]] = None):
        """Get the least penalized solver from the list of solvers filtered and
        ordered according to user config.

        Note: we need to partially replicate
        :class:`dwave.cloud.Client.get_solver` logic.
        """
        if penalty is None:
            penalty = {}

        # the only solver filters used by `DWaveSampler` are those
        # propagated to `Client.from_config` on construction
        filters = copy.deepcopy(self.client.config.solver)
        order_by = filters.pop('order_by', 'avg_load')
        solvers = self.client.get_solvers(refresh=refresh, order_by=order_by, **filters)

        # we now just need to de-prioritize penalized solvers
        solvers.sort(key=lambda solver: penalty.get(_get_solver_id(solver), 0))

        try:
            return solvers[0]
        except IndexError:
            raise SolverNotFoundError("Solver with the requested features not available") from None

    warnings_default = WarningAction.IGNORE
    """Defines default behavior for ``warnings`` keyword arguments of the
    :meth:`~DWaveSampler.sample_ising`  and :meth:`~DWaveSampler.sample_qubo`
    methods.
    """

    @property
    def properties(self):
        """dict: Solver properties as returned by a :term:`SAPI` query.

        Solver properties are dependent on the selected solver and subject to
        change; for example, new features may add properties. The
        :ref:`qpu_index_solver_properties` and :ref:`qpu_solver_parameters`
        sections describe the parameters and properties supported on D-Wave
        quantum computers.

        Examples:

            >>> from dwave.system import DWaveSampler
            >>> with DWaveSampler() as sampler:
            ...     print(sampler.properties['category'])
            qpu

        """
        try:
            return self._properties
        except AttributeError:
            self._properties = properties = self.solver.properties.copy()
            return properties

    @property
    def parameters(self):
        """dict[str, list]: Solver parameters as returned by a :term:`SAPI`
        query.

        Keys of the returned dict are keyword parameters accepted by a SAPI
        query and values are lists of properties in
        :attr:`~DWaveSampler.properties` for each key.

        Solver parameters are dependent on the selected solver and subject to
        change; for example, new features may add parameters. The
        :ref:`qpu_index_solver_properties` and :ref:`qpu_solver_parameters`
        sections describe the parameters and properties supported on D-Wave
        quantum computers.

        Examples:

            >>> from dwave.system import DWaveSampler
            >>> with DWaveSampler() as sampler:
            ...     'auto_scale' in sampler.parameters
            True

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
        """list: List of active couplers for the solver.

        Active couplers are those that are included in the
        :ref:`working graph <topologies_working_graph>`.

        Examples:
            First coupler for a selected Advantage2 system.

            >>> from dwave.system import DWaveSampler
            >>> with DWaveSampler(topology_type='zephyr') as sampler:
            ...     sampler.edgelist[0]
            (0, 1)

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
        """list: List of active qubits for the solver.

        Active qubits are those that are included in the
        :ref:`working graph <topologies_working_graph>`.

        Examples:
            First three qubits for a selected Advantage2 system.

            >>> from dwave.system import DWaveSampler
            >>> with DWaveSampler(topology_type='zephyr') as sampler:
            ...     sampler.nodelist[:3]
            [0, 1, 2]

        """
        # Assumption: cloud client nodes are always integer-labelled
        try:
            nodelist = self._nodelist
        except AttributeError:
            self._nodelist = nodelist = sorted(self.solver.nodes)
        return nodelist

    def trigger_failover(self):
        """Trigger a failover and connect to a new solver."""

        # penalize the solver that just failed
        self._solver_penalty[_get_solver_id(self.solver)] += 1

        # select the next solver in user-defined preference order, but try to
        # avoid the penalized (failed) ones
        self.solver = self._get_solver(refresh=True, penalty=self._solver_penalty)

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

    def sample(self, bqm, warnings=None, **kwargs):
        """Sample from the specified binary quadratic model.

        Args:
            bqm (:class:`~dimod.BinaryQuadraticModel`):
                The :term:`binary quadratic model`. Must match
                :attr:`~DWaveSampler.nodelist` and :attr:`~DWaveSampler.edgelist`.

            warnings (:class:`~dwave.system.warnings.WarningAction`, optional):
                Defines what warning action to take, if any (see the
                :ref:`system_warnings` section). The default behavior is to
                ignore warnings.

            **kwargs:
                Optional keyword arguments for the sampling method, specified
                per solver in :attr:`.parameters`. The
                :ref:`qpu_index_solver_properties` and
                :ref:`qpu_solver_parameters` sections describe the parameters
                and properties supported on D-Wave quantum computers.

        Returns:
            :class:`~dimod.SampleSet`: Sample set constructed from a
            (non-blocking) :class:`~concurrent.futures.Future`-like object.
            In it this sampler also provides timing information in the ``info``
            field as described in the :ref:`qpu_sapi_qpu_timing` section.

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
            >>> with DWaveSampler() as sampler:
            ...     qubit_a = sampler.nodelist[0]
            ...     qubit_b = next(iter(sampler.adjacency[qubit_a]))
            ...     sampleset = sampler.sample_ising({qubit_a: -1, qubit_b: 1},
            ...                                      {},
            ...                                      num_reads=100)
            ...     print(sampleset.first.sample[qubit_a] == 1 and sampleset.first.sample[qubit_b] == -1)
            True

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

        # need a hook so that we can lazily check the sampleset for warnings
        # and handle failover consistently
        def _hook(computation):
            def resolve(computation):
                sampleset = computation.sampleset
                sampleset.resolve()

                if warninghandler is not None:
                    warninghandler.too_few_samples(sampleset)
                    if warninghandler.action is WarningAction.SAVE:
                        sampleset.info['warnings'] = warninghandler.saved

                return sampleset

            try:
                return resolve(computation)

            except (ProblemUploadError, RequestTimeout, PollingTimeout) as exc:
                if not self.failover:
                    raise exc

                # failover with retry on:
                # - request or polling timeout
                # - upload errors
                raise RetryCondition("resubmit problem") from exc

            except (SolverError, InvalidAPIResponseError) as exc:
                if not self.failover:
                    raise exc
                if isinstance(exc, SolverAuthenticationError):
                    raise exc

                # failover on:
                # - solver offline, solver disabled or not found
                # - internal SAPI errors (like malformed response)
                # - generic solver errors
                # but NOT on auth errors
                raise FailoverCondition("switch solver and resubmit problem") from exc

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
        """Raise an exception if the specified schedule is invalid for the
        sampler.

        Args:
            anneal_schedule (list):
                An anneal schedule is defined by a series of pairs of
                floating-point numbers identifying points in the schedule at
                which to change slope. The first element in the pair is time,
                :math:`t` in microseconds; the second, normalized anneal
                fraction (persistent current) :math:`s` in the range [0,1]. The
                resulting schedule is the piecewise-linear curve that connects
                the provided points.

                An anneal schedule must satisfy the conditions described in the
                :ref:`parameter_qpu_anneal_schedule` section.

        Raises:
            ValueError: If the schedule violates any of the conditions
                described in the :ref:`parameter_qpu_anneal_schedule` section.

            RuntimeError: If the sampler does not accept the ``anneal_schedule``
                parameter or if it does not have
                :ref:`property_qpu_annealing_time_range` or
                :ref:`property_qpu_max_anneal_schedule_points` properties.

        Examples:
            This example sets a quench schedule on a D-Wave quantum computer.

            >>> from dwave.system import DWaveSampler
            >>> with DWaveSampler() as sampler:     # doctest: +SKIP
            ...     quench_schedule=[[0.0, 0.0], [12.0, 0.6], [12.8, 1.0]]
            ...     DWaveSampler().validate_anneal_schedule(quench_schedule)

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
        """Output the QPU's :term:`working graph` in NetworkX format.

        Returns:
            :class:`networkx.Graph`:
                Either a :ref:`Chimera lattice <topology_intro_chimera>`, a
                :ref:`Pegasus lattice <topology_intro_pegasus>` or a
                :ref:`Zephyr lattice <topology_intro_zephyr>`.

        Examples:
            This example converts a selected :term:`QPU` to a graph and verifies
            that it has a greater number of edges (couplers) than nodes
            (qubits).

            >>> from dwave.system import DWaveSampler
            ...
            >>> with DWaveSampler() as sampler:
            ...     g = sampler.to_networkx_graph()
            ...     len(g.edges) > len(g.nodes)
            True
        """
        return qpu_graph(self.properties['topology']['type'],
                         self.properties['topology']['shape'],
                         self.nodelist, self.edgelist)
