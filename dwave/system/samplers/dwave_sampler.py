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

import collections.abc as abc
import functools
import numpy as np
import time

from typing import List, Optional, Tuple
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
        """Converts DWaveSampler's structure to a Chimera, Pegasus or Zephyr NetworkX graph.

        Returns:
            :class:`networkx.Graph`:
                Either an (m, n, t) Chimera lattice, a Pegasus lattice of size m or a
                Zephyr lattice of size m.

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

        elif topology_type == 'zephyr':
            G = dnx.zephyr_graph(shape[0],
                                 node_list=self.nodelist,
                                 edge_list=self.edgelist)

        return G

    def estimate_qpu_access_time(self,
                                 num_qubits: int,
                                 num_reads: int = 1,
                                 annealing_time: Optional[float] = None,
                                 anneal_schedule: Optional[List[Tuple[float, float]]] = None,
                                 initial_state:  Optional[List[Tuple[float, float]]] = None,
                                 reverse_anneal: bool = False,
                                 reinitialize_state: bool = False,
                                 programming_thermalization: Optional[float] = None,
                                 readout_thermalization: Optional[float] = None,
                                 reduce_intersample_correlation: bool = False,
                                 **kwargs) -> float:
        """Estimates QPU access time for a submission to the selected solver.

        Estimates a problem’s quantum processing unit (QPU) access time from the
        parameter values you specify, timing data provided in the ``problem_timing_data``
        `solver property <https://docs.dwavesys.com/docs/latest/c_solver_properties.html>`_,
        and the number of qubits used to embed the problem on the selected QPU, as
        described in the
        `system documentation <https://docs.dwavesys.com/docs/latest/c_qpu_timing.html>`_.

        Requires that you provide the number of qubits to be used for your
        problem submission. :term:`Embedding` is typically heuristic and the number
        of required qubits can vary between executions. If you are using a heuristic
        embedding tool such as
        :std:doc:`minorminer <oceandocs:docs_minorminer/source/sdk_index>` indirectly
        through your sampler (e.g., by using the :class:`~dwave.system.composites.EmbeddingComposite`
        or :class:`~DWaveCliqueSampler` classes), you can use the same tool on
        your problem to estimate the expected number of qubits: for large, complex
        problems you might run the tool several times and take the number of qubits
        from the produced average or worst-case embedding; for small, simple
        problems even a single run might be sufficient. If you are using such a
        tool directly (e.g., in conjunction with the
        :class:`~dwave.system.composites.FixedEmbeddingComposite`)
        or otherwise generating a heuristic or non-heuristic embedding, take
        the required number of qubits from your embedding. Because embedding
        depends on a QPU’s working graph, such embeddings should be for
        the particular QPU for which you are estimating the access time.
        See examples below.

        Args:
            num_qubits:
                Number of qubits required to represent your binary quadratic model
                on the selected solver.
            num_reads:
                Number of reads. Provide this value if you explictly set ``num_reads``
                in your submission.
            annealing_time:
                Annealing duration. Provide this value of if you set
                ``annealing_time`` in your submission.
            anneal_schedule:
                Anneal schedule. Provide the ``anneal_schedule`` if you set it in
                your submission.
            initial_state:
                Initial state. Provide the ``initial_state`` if your submission
                uses reverse annealing.
            reinitialize_state:
                Set to ``True`` if your submission sets ``reinitialize_state``.
            programming_thermalization:
                programming thermalization time. Provide this value if you explictly
                set a value for ``programming_thermalization`` in your submission.
            readout_thermalization:
                Set to ``True`` if your submission sets ``readout_thermalization``.
            reduce_intersample_correlation:
                Set to ``True`` if your submission sets ``reduce_intersample_correlation``.

        Returns:
            Estimated QPU access time, in microseconds.

        Raises:
            KeyError: If a solver property, or a field in the ``problem_timing_data``
                solver property, required by the timing model is missing for the
                selected solver.

            ValueError: If conflicting parameters are set or the selected solver
                uses an unsupported timing model.

        Examples:

            This example compares the estimation with the QPU access time from a
            run on a particular quantum computer for a ferromagnetic problem using
            all the QPU qubits.

            >>> from dwave.system import DWaveSampler
            >>> import dimod
            ...
            >>> qpu = DWaveSampler()
            >>> bqm = dimod.BinaryQuadraticModel({}, {edge: -1 for edge in qpu.edgelist}, "BINARY")
            >>> num_qubits = len(qpu.nodelist)
            >>> estimated_runtime = qpu.estimate_qpu_access_time(num_qubits, num_reads=100)    # doctest: +SKIP
            >>> print("%.0f" % estimated_runtime) # doctest: +SKIP
            42657
            >>> sampleset = qpu.sample(bqm, num_reads=100)
            >>> print(sampleset.info["timing"]["qpu_access_time"])     # doctest: +SKIP
            42643.96

            This example estimates the QPU access time for a random problem
            with a 50-node complete graph using an anneal schedule that sets a
            ~1 ms pause on a D-Wave system.

            >>> from dwave.system import DWaveSampler, FixedEmbeddingComposite
            >>> from minorminer import find_embedding
            >>> import dimod
            >>> import random
            >>> import networkx as nx
            ...
            >>> G = nx.generators.complete_graph(100)
            >>> bqm = dimod.from_networkx_graph(G, "BINARY")
            >>> bqm.add_linear_from({node: random.choices([-0.5, 0.5])[0] for node in bqm.linear.keys()})
            >>> bqm.add_quadratic_from({edge: random.choices([-1, 1])[0] for edge in bqm.quadratic.keys()})
            ...
            >>> qpu = DWaveSampler()
            >>> embedding = find_embedding(bqm.quadratic.keys(), qpu.edgelist)  # doctest: +SKIP
            >>> num_qubits = sum(len(chain) for chain in embedding.values())    # doctest: +SKIP
            ...
            >>> reads = 500
            >>> anneal_schedule = [[0.0, 0.0], [40.0, 0.4], [1040.0, 0.4], [1042, 1.0]]
            >>> estimated_runtime = qpu.estimate_qpu_access_time(num_qubits,
            ...                          num_reads=reads,
            ...                          anneal_schedule=anneal_schedule)    # doctest: +SKIP
            >>> print("%.1f" % estimated_runtime)  # doctest: +SKIP
            83952.1

        """
        if anneal_schedule and annealing_time:
            raise ValueError("set only one of ``anneal_schedule`` or ``annealing_time``")

        if anneal_schedule and anneal_schedule[0][1] == 1 and not initial_state:
            raise ValueError("reverse annealing requested (``anneal_schedule`` "
                             "starts with s==1) but ``initial_state`` not provided")

        try:
            problem_timing_data = self.properties['problem_timing_data']

            version_timing_model = problem_timing_data['version']
            typical_programming_time = problem_timing_data['typical_programming_time']
            ra_with_reinit_prog_time_delta = problem_timing_data['reverse_annealing_with_reinit_prog_time_delta']
            ra_without_reinit_prog_time_delta = problem_timing_data['reverse_annealing_without_reinit_prog_time_delta']
            default_programming_thermalization = problem_timing_data['default_programming_thermalization']
            default_annealing_time = problem_timing_data['default_annealing_time']
            readout_time_model = problem_timing_data['readout_time_model']
            readout_time_model_parameters = problem_timing_data['readout_time_model_parameters']
            qpu_delay_time_per_sample = problem_timing_data['qpu_delay_time_per_sample']
            ra_with_reinit_delay_time_delta = problem_timing_data['reverse_annealing_with_reinit_delay_time_delta']
            ra_without_reinit_delay_time_delta = problem_timing_data['reverse_annealing_without_reinit_delay_time_delta']
            decorrelation_max_nominal_anneal_time = problem_timing_data['decorrelation_max_nominal_anneal_time']
            decorrelation_time_range = problem_timing_data['decorrelation_time_range']
            default_readout_thermalization = problem_timing_data['default_readout_thermalization']

        except KeyError as err:
            if err == 'problem_timing_data':
                err_msg = f"selected solver is missing required property {err}"
            elif err == 'version':
                err_msg = f"selected solver is missing ``problem_timing_data`` field {err}"
            else:
                err_msg = f"selected solver is missing ``problem_timing_data`` field " + \
                          f"{err} required by timing model version {version_timing_model}"
            raise ValueError(err_msg)

        # Support for sapi timing model versions: 1.0.x
        version_tuple = tuple(int(i) for i in version_timing_model.split("."))
        if not version_tuple[0] == 1 and not version_tuple[1] == 0:
            raise ValueError(f"``estimate_qpu_access_time`` does not currently "
                             f"support timing model {version_timing_model} "
                             "used by the selected solver")

        ra_programming_time = 0
        ra_delay_time = 0
        if anneal_schedule and anneal_schedule[0][1] == 1:
            if reinitialize_state:
                ra_programming_time = ra_with_reinit_prog_time_delta
                ra_delay_time = ra_with_reinit_delay_time_delta
            else:
                ra_programming_time = ra_without_reinit_prog_time_delta
                ra_delay_time = ra_without_reinit_delay_time_delta

        if programming_thermalization:
            programming_thermalization_time = programming_thermalization
        else:
            programming_thermalization_time = default_programming_thermalization

        programming_time = typical_programming_time + ra_programming_time + \
                           programming_thermalization_time

        anneal_time = default_annealing_time
        if annealing_time:
            anneal_time = annealing_time
            print(anneal_time)
        elif anneal_schedule:
            anneal_time = anneal_schedule[-1][0]

        n = len(readout_time_model_parameters)
        if n % 2:
             raise ValueError(f"for the selected solver, ``problem_timing_data`` "
                              f"property field ``readout_time_model_parameters`` "
                              f"is not of an even length as required by timing "
                              f"model version {version_timing_model}")

        q = readout_time_model_parameters[:int(n/2)]
        t = readout_time_model_parameters[int(n/2):]
        if readout_time_model == 'pwl_log_log':
            readout_time = pow(10, np.interp(np.emath.log10(num_qubits), q, t))
        elif readout_time_model == 'pwl_linear':
            readout_time = np.interp(num_qubits, q, t)
        else:
            raise ValueError("``estimate_qpu_access_time`` does not support "
                             f"``readout_time_model`` value {readout_time_model} "
                             f"in version {version_timing_model}")

        if readout_thermalization:
            readout_thermalization_time = readout_thermalization
        else:
            readout_thermalization_time = default_readout_thermalization

        decorrelation_time = 0
        if reduce_intersample_correlation:
            r_min = decorrelation_time_range[0]
            r_max = decorrelation_time_range[1]
            decorrelation_time = anneal_time/decorrelation_max_nominal_anneal_time * (r_max - r_min) + r_min

        sampling_time_per_read = anneal_time + readout_time + qpu_delay_time_per_sample + \
            ra_delay_time + max(readout_thermalization_time, decorrelation_time)

        sampling_time = num_reads * sampling_time_per_read

        return sampling_time + programming_time
