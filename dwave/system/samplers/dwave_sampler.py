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
A :std:doc:`dimod sampler <dimod:reference/samplers>` for the D-Wave system.

See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_
for explanations of technical terms in descriptions of Ocean tools.
"""
from __future__ import division

import dimod

from dwave.cloud import Client

__all__ = ['DWaveSampler']


class DWaveSampler(dimod.Sampler, dimod.Structured):
    """A class for using the D-Wave system as a sampler.

    Inherits from :class:`dimod.Sampler` and :class:`dimod.Structured`.

    Enables quick incorporation of the D-Wave system as a sampler in
    the D-Wave Ocean software stack. Also enables optional customizing of input
    parameters to :std:doc:`D-Wave Cloud Client <cloud-client:index>`
    (the stack's communication-manager package).

    Args:
        config_file (str, optional):
            Path to a
            :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`
            that identifies a D-Wave system and provides connection information.

        profile (str, optional):
            Profile to select from a
            :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`.

        endpoint (str, optional):
            D-Wave API endpoint URL. If specified, used instead of retrieving a value from
            a :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`.

        token (str, optional):
            Authentication token for the D-Wave API to authenticate the client session.
            If specified, used instead of retrieving a value from a
            :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`.

        solver (str, optional):
            Solver (a D-Wave system on which to run submitted problems).
            If specified, used instead of retrieving a value from a
            :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`.

        solver_features (dict, optional):
            Set of features the used solver has to have. Name-based selection via `solver`
            argument overrides the feature-based selection. For available features (and values),
            see: :meth:`~dwave.cloud.client.Client.solvers`.

        proxy (str, optional):
            Proxy URL to be used for accessing the D-Wave API. If specified, used instead of
            retrieving a value from a
            :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`.

    Examples:
        This example creates a :class:`DWaveSampler` based on a fictive user's
        :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`
        and submits a simple Ising problem of just two variables
        that map to qubits 0 and 1 on the example system. (The simplicity of this example
        obviates the need for an embedding composite---the presence of qubits 0 and 1 on
        the selected D-Wave system can be verified manually.)

        >>> # Example configuration file /home/susan/.config/dwave/dwave.conf:
        >>> #    [defaults]
        >>> #    endpoint = https://url.of.some.dwavesystem.com/sapi
        >>> #    client = qpu
        >>> #
        >>> #    [dw2000]
        >>> #    solver = EXAMPLE_2000Q_SYSTEM
        >>> #    token = ABC-123456789123456789123456789
        >>> from dwave.system.samplers import DWaveSampler
        >>> sampler = DWaveSampler()
        >>> response = sampler.sample_ising({0: -1, 1: 1}, {})
        >>> for sample in response.samples():  # doctest: +SKIP
        ...    print(sample)
        ...
        {0: 1, 1: -1}

    See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_
    for explanations of technical terms in descriptions of Ocean tools.

    """
    def __init__(self, config_file=None, profile=None, endpoint=None, token=None,
                 solver=None, solver_features=None, proxy=None, permissive_ssl=False):

        self.client = Client.from_config(config_file=config_file, profile=profile,
                                         endpoint=endpoint, token=token, proxy=proxy,
                                         permissive_ssl=permissive_ssl)

        # TODO: deprecate `solver`` name in favor of name regex in `solver_features`
        self.solver = self.client.get_solver(name=solver, features=solver_features)

        # need to set up the nodelist and edgelist, properties, parameters
        self._nodelist = sorted(self.solver.nodes)
        self._edgelist = sorted(set(tuple(sorted(edge)) for edge in self.solver.edges))
        self._properties = self.solver.properties.copy()  # shallow copy
        self._parameters = {param: ['parameters'] for param in self.solver.properties['parameters']}

    @property
    def properties(self):
        """dict: D-Wave solver properties as returned by a SAPI query.

        Solver properties are dependent on the selected D-Wave solver and subject to change;
        for example, new released features may add properties.

        Examples:
            This example creates a :class:`DWaveSampler` and prints the properties retrieved
            from a D-Wave solver selected by the user's default
            :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`.

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
        return self._properties

    @property
    def parameters(self):
        """dict[str, list]: D-Wave solver parameters in the form of a dict, where keys are
        keyword parameters accepted by a SAPI query and values are lists of properties in
        :attr:`.DWaveSampler.properties` for each key.

        Solver parameters are dependent on the selected D-Wave solver and subject to change;
        for example, new released features may add parameters.

        Examples:
            This example creates a :class:`DWaveSampler` and prints the parameters retrieved
            from a D-Wave solver selected by the user's default
            :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`.

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
        return self._parameters

    @property
    def edgelist(self):
        """list: List of active couplers for the D-Wave solver.

        Examples:
            This example creates a :class:`DWaveSampler` and prints the active couplers retrieved
            from a D-Wave solver selected by the user's default
            :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`.

            >>> from dwave.system.samplers import DWaveSampler
            >>> sampler = DWaveSampler()
            >>> sampler.edgelist    # doctest: +SKIP
            [(0, 4),
             (0, 5),
             (0, 6),
             (0, 7),
             (0, 128),
             (1, 4),
             (1, 5),
             (1, 6),
             (1, 7),
             (1, 129),
             (2, 4),
            # Snipped above response for brevity

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_
        for explanations of technical terms in descriptions of Ocean tools.

        """
        return self._edgelist

    @property
    def nodelist(self):
        """list: List of active qubits for the D-Wave solver.

        Examples:
            This example creates a :class:`DWaveSampler` and prints the active qubits retrieved
            from a D-Wave solver selected by the user's default
            :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`.

            >>> from dwave.system.samplers import DWaveSampler
            >>> sampler = DWaveSampler()
            >>> sampler.nodelist    # doctest: +SKIP
            [0,
             1,
             2,
             3,
             4,
             5,
            # Snipped above response for brevity

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_
        for explanations of technical terms in descriptions of Ocean tools.

        """
        return self._nodelist

    def sample_ising(self, h, J, **kwargs):
        """Sample from the provided Ising model.

        Args:
            h (list/dict):
                Linear biases of the Ising model. If a list, the list's indices are
                used as variable labels.

            J (dict[(int, int): float]):
                Quadratic biases of the Ising model.

            **kwargs:
                Optional keyword arguments for the sampling method, specified per solver in
                :attr:`.DWaveSampler.parameters`

        Returns:
            :class:`dimod.Response`

        Examples:
            This example creates a :class:`DWaveSampler` based on a D-Wave solver selected by the
            user's default
            :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`
            and submits a simple Ising problem of just two variables that map to qubits
            0 and 1 on the example system. (The simplicity of this example obviates
            the need for an embedding composite---the presence of qubits 0 and 1 on
            the selected D-Wave system can be verified manually.)

            >>> from dwave.system.samplers import DWaveSampler
            >>> sampler = DWaveSampler()
            >>> response = sampler.sample_ising({0: -1, 1: 1}, {})
            >>> for sample in response.samples():    # doctest: +SKIP
            ...    print(sample)
            ...
            {0: 1, 1: -1}

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_
        for explanations of technical terms in descriptions of Ocean tools.

        """
        if isinstance(h, list):
            h = dict(enumerate(h))

        variables = set(h).union(*J)
        try:
            active_variables = sorted(variables)
        except TypeError:
            active_variables = list(variables)
        num_variables = len(active_variables)

        future = self.solver.sample_ising(h, J, **kwargs)

        return dimod.Response.from_future(future, _result_to_response_hook(active_variables, dimod.SPIN))

    def sample_qubo(self, Q, **kwargs):
        """Sample from the provided QUBO.

        Args:
            Q (dict):
                Coefficients of a quadratic unconstrained binary optimization (QUBO) model.

            **kwargs:
                Optional keyword arguments for the sampling method, specified per solver in
                :attr:`.DWaveSampler.parameters`

        Returns:
            :class:`dimod.Response`

        Examples:
            This example creates a :class:`DWaveSampler` based on a D-Wave solver selected by the
            user's default
            :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`
            and submits a simple QUBO problem of just two variables that map to coupled
            qubits 0 and 4 on the example system. (The simplicity of this example obviates
            the need for an embedding composite---the presence of qubits 0 and 4, and
            their coupling, on the selected D-Wave system can be verified manually.)

            >>> from dwave.system.samplers import DWaveSampler
            >>> sampler = DWaveSampler()
            >>> Q = {(0, 0): -1, (4, 4): -1, (0, 4): 2}
            >>> response = sampler.sample_qubo(Q)
            >>> for sample in response.samples():    # doctest: +SKIP
            ...    print(sample)
            ...
            {0: 0, 4: 1}

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_
        for explanations of technical terms in descriptions of Ocean tools.

        """
        variables = set().union(*Q)
        try:
            active_variables = sorted(variables)
        except TypeError:
            active_variables = list(variables)
        num_variables = len(active_variables)

        future = self.solver.sample_qubo(Q, **kwargs)

        return dimod.Response.from_future(future, _result_to_response_hook(active_variables, dimod.BINARY))

    def validate_anneal_schedule(self, anneal_schedule):
        """Raise an exception of the given schedule is not valid for the sampler.

        Args:
            anneal_schedule (list):
                An anneal schedule variation is defined by a series of pairs of floating-point
                numbers identifying points in the schedule at which to change slope. The first
                element in the pair is time t in microseconds; the second, normalized persistent
                current s in the range [0,1]. The resulting schedule is the piecewise-linear curve
                that connects the provided points.

        Raises:
            ValueError: If any of the above conditions are not satisfied.

            RuntimeError: If the sampler does not accept the anneal_schedule parameter or
                if it does not have annealing_time_range or max_anneal_schedule_points properties.

        An anneal schedule must satisfy the following conditions:

            * Time t must increase for all points in the schedule.
            * For forward annealing, the first point must be (0,0) and the anneal fraction s must
              increase monotonically.
            * For reverse annealing, the anneal fraction s must start and end at s=1.
            * In the final point, anneal fraction s must equal 1 and time t must not exceed the
              maximum  value in the annealing_time_range property.
            * The number of points must be >=2.
            * The upper bound is system-dependent; check the max_anneal_schedule_points property.
              For reverse annealing, the maximum number of points allowed is one more than the
              number given by this property.

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
            if not all(tail_s < lead_s for tail_s, lead_s in zip(s_list, s_list[1:])):
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


def _result_to_response_hook(variables, vartype):
    def _hook(computation):
        result = computation.result()

        # get the samples. The future will return all spins so filter for the ones in variables
        samples = [[sample[v] for v in variables] for sample in result.get('solutions')]

        # the only two data vectors we're interested in are energies and num_occurrences
        vectors = {'energy': result['energies']}

        if 'num_occurrences' in result:
            vectors['num_occurrences'] = result['num_occurrences']

        # finally put the timing information (if present) into the misc info. We ignore everything
        # else
        if 'timing' in result:
            info = {'timing': result['timing']}
        else:
            info = {}

        return dimod.Response.from_samples(samples, vectors, info, vartype, variable_labels=variables)

    return _hook
