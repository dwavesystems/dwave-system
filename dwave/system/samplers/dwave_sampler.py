"""
A dimod sampler_ for the D-Wave system.

.. _sampler: http://dimod.readthedocs.io/en/latest/reference/samplers.html#samplers-and-composites
"""
import dimod
import dwave.cloud.qpu as qpuclient

__all__ = ['DWaveSampler']


class DWaveSampler(dimod.Sampler, dimod.Structured):
    """A class for using the D-Wave system as a sampler.

    Inherits from :class:`dimod.Sampler` and :class:`dimod.Structured`.

    Enables quick incorporation of the D-Wave system as a sampler in
    the D-Wave Ocean software stack. Also enables optional customizing of input
    parameters to `D-Wave Cloud Client <http://dwave-cloud-client.readthedocs.io/en/latest/>`_
    (the stack's communication-manager package).

    Args:
        config_file (str, optional):
            Path to a D-Wave Cloud Client configuration_ file that identifies a
            D-Wave system and provides connection information.

        profile (str, optional):
            Profile to select from a D-Wave Cloud Client configuration_ file.

        endpoint (str, optional):
            D-Wave API endpoint URL. If specified, used instead of retrieving a value from
            a D-Wave Cloud Client configuration_ file.

        token (str, optional):
            Authentication token for the D-Wave API to authenticate the client session.
            If specified, used instead of retrieving a value from a D-Wave Cloud Client
            configuration_ file.

        solver (str, optional):
            Solver (a D-Wave system on which to run submitted problems).
            If specified, used instead of retrieving a value from a D-Wave Cloud Client
            configuration_ file.

        proxy (str, optional):
            Proxy URL to be used for accessing the D-Wave API. If specified, used instead of
            retrieving a value from a D-Wave Cloud Client configuration_ file.

    Examples:
        This example creates a :class:`DWaveSampler` based on a fictive user's D-Wave Cloud Client
        configuration_ file and submits a simple Ising problem of just two variables
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

    .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config

    """
    def __init__(self, config_file=None, profile=None, endpoint=None, token=None, solver=None,
                 proxy=None, permissive_ssl=False):

        self.client = client = qpuclient.Client.from_config(config_file=config_file, profile=profile,
                                                            endpoint=endpoint, token=token, proxy=proxy,
                                                            permissive_ssl=permissive_ssl)
        self.solver = solver = client.get_solver(name=solver)

        # need to set up the nodelist and edgelist, properties, parameters
        self._nodelist = sorted(solver.nodes)
        self._edgelist = sorted(set(tuple(sorted(edge)) for edge in solver.edges))
        self._properties = solver.properties.copy()  # shallow copy
        self._parameters = {param: ['parameters'] for param in solver.properties['parameters']}

    @property
    def properties(self):
        """dict: D-Wave solver properties as returned by a SAPI query.

        Solver properties are dependent on the selected D-Wave solver and subject to change;
        for example, new released features may add properties.

        Examples:
            This example creates a :class:`DWaveSampler` and prints the properties retrieved
            from a D-Wave solver selected by the user's default D-Wave Cloud Client
            configuration_ file.

            >>> from dwave.system.samplers import DWaveSampler
            >>> sampler = DWaveSampler()
            >>> sampler.properties    # doctest: +SKIP
            {u'anneal_offset_ranges': [[-0.2197463755538704, 0.03821687759418928],
              [-0.2242514597680286, 0.01718456460967399],
              [-0.20860153999435985, 0.05511969218508182],
            # Snipped above response for brevity

        .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config

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
            from a D-Wave solver selected by the user's default D-Wave Cloud Client
            configuration_ file.

            >>> from dwave.system.samplers import DWaveSampler
            >>> sampler = DWaveSampler()
            >>> sampler.parameters    # doctest: +SKIP
            {u'anneal_offsets': ['parameters'],
            u'anneal_schedule': ['parameters'],
            u'annealing_time': ['parameters'],
            u'answer_mode': ['parameters'],
            u'auto_scale': ['parameters'],
            # Snipped above response for brevity

        .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config

        """
        return self._parameters

    @property
    def edgelist(self):
        """list: List of active couplers for the D-Wave solver.

        Examples:
            This example creates a :class:`DWaveSampler` and prints the active couplers retrieved
            from a D-Wave solver selected by the user's default D-Wave Cloud Client
            configuration_ file.

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

        .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config

        """
        return self._edgelist

    @property
    def nodelist(self):
        """list: List of active qubits for the D-Wave solver.

        Examples:
            This example creates a :class:`DWaveSampler` and prints the active qubits retrieved
            from a D-Wave solver selected by the user's default D-Wave Cloud Client
            configuration_ file.

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

        .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config

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
            user's default D-Wave Cloud Client configuration_ file and submits a simple
            Ising problem of just two variables that map to qubits 0 and 1 on the example
            system. (The simplicity of this example obviates the need for an embedding
            composite---the presence of qubits 0 and 1 on the selected D-Wave system can
            be verified manually.)

            >>> from dwave.system.samplers import DWaveSampler
            >>> sampler = DWaveSampler()
            >>> response = sampler.sample_ising({0: -1, 1: 1}, {})
            >>> for sample in response.samples():    # doctest: +SKIP
            ...    print(sample)
            ...
            {0: 1, 1: -1}

        .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config

        """
        if isinstance(h, list):
            h = dict(enumerate(h))

        variables = set(h).union(*J)
        try:
            active_variables = sorted(variables)
        except TypeError:
            active_variables = list(variables)
        num_variables = len(active_variables)

        data_vector_keys = {'energies': 'energy',
                            'num_occurrences': 'num_occurrences'}
        info_keys = {'timing'}

        future = self.solver.sample_ising(h, J, **kwargs)
        return dimod.Response.from_futures((future,), vartype=dimod.SPIN,
                                           num_variables=num_variables,
                                           data_vector_keys=data_vector_keys,
                                           active_variables=active_variables,
                                           info_keys=info_keys)

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
            user's default D-Wave Cloud Client configuration_ file and submits a simple
            QUBO problem of just two variables that map to coupled qubits 0 and 4 on the
            example system. (The simplicity of this example obviates the need for an embedding
            composite---the presence of qubits 0 and 4, and their coupling, on the selected
            D-Wave system can be verified manually.)

            >>> from dwave.system.samplers import DWaveSampler
            >>> sampler = DWaveSampler()
            >>> Q = {(0, 0): -1, (4, 4): -1, (0, 4): 2}
            >>> response = sampler.sample_qubo(Q)
            >>> for sample in response.samples():    # doctest: +SKIP
            ...    print(sample)
            ...
            {0: 0, 4: 1}

        .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config

        """
        variables = set().union(*Q)
        try:
            active_variables = sorted(variables)
        except TypeError:
            active_variables = list(variables)
        num_variables = len(active_variables)

        data_vector_keys = {'energies': 'energy',
                            'num_occurrences': 'num_occurrences'}
        info_keys = {'timing'}

        future = self.solver.sample_qubo(Q, **kwargs)
        return dimod.Response.from_futures((future,), vartype=dimod.BINARY,
                                           num_variables=num_variables,
                                           data_vector_keys=data_vector_keys,
                                           active_variables=active_variables,
                                           info_keys=info_keys)
