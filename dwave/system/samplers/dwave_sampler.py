"""
Creates a dimod Sampler_ for the D-Wave System.

.. _Sampler: http://dimod.readthedocs.io/en/latest/reference/samplers.html#samplers-and-composites
"""
import dimod
import dwave.cloud.qpu as qpuclient

__all__ = ['DWaveSampler']


class DWaveSampler(dimod.Sampler, dimod.Structured):
    """dimod Sampler for a D-Wave System.

    A :class:`dimod.Sampler` that allows the D-Wave System to be used with the Ocean tools.

    Also inherits from :class:`dimod.Structured`.

    Args:
        solver_name (str, optional):
            ID of the requested solver. If unspecificed, will return the default
            solver (see configuration_).

        url (str, optional):
            URL of the SAPI server. If unspecificed, will return the default URL
            (see configuration_).

        token (str, optional):
            Authentication token from the SAPI server. If unspecificed, will return
            the default token (see configuration_).

        proxies (dict, optional):
            Mapping from the connection scheme (http[s]) to the proxy server
            address.

        permissive_ssl (boolean, optional, default=False):
            Disables SSL verification.

    .. _configuration: http://dwave-micro-client.readthedocs.io/en/latest/#configuration

    """
    def __init__(self, solver_name=None, url=None, token=None, proxies=None, permissive_ssl=False):

        self.client = client = qpuclient.Client(url=url, token=token, proxies=proxies,
                                                permissive_ssl=permissive_ssl)
        self.solver = solver = client.get_solver(solver_name)

        # need to set up the nodelist and edgelist, properties, parameters
        self._nodelist = sorted(solver.nodes)
        self._edgelist = sorted(set(tuple(sorted(edge)) for edge in solver.edges))
        self._properties = solver.properties.copy()  # shallow copy
        self._parameters = {param: ['parameters'] for param in solver.properties['parameters']}

    @property
    def properties(self):
        """dict: The properties as exposed by the SAPI web service."""
        return self._properties

    @property
    def parameters(self):
        """dict[str, list]: The keys are the keyword parameters accepted by SAPI web service. The
        values are lists properties in :attr:`.DWaveSampler.properties` that are relevent to the
        keyword.
        """
        return self._parameters

    @property
    def edgelist(self):
        """list: The list of active couplers."""
        return self._edgelist

    @property
    def nodelist(self):
        """list: The list of active qubits."""
        return self._nodelist

    def sample_ising(self, h, J, **kwargs):
        """Sample from the provided Ising model.

        Args:
            h (list/dict):
                The linear biases of the model.

            quadratic (dict[(int, int): float]):
                The quadratic biases of the model.

            **kwargs:
                Optional keyword arguments for the sampling method, specified per solver in
                :attr:`.DWaveSampler.parameters`

        Returns:
            :class:`dimod.Response`

        """
        if isinstance(h, list):
            h = dict(enumerate(h))

        variables = set(h).union(*J)
        try:
            active_variables = sorted(variables)
        except TypeError:
            active_variables = list(variables)
        num_variables = len(active_variables)

        data_vector_keys = {'energies': 'energy'}

        future = self.solver.sample_ising(h, J, **kwargs)
        return dimod.Response.from_futures((future,), vartype=dimod.SPIN,
                                           num_variables=num_variables,
                                           data_vector_keys=data_vector_keys,
                                           active_variables=active_variables)

    def sample_qubo(self, Q, **kwargs):
        """Sample from the provided QUBO.

        Args:
            Q (dict):
                The QUBO coefficients.

            **kwargs:
                Optional keyword arguments for the sampling method, specified per solver in
                :attr:`.DWaveSampler.parameters`

        Returns:
            :class:`dimod.Response`

        """
        variables = set().union(*Q)
        try:
            active_variables = sorted(variables)
        except TypeError:
            active_variables = list(variables)
        num_variables = len(active_variables)

        data_vector_keys = {'energies': 'energy'}

        future = self.solver.sample_qubo(Q, **kwargs)
        return dimod.Response.from_futures((future,), vartype=dimod.SPIN,
                                           num_variables=num_variables,
                                           data_vector_keys=data_vector_keys,
                                           active_variables=active_variables)
