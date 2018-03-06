"""
todo
"""
import collections

import dimod
import dwave.cloud.qpu as qpuclient

__all__ = ['DWaveSampler']


class DWaveSampler(dimod.Sampler, dimod.Structured):
    """dimod wrapper for a D-Wave Micro Client.

    Args:
        solver_name (str, optional):
            Id of the requested solver. None will return the default
            solver (see configuration_).
        url (str, optional):
            URL of the SAPI server. None will return the default url
            (see configuration_).
        token (str, optional):
            Authentication token from the SAPI server. None will return
            the default token (see configuration_).
        proxies (dict, optional):
            Mapping from the connection scheme (http[s]) to the proxy server
            address.
        permissive_ssl (boolean, optional, default=False):
            Disables SSL verification.

    Attributes:
        todo

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
        return self._properties

    @property
    def parameters(self):
        return self._parameters

    @property
    def edgelist(self):
        return self._edgelist

    @property
    def nodelist(self):
        return self._nodelist

    def sample_ising(self, h, J, **kwargs):
        """Sample from the provided Ising model.

        Args:
            linear (list/dict): Linear terms of the model.
            quadratic (dict of (int, int):float): Quadratic terms of the model.
            **kwargs: Parameters for the sampling method, specified per solver.

        Returns:
            :class:`.FutureResponse`

        """
        num_variables = len(h)
        data_vector_keys = {'energies': 'energy'}
        if isinstance(h, list):
            active_variables = list(range(num_variables))
        else:
            active_variables = list(h)

        future = self.solver.sample_ising(h, J, **kwargs)
        return dimod.Response.from_futures((future,), vartype=dimod.SPIN,
                                           num_variables=num_variables,
                                           data_vector_keys=data_vector_keys,
                                           active_variables=active_variables)

    def sample_qubo(self, Q, **kwargs):
        """Sample from the provided QUBO.

        Args:
            Q (dict): The coefficients of the QUBO.
            **kwargs: Parameters for the sampling method, specified per solver.

        Returns:
            :class:`.FutureResponse`

        """
        active_variables = list(set().union(*Q))

        num_variables = len(active_variables)
        data_vector_keys = {'energies': 'energy'}
        active_variables = list(variables)

        future = self.solver.sample_qubo(Q, **kwargs)
        return dimod.Response.from_futures((future,), vartype=dimod.SPIN,
                                           num_variables=num_variables,
                                           data_vector_keys=data_vector_keys,
                                           active_variables=active_variables)
