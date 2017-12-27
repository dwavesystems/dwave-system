"""
DWaveSampler
============
"""
import dimod
import dwave_micro_client as microclient

from dwave_micro_client_dimod.response import FutureResponse

__all__ = ['DWaveSampler']


class DWaveSampler(dimod.TemplateSampler):
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
        structure (tuple):
            A 3-tuple:

                list: The nodes available to the sampler.

                list[(node, node)]: The edges available to the sampler.

                dict: Encodes the edges of the sampler in nested dicts. The keys of adj
                are the nodes of the sampler and the values are neighbor-dicts.

        accepted_kwargs (dict[str, :class:`dimod.SamplerKeywordArg`]):
            The keyword arguments accepted by the `sample_ising` and `sample_qubo`
            methods for this sampler.


    .. _configuration: http://dwave-micro-client.readthedocs.io/en/latest/#configuration

    """

    def __init__(self, solver_name=None, url=None, token=None, proxies=None, permissive_ssl=False):
        self.connection = connection = microclient.Connection(url, token, proxies, permissive_ssl)
        self.solver = solver = connection.get_solver(solver_name)
        self.name = solver_name

        # initilize adj dict
        adj = {node: set() for node in solver.nodes}

        # add neighbors. edges is bi-directional so don't need to add it twice here.
        for u, v in solver.edges:
            adj[u].add(v)

        # nodelist, make a new list and ensure that it's sorted
        nodelist = sorted(solver.nodes)

        # edgelist, make a new list (and remove doubled edges)
        edgelist = sorted((u, v) for u, v in solver.edges if u <= v)  # all index-labeled

        self.structure = (nodelist, edgelist, adj)

    def my_kwargs(self):
        """The keyword arguments accepted by DWaveSampler

        Returns:
            dict[str: :class:`.SamplerKeywordArg`]: The keyword arguments
            accepted by the `sample_ising` and `sample_qubo` methods for this
            sampler or the top-level composite layer. For all accepted keyword
            arguments see `accepted_kwargs`.

        """
        kwargs = dimod.TemplateSampler.my_kwargs(self)
        for param in self.solver.parameters:
            # this should be replaced by more complete info later
            kwargs[param] = dimod.SamplerKeywordArg(param)
        return kwargs

    @dimod.decorators.ising(1, 2)
    def sample_ising(self, linear, quadratic, **kwargs):
        """Sample from the provided Ising model.

        Args:
            linear (list/dict): Linear terms of the model.
            quadratic (dict of (int, int):float): Quadratic terms of the model.
            **kwargs: Parameters for the sampling method, specified per solver.

        Returns:
            :class:`.FutureResponse`

        """
        future = self.solver.sample_ising(linear, quadratic, **kwargs)
        response = FutureResponse(vartype=dimod.Vartype.SPIN)
        response.add_samples_future(future)

        return response

    @dimod.decorators.qubo(1)
    def sample_qubo(self, Q, **kwargs):
        """Sample from the provided QUBO.

        Args:
            Q (dict): The coefficients of the QUBO.
            **kwargs: Parameters for the sampling method, specified per solver.

        Returns:
            :class:`.FutureResponse`

        """
        future = self.solver.sample_qubo(Q, **kwargs)
        response = FutureResponse(vartype=dimod.Vartype.BINARY)
        response.add_samples_future(future)

        return response
