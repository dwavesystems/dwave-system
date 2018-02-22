"""
DWaveSampler
============
"""
import collections

import dimod
import dwave_micro_client as microclient

__all__ = ['DWaveSampler']


class DWaveSampler(dimod.Sampler):
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
            A named 3-tuple with the following properties/values:

                nodelist (list): The nodes available to the sampler.

                edgelist (list[(node, node)]): The edges available to the sampler.

                adjacency (dict): Encodes the edges of the sampler in nested dicts. The keys of
                adjacency are the nodes of the sampler and the values are neighbor-dicts.

        accepted_kwargs (dict[str, :class:`dimod.SamplerKeywordArg`]):
            The keyword arguments accepted by the `sample_ising` and `sample_qubo`
            methods for this sampler.


    .. _configuration: http://dwave-micro-client.readthedocs.io/en/latest/#configuration

    """

    def __init__(self, solver_name=None, url=None, token=None, proxies=None, permissive_ssl=False):
        dimod.Sampler.__init__(self)
        properties = self.properties

        connection = microclient.Connection(url, token, proxies, permissive_ssl)
        self.connection = properties['connection'] = connection
        self.solver = properties['solver'] = solver = connection.get_solver(solver_name)
        self.name = properties['name'] = solver_name

        # initilize adj dict
        adj = {node: set() for node in solver.nodes}

        # add neighbors. edges is bi-directional so don't need to add it twice here.
        for u, v in solver.edges:
            adj[u].add(v)

        # nodelist, make a new list and ensure that it's sorted
        nodelist = sorted(solver.nodes)

        # edgelist, make a new list (and remove doubled edges)
        edgelist = sorted((u, v) for u, v in solver.edges if u <= v)  # all index-labeled

        self.structure = properties['structure'] = Structure(nodelist, edgelist, adj)

        properties.update(self.solver.properties)

        self.sample_kwargs = dict(self.solver.parameters)

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
        response = dimod.Response(dimod.SPIN)
        response.add_samples_future(future)

        return response

    def sample_qubo(self, Q, **kwargs):
        """Sample from the provided QUBO.

        Args:
            Q (dict): The coefficients of the QUBO.
            **kwargs: Parameters for the sampling method, specified per solver.

        Returns:
            :class:`.FutureResponse`

        """
        future = self.solver.sample_qubo(Q, **kwargs)
        response = dimod.Response(dimod.BINARY)
        response.add_samples_future(future)

        return response


Structure = collections.namedtuple("Structure", ['nodelist', 'edgelist', 'adjacency'])
