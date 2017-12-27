import dimod
import dwave_micro_client as microclient

from dwave_micro_client_dimod.response import FutureResponse


class DWaveSampler(dimod.TemplateSampler):
    """dimod wrapper for a D-Wave Micro Client.

    Args:
        name (str): Id of the requested solver. None will return the
            default solver.
        url (str): URL of the SAPI server.
        token (str): Authentication token from the SAPI server.
        proxies (dict): Mapping from the connection scheme (http[s]) to
            the proxy server address.
        permissive_ssl (boolean; false by default): Disables SSL
            verification.

    Attributes:
        structure (tuple): (nodes, edges, adjency dict), the set of nodes, edges and adjeceny matrix (as a dict)
            available to the solver.

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
            :obj:`ResponseFuture`

        """
        future = self.solver.sample_ising(linear, quadratic, **kwargs)
        response = FutureResponse(vartype=dimod.Vartype.SPIN)
        response.add_samples_future(future)

        return response

    @dimod.decorators.qubo(1)
    def sample_qubo(self, Q, **kwargs):
        """
        """
        future = self.solver.sample_qubo(Q, **kwargs)
        response = FutureResponse(vartype=dimod.Vartype.BINARY)
        response.add_samples_future(future)

        return response
