import dimod
import dwave_micro_client as micro


class DWaveMicroClient(dimod.TemplateSampler):
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
        self.connection = connection = micro.Connection(url, token, proxies, permissive_ssl)
        self.solver = solver = connection.get_solver(solver_name)
        self.name = solver_name

        # initilize dict
        adj = {node: set() for node in solver.nodes}

        # add neighbors.  edges is bi-directional so don't need to add it twice here.
        for edge in solver.edges:
            adj[edge[0]].add(edge[1])

        self.structure = (solver.nodes, solver.edges, adj)

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

        raise NotImplementedError

        future_response = ResponseFuture(future=future)

        return future_response
