import dimod
import dwave_micro_client as micro

try:
    import numpy as np
    _numpy = True
except ImportError:  # pragma: no cover
    _numpy = False


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

        #initilize dict
        adj = {node: set() for node in solver.nodes}

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
            :obj:`SpinResponse`

        """
        future = self.solver.sample_ising(linear, quadratic, **kwargs)

        # for now we just wait until the future is done and immediatly load into dimod response
        if _numpy:
            response = dimod.NumpySpinResponse()

            # get the samples in an array
            samples = np.asarray(future.samples)
            energies = np.asarray(future.energies)

            # finally load into the response
            response.add_samples_from_array(samples, energies)
        else:
            response = dimod.SpinResponse()

            # convert samples to a dict
            samples = (dict(enumerate(sample)) for sample in future.samples)
            energies = future.energies

            response.add_samples_from(samples, energies)

        # we will want to add other data from Future into the response.

        return response
