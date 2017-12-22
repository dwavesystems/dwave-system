import dimod
import dwave_micro_client as micro
from dimod.responses.response import TemplateResponse
from dimod.utilities import qubo_to_ising

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

        future_response = ResponseFuture(future=future)

        return future_response

    @dimod.decorators.qubo(1)
    def sample_qubo(self, Q, **kwargs):
        """Converts the given QUBO into an Ising problem, then invokes the
        sample_ising method. Returns a future response and doesn't block until get_response is called

        See sample_ising documentation for more information.

        Args:
            Q (dict): A dictionary defining the QUBO. Should be of the form
                {(u, v): bias} where u, v are variables and bias is numeric.
            **kwargs: Any keyword arguments are passed directly to
                sample_ising.

        Returns:
            :obj:`ResponseFuture`:

        """
        h, J, offset = qubo_to_ising(Q)

        future = self.solver.sample_ising(h, J, **kwargs)
        future_response = ResponseFuture(future=future, offset=offset)

        return future_response


class ResponseFuture(TemplateResponse):

    """
        Delays blocking until get_response is called.
        TODO: Add function for returning an iterator that only blocks if no results are ready
    """
    def __init__(self, future=None, offset=None):
        if future is None:
            raise ValueError("Given future can't be None")

        self.future = future
        self.offset = offset

    def get_response(self):
        """Returns a spin response object (numpy compatible or not depending on if numpy is installed"""

        # for now we just wait until the future is done and immediatly load into dimod response
        if _numpy:
            response = dimod.NumpySpinResponse()

            # get the samples in an array
            samples = np.asarray(self.future.samples)
            energies = np.asarray(self.future.energies)

            # finally load into the response
            response.add_samples_from_array(samples, energies)
        else:
            response = dimod.SpinResponse()

            # convert samples to a dict
            samples = (dict(enumerate(sample)) for sample in self.future.samples)
            energies = self.future.energies

            response.add_samples_from(samples, energies)

        # we will want to add other data from Future into the response.

        return response

    def get_qubo_response(self):
        """

            Returns:
                    : A `BinaryResponse`, converted from the `SpinResponse` return
                        from get_response.
        """

        spin_response = self.get_response()
        return spin_response.as_binary(self.offset)
