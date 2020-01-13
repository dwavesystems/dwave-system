# Copyright 2020 D-Wave Systems Inc.
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
A :std:doc:`dimod sampler <dimod:reference/samplers>` for Leap's hybrid solvers.

"""
from __future__ import division
import numpy as np
from warnings import warn

import dimod
from dwave.cloud import Client

__all__ = ['LeapHybridSampler']

class LeapHybridSampler(dimod.Sampler):
    """A class for using the Leap's cloud-based hybrid solvers.

    Uses parameters set in a configuration file, as environment variables, or
    explicitly as input arguments for selecting and communicating with a hybrid solver.
    For more information, see
    `D-Wave Cloud Client <https://docs.ocean.dwavesys.com/projects/cloud-client/en/latest/>`_.

    Inherits from :class:`dimod.Sampler`.

    Args:
        config_file (str, optional):
            Path to a configuration file that identifies a hybrid solver and provides
            connection information.

        profile (str, optional):
            Profile to select from the configuration file.

        endpoint (str, optional):
            D-Wave API endpoint URL.

        token (str, optional):
            Authentication token for the D-Wave API to authenticate the client session.

        solver (dict/str, optional):
            Solver (a hybrid solver on which to run submitted problems) to select,
            formatted as a string.

        proxy (str, optional):
            Proxy URL to be used for accessing the D-Wave API.

        **config:
            Keyword arguments passed directly to :meth:`~dwave.cloud.client.Client.from_config`.

    Examples:
        This example builds a random sparse graph and uses a hybrid solver to find a
        maximum independent set.

        >>> import dimod
        >>> import networkx as nx
        >>> import dwave_networkx as dnx
        >>> import numpy as np
        ...
        >>> # Create a maximum-independent set problem from a random graph
        >>> problem_node_count = 300
        >>> G = nx.random_geometric_graph(problem_node_count, radius=0.0005*problem_node_count)
        >>> qubo = dnx.algorithms.independent_set.maximum_weighted_independent_set_qubo(G)
        >>> bqm = dimod.BQM.from_qubo(qubo)
        ...
        >>> # Find a good solution
        >>> sampler = LeapHybridSampler(solver="hybrid-solver1")    # doctest: +SKIP
        >>> sampleset = sampler.sample(bqm, time_limit=10)           # doctest: +SKIP
        >>> print("Found solution with {} nodes at energy {}.".format(
                  np.sum(sampleset.record.sample),
                  sampleset.first.energy))     # doctest: +SKIP
    """

    def __init__(self, **config):

        if config.get('solver_features') is not None:
            warn("'solver_features' argument has been renamed to 'solver'.", DeprecationWarning)

            if config.get('solver') is not None:
                raise ValueError("cannot combine 'solver' and 'solver_features'")

            config['solver'] = config.pop('solver_features')

        # Non-hybrid named solvers are caught post client.get_solver() resolution
        if (config.get('solver') is not None) and (not isinstance(config['solver'], str)):
            if 'category' not in config['solver'].keys():
                config['solver']['category'] = 'hybrid'
            elif config['solver']['category'] is not 'hybrid':
                raise ValueError("the only 'category' this sampler supports is 'hybrid'.")

        self.client = Client.from_config(**config)
        self.solver = self.client.get_solver()

        if ('category' not in self.properties.keys()) or (
                      not self.properties['category'] == 'hybrid'):
            raise ValueError("selected solver is not a hybrid solver.")

    @property
    def properties(self):
        """dict: solver properties as returned by a SAPI query.

        Solver properties are dependent on the selected solver and subject to change.
        """
        try:
            return self._properties
        except AttributeError:
            self._properties = properties = self.solver.properties.copy()
            return properties

    @property
    def parameters(self):
        """dict[str, list]: solver parameters in the form of a dict, where keys are
        keyword parameters accepted by a SAPI query and values are lists of properties in
        :attr:`.LeapHybridSampler.properties` for each key.

        Solver parameters are dependent on the selected solver and subject to change.
        """
        try:
            return self._parameters
        except AttributeError:
            parameters = {param: ['parameters']
                          for param in self.properties['parameters']}
            self._parameters = parameters
            return parameters

    def sample(self, bqm, time_limit=None, **kwargs):
        """Sample from the specified binary quadratic model.

        Args:
            bqm (:obj:`.BinaryQuadraticModel`):
                The binary quadratic model.

            time_limit (int):
                Maximum run time, in seconds, to allow the solver to work on the problem.
                Must be a least the minimum required for the number of problem variables,
                which is calculated and set by default.
                The minimum time for a hybrid solver is specified as a piecewise-linear
                curve defined by a set of floating-point pairs, the `minimum_time_limit`
                field under :attr:`.LeapHybridSampler.properties`. The first element in each
                pair is the number of problem variables; the second is the minimum
                required time. The minimum time for any particular number of variables
                is a linear interpolation calculated on two pairs that represent the
                relevant range for the given number of variables.
                For example, if `LeapHybridSampler().properties["minimum_time_limit"]`
                returns `[[1, 0.1], [100, 10.0], [1000, 20.0]]`, then the minimum time
                for a 50-variable problem is 5 seconds, the linear interpolation of the
                first two pairs that represent problems with between 1 to 100 variables.

            **kwargs:
                Optional keyword arguments for the solver, specified in
                :attr:`.LeapHybridSampler.parameters`.

        Returns:
            :class:`dimod.SampleSet`: A `dimod` :obj:`~dimod.SampleSet` object.
            <<TOD DO: add once I have input:>> In it this sampler also provides
            timing information in the `info` field.

        Examples:
            This example builds a random sparse graph and uses a hybrid solver to find a
            maximum independent set.

            >>> import dimod
            >>> import networkx as nx
            >>> import dwave_networkx as dnx
            >>> import numpy as np
            ...
            >>> # Create a maximum-independent set problem from a random graph
            >>> problem_node_count = 300
            >>> G = nx.random_geometric_graph(problem_node_count, radius=0.0005*problem_node_count)
            >>> qubo = dnx.algorithms.independent_set.maximum_weighted_independent_set_qubo(G)
            >>> bqm = dimod.BQM.from_qubo(qubo)
            ...
            >>> # Find a good solution
            >>> sampler = LeapHybridSampler(solver="hybrid-solver1")    # doctest: +SKIP
            >>> sampleset = sampler.sample(bqm, time_limit=10)           # doctest: +SKIP
            >>> print("Found solution with {} nodes at energy {}.".format(
                      np.sum(sampleset.record.sample),
                      sampleset.first.energy))     # doctest: +SKIP
        """

        xx, yy = zip(*self.properties["minimum_time_limit"])
        min_time_limit = np.interp([len(bqm.variables)], xx, yy)[0]

        if time_limit is None:
            time_limit = min_time_limit
        if time_limit < min_time_limit:
            msg = ("time limit for problem size {} must be at least {}"
                   ).format(len(bqm.variables), min_time_limit)
            raise ValueError(msg)

        sapi_problem_id = self.solver.upload_bqm(bqm).result()
        return self.solver.sample_bqm(sapi_problem_id, time_limit=time_limit).sampleset
