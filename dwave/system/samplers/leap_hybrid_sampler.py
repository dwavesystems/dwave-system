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

import functools
import time

from warnings import warn

import dimod

from dimod.exceptions import BinaryQuadraticModelStructureError
from dwave.cloud.exceptions import SolverOfflineError, SolverNotFoundError
from dwave.cloud import Client

from dwave.cloud.solver import UnstructuredSolver

__all__ = ['DWaveSampler']


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
        ...
        >>> # Create a maximum-independent set problem from a random graph
        >>> problem_node_count = 300
        >>> G = nx.random_geometric_graph(problem_node_count, radius=0.0005*problem_node_count)
        >>> qubo = dnx.algorithms.independent_set.maximum_weighted_independent_set_qubo(G)
        >>> bqm = dimod.BQM.from_qubo(qubo)
        ...
        >>> # Find a good solution
        >>> sampler = LeapHybridSampler(solver="hybrid-solver1")    # doctest: +SKIP
        >>> sampleset = sampler.sample(bqm, time_limit=1)           # doctest: +SKIP
        >>> print("Found solution with {} nodes at energy {}.".format(
                  np.sum(result.record.sample), result.first.energy))     # doctest: +SKIP

    """
    def __init__(self, **config):

        self.client = Client.from_config(**config)
        self.solver = self.client.get_solver()

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

    def sample(self, bqm, time_limit, **kwargs):
        """Sample from the specified binary quadratic model.

        Args:
            bqm (:obj:`.BinaryQuadraticModel`):
                The binary quadratic model.

            time_limit (int):
                Maximum run time to allow the solver to work on the problem.
                <<NEED MOTE INFO HERE>>

            **kwargs:
                Optional keyword arguments for the solver, specified in
                :attr:`.LeapHybridSampler.parameters`.

        Returns:
            :class:`dimod.SampleSet`: A `dimod` :obj:`~dimod.SampleSet` object.
            <<MAYBE>> In it this sampler also provides timing information in the `info`
            field as described in the D-Wave System Documentation's
            `timing guide <https://docs.dwavesys.com/docs/latest/doc_timing.html>`_.

        Examples:
            This example builds a random sparse graph and uses a hybrid solver to find a
            maximum independent set.

            >>> import dimod
            >>> import networkx as nx
            >>> import dwave_networkx as dnx
            ...
            >>> # Create a maximum-independent set problem from a random graph
            >>> problem_node_count = 300
            >>> G = nx.random_geometric_graph(problem_node_count, radius=0.0005*problem_node_count)
            >>> qubo = dnx.algorithms.independent_set.maximum_weighted_independent_set_qubo(G)
            >>> bqm = dimod.BQM.from_qubo(qubo)
            ...
            >>> # Find a good solution
            >>> sampler = LeapHybridSampler(solver="hybrid-solver1")    # doctest: +SKIP
            >>> sampleset = sampler.sample(bqm, time_limit=1)           # doctest: +SKIP
            >>> print("Found solution with {} nodes at energy {}.".format(
                      np.sum(result.record.sample), result.first.energy))     # doctest: +SKIP

        """

        sapi_problem_id = self.solver.upload_bqm(bqm).result()
        return self.solver.sample_bqm(sapi_problem_id, **kwargs).result()
