# Copyright 2019 D-Wave Systems Inc.
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

"""Utility functions."""

import os
import json
import networkx as nx
import warnings

__all__ = ['common_working_graph', 'classproperty']


# taken from https://stackoverflow.com/a/39542816, licensed under CC BY-SA 3.0
# not needed in py39+
class classproperty(property):
    def __get__(self, obj, objtype=None):
        return super(classproperty, self).__get__(objtype)


def common_working_graph(graph0, graph1):
    """Creates a graph using the common nodes and edges of two given graphs.

    This function finds the edges and nodes with common labels. Note that this
    not the same as finding the greatest common subgraph with isomorphisms.

    Args:
        graph0: (dict[dict]/:obj:`~networkx.Graph`)
            A NetworkX graph or a dictionary of dictionaries adjacency
            representation.

        graph1: (dict[dict]/:obj:`~networkx.Graph`)
            A NetworkX graph or a dictionary of dictionaries adjacency
            representation.

    Returns:
        :obj:`~networkx.Graph`: A graph with the nodes and edges common to both
        input graphs.

    Examples:

        This example creates a graph that represents a part of a particular 
        Advantage quantum computer's working graph.

        >>> import dwave_networkx as dnx
        >>> from dwave.system import DWaveSampler, common_working_graph
        ...
        >>> sampler = DWaveSampler(solver={'topology__type': 'pegasus'})
        >>> P3 = dnx.pegasus_graph(3)  
        >>> p3_working_graph = common_working_graph(P3, sampler.adjacency)   

    """
    warnings.warn("dwave.system.common_working_graph() is deprecated as of dwave-system 1.23.0 "
                  "and will be removed in dwave-system 2.0. Use networkx.intersection() instead.",
                  DeprecationWarning, stacklevel=2)

    G = nx.Graph()
    G.add_nodes_from(v for v in graph0 if v in graph1)
    G.add_edges_from((u, v) for u in graph0 for v in graph0[u]
                     if v in graph1 and u in graph1[v])

    return(G)


class FeatureFlags:
    """User environment-level Ocean feature flags pertinent to dwave-system."""

    # NOTE: This is an experimental feature. If we decide to keep it, we'll want
    # to move this machinery level up to Ocean-common.

    @staticmethod
    def get(name, default=False):
        try:
            return json.loads(os.getenv('DWAVE_FEATURE_FLAGS')).get(name, default)
        except:
            return default

    @classproperty
    def hss_solver_config_override(cls):
        return cls.get('hss_solver_config_override')
