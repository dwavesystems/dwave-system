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
#
# =============================================================================

import networkx as nx

__all__ = ['common_working_graph']


def common_working_graph(graph0, graph1):
    """Creates a graph using the common nodes and edges of two given graphs.

    Parameters
    ----------
    graph0: (dict[adjacency]/:obj:`~networkx.Graph`)
        A NetworkX graph or a dimod sampler adjacency.

    graph1: (dict[adjacency]/:obj:`~networkx.Graph`)
        A NetworkX graph or a dimod sampler adjacency.

    Returns
    -------
    G : NetworkX Graph
        A graph with the nodes and edges common to both input graphs.

    Examples
    ========

    This example creates a graph that represents a quarter (4 by 4 Chimera tiles)
    of a particular D-Wave system's working graph.

    >>> from dwave.system.samplers import DWaveSampler
    >>> sampler = DWaveSampler(solver={'qpu': True}) # doctest: +SKIP
    >>> C4 = dnx.chimera_graph(4)  # a 4x4 lattice of Chimera tiles
    >>> c4_working_graph = common_graph(C4, sampler.adjacency)   # doctest: +SKIP
    """

    G = nx.Graph()
    G.add_nodes_from(v for v in graph0 if v in graph1)
    G.add_edges_from((u, v) for u in graph0 for v in graph0[u] if v in graph1 and u in graph1[v])

    return(G)
