# Copyright 2022 D-Wave Systems Inc.
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

import dwave_networkx as dnx
import networkx as nx

from minorminer.busclique import busgraph_cache


def _get_target_graph(m=None, target_graph=None):
    if target_graph is None:
        if m is None:
            raise TypeError("m and target_graph cannot both be None.")
        target_graph = dnx.zephyr_graph(m)
    return target_graph


@nx.utils.decorators.nodes_or_number(0)
def find_clique_embedding(k, m=None, target_graph=None):
    """Find an embedding for a clique in a Zephyr graph.

    Given a clique (fully connected graph) and target Zephyr graph, attempts
    to find an embedding.

    Args:
        k (int/iterable/:obj:`networkx.Graph`): A complete graph to embed,
            formatted as a number of nodes, node labels, or a NetworkX graph.

        m (int): Number of tiles in a row of a square Zephyr graph.  Required
            to generate an ``m``-by-``m`` Zephyr graph when ``target_graph`` is None.

        target_graph (:obj:`networkx.Graph`): A Zephyr graph.  Required when
            ``m`` is None.

    Returns:
        dict: An embedding as a dict, where keys represent the clique's nodes
        and values, formatted as lists, represent chains of zephyr coordinates.

    Examples:
        This example finds an embedding for a :math:`K_5` complete graph in a
        2-by-2 Zephyr graph.

        >>> from dwave.embedding.zephyr import find_clique_embedding
        >>> find_clique_embedding(5, 2)
        {0: (16, 96), 1: (18, 98), 2: (20, 100), 3: (22, 102), 4: (24, 104)}
    """
    _, nodes = k
    g = _get_target_graph(m, target_graph)
    embedding = busgraph_cache(g).find_clique_embedding(nodes)

    if len(embedding) != len(nodes):
        raise ValueError("No clique embedding found")

    return embedding


@nx.utils.decorators.nodes_or_number(0)
@nx.utils.decorators.nodes_or_number(1)
def find_biclique_embedding(a, b, m=None, target_graph=None):
    """Find an embedding for a biclique in a Zephyr graph.

    Given a biclique (a bipartite graph where every vertex in a set in
    connected to all vertices in the other set) and a target :term:`Zephyr`
    graph, attempts to find an embedding.

    Args:
        a (int/iterable):
            Describes the left shore of the biclique to embed.  If ``a`` is an
            integer, the left shore will be labelled [0, a-1].  If ``a`` is an
            iterable, the left shore will be labelled by ``a``.

        b (int/iterable):
            Describes the right shore of the biclique to embed.  If ``b`` is an
            integer and ``a`` is an iterable, the right shore will be labelled
            [0, b-1].  If both ``a`` and ``b`` are integers, the right shore
            will be labelled [a, a+b-1].  If ``b`` is an iterable, the right
            shore will be labelled by ``b``.

        m (int): Number of tiles in a row of a square Zephyr graph.  Required to
            generate an ``m``-by-``m`` Zephyr graph when ``target_graph`` is None.

        target_graph (:obj:`networkx.Graph`): A Zephyr graph.  Required when ``m``
            is None.

    Returns:
        tuple: A 2-tuple containing:

            dict: An embedding mapping the left shore of the biclique to
            the Zephyr lattice.

            dict: An embedding mapping the right shore of the biclique to
            the Zephyr lattice.

    Examples:
        This example finds an embedding for an alphanumerically labeled
        biclique in a 2x2 Zephyr graph.

        >>> from dwave.embedding.zephyr import find_biclique_embedding
        >>> left, right = find_biclique_embedding(['a', 'b', 'c'], ['d', 'e'], 2)
        >>> print(left, right)
        {'a': (0,), 'b': (4,), 'c': (8,)} {'d': (80,), 'e': (84,)}
    """
    _a, anodes = a
    _b, bnodes = b

    if isinstance(_a, int) and isinstance(_b, int):
        bnodes = [len(anodes) + x for x in bnodes]

    if set(anodes).intersection(set(bnodes)):
        raise ValueError("a and b overlap")

    g = _get_target_graph(m, target_graph)
    embedding = busgraph_cache(g).find_biclique_embedding(len(anodes), len(bnodes))

    if not embedding:
        raise ValueError("No biclique embedding found")

    return ({x: embedding[anodes.index(x)] for x in anodes},
            {y: embedding[bnodes.index(y) + len(anodes)] for y in bnodes})
