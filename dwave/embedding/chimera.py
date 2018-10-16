# Copyright 2018 D-Wave Systems Inc.
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

import dwave_networkx as dnx
import networkx as nx

from dwave.embedding.polynomialembedder import processor

__all__ = ['find_clique_embedding', 'find_biclique_embedding']


@nx.utils.decorators.nodes_or_number(0)
def find_clique_embedding(k, m, n=None, t=None, target_edges=None):
    """Find an embedding for a clique.

    Args:
        k (int/iterable):
            If k is an integer, generates an embedding for a clique of size k labelled [0,k-1].
            If k is an iterable, generates an embedding for a clique of size len(k) where k
            provides the variable labels.

        m (int):
            Number of rows in the Chimera lattice.

        n (int, optional, default=m):
            Number of columns in the Chimera lattice.

        t (int, optional, default 4):
            Size of the shore within each Chimera tile.

        target_edges (iterable[edge]):
            A list of edges in the target Chimera graph. Expects the nodes to be labelled as
            returned by :func:`~dwave_networkx.generators.chimera_graph`.

    Returns:
        dict: An embedding mapping a clique to the Chimera lattice.

    Examples:

        >>> from dwave.embedding.chimera import find_clique_embedding
        ...
        >>> embedding = find_clique_embedding(3, m=2, n=2, t=4)
        >>> embedding  # doctest: +SKIP
        {0: [20, 16], 1: [21, 17], 2: [22, 18]}

        >>> from dwave.embedding.chimera import find_clique_embedding
        ...
        >>> embedding = find_clique_embedding(['a', 'b', 'c'], m=2, n=2, t=4)
        >>> embedding  # doctest: +SKIP
        {'a': [20, 16], 'b': [21, 17], 'c': [22, 18]}

    """
    _, nodes = k

    m, n, t, target_edges = _chimera_input(m, n, t, target_edges)

    embedding = processor(target_edges, M=m, N=n, L=t).tightestNativeClique(len(nodes))

    if not embedding:
        raise ValueError("cannot find a K{} embedding for given Chimera lattice".format(k))

    return dict(zip(nodes, embedding))


@nx.utils.decorators.nodes_or_number(0)
@nx.utils.decorators.nodes_or_number(1)
def find_biclique_embedding(a, b, m, n=None, t=None, target_edges=None):
    """Find an embedding for a biclique.

    Args:
        a (int/iterable):
            If a is an integer, generates an embedding for a biclique with the left shore of size a
            labelled [0,a-1].
            If a is an iterable, generates an embedding for a biclique with the left shore of size
            len(a) where a provides the variable labels.

        b (int/iterable):
            If b is an integer, generates an embedding for a biclique with the right shore of size b
            labelled [0,b-1].
            If b is an iterable, generates an embedding for a biclique with the right shore of size
            len(b) where a provides the variable labels.

        m (int):
            Number of rows in the Chimera lattice.

        n (int, optional, default=m):
            Number of columns in the Chimera lattice.

        t (int, optional, default 4):
            Size of the shore within each Chimera tile.

        target_edges (iterable[edge]):
            A list of edges in the target Chimera graph. Expects the nodes to be labelled as
            returned by :func:`~dwave_networkx.generators.chimera_graph`.

    Returns:
        tuple: A 2-tuple containing:

            dict: An embedding mapping the left shore of the biclique to the Chimera lattice.

            dict: An embedding mapping the right shore of the biclique to the Chimera lattice

    Examples:

        >>> from dwave.embedding.chimera import find_biclique_embedding
        ...
        >>> left, right = find_biclique_embedding(3, 3, m=2, n=2, t=4)
        >>> left  # doctest: +SKIP
        {0: [28], 1: [29], 2: [30]}
        >>> right  # doctest: +SKIP
        {0: [24], 1: [25], 2: [26]}

    """
    _, anodes = a
    _, bnodes = b

    m, n, t, target_edges = _chimera_input(m, n, t, target_edges)
    embedding = processor(target_edges, M=m, N=n, L=t).tightestNativeBiClique(len(anodes), len(bnodes))

    if not embedding:
        raise ValueError("cannot find a K{},{} embedding for given Chimera lattice".format(a, b))

    left, right = embedding
    return dict(zip(anodes, left)), dict(zip(bnodes, right))


def find_grid_embedding(dim, m, n=None, t=4):
    """Find an embedding for a grid.

    Args:
        dim (iterable[int]):
            The size of each grid dimension. Length can be between 1 and 3.

        m (int):
            Number of rows in the Chimera lattice.

        n (int, optional, default=m):
            Number of columns in the Chimera lattice.

        t (int, optional, default 4):
            Size of the shore within each Chimera tile.

    Returns:
        dict: An embedding mapping a grid to the Chimera lattice.

    Examples:

        >>> from dwave.embedding.chimera import find_grid_embedding
        ...
        >>> embedding = find_grid_embedding([2, 3], m=12, n=12, t=4)
        >>> embedding  # doctest: +SKIP
        {(0, 0): [0, 4],
         (0, 1): [8, 12],
         (0, 2): [16, 20],
         (1, 0): [96, 100],
         (1, 1): [104, 108],
         (1, 2): [112, 116]}

    """

    m, n, t, target_edges = _chimera_input(m, n, t, None)
    indexer = dnx.generators.chimera.chimera_coordinates(m, n, t)

    dim = list(dim)
    num_dim = len(dim)
    if num_dim == 1:
        def _key(row, col, aisle): return row
        dim.extend([1, 1])
    elif num_dim == 2:
        def _key(row, col, aisle): return row, col
        dim.append(1)
    elif num_dim == 3:
        def _key(row, col, aisle): return row, col, aisle
    else:
        raise ValueError("find_grid_embedding supports between one and three dimensions")

    rows, cols, aisles = dim
    if rows > m or cols > n or aisles > t:
        msg = ("the largest grid that can fit in a ({}, {}, {}) Chimera-lattice "
               "is {}x{}x{}").format(m, n, t, rows, cols, aisles)
        raise ValueError(msg)

    return {_key(row, col, aisle): [indexer.int((row, col, 0, aisle)), indexer.int((row, col, 1, aisle))]
            for row in range(dim[0]) for col in range(dim[1]) for aisle in range(dim[2])}


def _chimera_input(m, n=None, t=None, target_edges=None):
    if not isinstance(m, int):
        raise TypeError('Chimera lattice parameter m must be an int and >= 1')
    if m <= 0:
        raise ValueError('Chimera lattice parameter m must be an int and >= 1')

    if n is None:
        n = m
    else:
        if not isinstance(n, int):
            raise TypeError('Chimera lattice parameter n must be an int and >= 1')
        if n <= 0:
            raise ValueError('Chimera lattice parameter n must be an int and >= 1')
    if t is None:
        t = 4
    else:
        if not isinstance(t, int):
            raise TypeError('Chimera lattice parameter t must be an int and >= 1')
        if t <= 0:
            raise ValueError('Chimera lattice parameter t must be an int and >= 1')

    if target_edges is None:
        target_edges = dnx.chimera_graph(m, n, t).edges

    return m, n, t, target_edges
