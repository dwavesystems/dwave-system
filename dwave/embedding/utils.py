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

import dimod
import numpy as np

from dwave.embedding.chain_breaks import broken_chains


__all__ = ['target_to_source',
           'chain_to_quadratic',
           'chain_break_frequency',
           'adjacency_to_edges']


def target_to_source(target_adjacency, embedding):
    """Derive the source adjacency from an embedding and target adjacency.

    Args:
        target_adjacency (dict/:class:`networkx.Graph`):
            A dict of the form {v: Nv, ...} where v is a node in the target graph and Nv is the
            neighbors of v as an iterable. This can also be a networkx graph.

        embedding (dict):
            A mapping from a source graph to a target graph.

    Returns:
        dict: The adjacency of the source graph.

    Raises:
        ValueError: If any node in the target_adjacency is assigned more
            than  one node in the source graph by embedding.

    Examples:

        >>> target_adjacency = {0: {1, 3}, 1: {0, 2}, 2: {1, 3}, 3: {0, 2}}  # a square graph
        >>> embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}
        >>> source_adjacency = dwave.embedding.target_to_source(target_adjacency, embedding)
        >>> # triangle graph:
        >>> source_adjacency   # doctest: +SKIP
        {'a': {'b', 'c'}, 'b': {'a', 'c'}, 'c': {'a', 'b'}}

        This function also works with networkx graphs.

        >>> import networkx as nx
        >>> target_graph = nx.complete_graph(5)
        >>> embedding = {'a': {0, 1, 2}, 'b': {3, 4}}
        >>> dwave.embedding.target_to_source(target_graph, embedding)   # doctest: +SKIP
        {'a': {'b'}, 'b': {'a'}}

    """
    # the nodes in the source adjacency are just the keys of the embedding
    source_adjacency = {v: set() for v in embedding}

    # we need the mapping from each node in the target to its source node
    reverse_embedding = {}
    for v, chain in embedding.items():
        for u in chain:
            if u in reverse_embedding:
                raise ValueError("target node {} assigned to more than one source node".format(u))
            reverse_embedding[u] = v

    # v is node in target, n node in source
    for v, n in reverse_embedding.items():
        neighbors = target_adjacency[v]

        # u is node in target
        for u in neighbors:

            # some nodes might not be assigned to chains
            if u not in reverse_embedding:
                continue

            # m is node in source
            m = reverse_embedding[u]

            if m == n:
                continue

            source_adjacency[n].add(m)
            source_adjacency[m].add(n)

    return source_adjacency


def chain_to_quadratic(chain, target_adjacency, chain_strength):
    """Determine the quadratic biases that induce the given chain.

    Args:
        chain (iterable):
            The variables that make up a chain.

        target_adjacency (dict/:class:`networkx.Graph`):
            Should be a dict of the form {s: Ns, ...} where s is a variable
            in the target graph and Ns is the set of neighbours of s.

        chain_strength (float):
            The magnitude of the quadratic bias that should be used to create chains.

    Returns:
        dict[edge, float]: The quadratic biases that induce the given chain.

    Raises:
        ValueError: If the variables in chain do not form a connected subgraph of target.

    Examples:

        >>> chain = {1, 2}
        >>> target_adjacency = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
        >>> dimod.embedding.chain_to_quadratic(chain, target_adjacency, 1)
        {(1, 2): -1}

    """
    quadratic = {}  # we will be adding the edges that make the chain here

    # do a breadth first search
    seen = set()
    try:
        next_level = {next(iter(chain))}
    except StopIteration:
        raise ValueError("chain must have at least one variable")
    while next_level:
        this_level = next_level
        next_level = set()
        for v in this_level:
            if v not in seen:
                seen.add(v)

                for u in target_adjacency[v]:
                    if u not in chain:
                        continue
                    next_level.add(u)
                    if u != v and (u, v) not in quadratic:
                        quadratic[(v, u)] = -chain_strength

    if len(chain) != len(seen):
        raise ValueError('{} is not a connected chain'.format(chain))

    return quadratic


def chain_break_frequency(samples_like, embedding):
    """Determine the frequency of chain breaks in the given samples.

    Args:
        samples_like (samples_like/:obj:`dimod.SampleSet`):
            A collection of raw samples. 'samples_like' is an extension of NumPy's array_like.
            See :func:`dimod.as_samples`.

        embedding (dict):
            Mapping from source graph to target graph as a dict of form {s: {t, ...}, ...},
            where s is a source-model variable and t is a target-model variable.

    Returns:
        dict: Frequency of chain breaks as a dict in the form {s: f, ...},  where s
        is a variable in the source graph and float f the fraction
        of broken chains.

    Examples:
        This example embeds a single source node, 'a', as a chain of two target nodes (0, 1)
        and uses :func:`.chain_break_frequency` to show that out of two synthetic samples,
        one ([-1, +1]) represents a broken chain.

        >>> import numpy as np
        ...
        >>> samples = np.array([[-1, +1], [+1, +1]])
        >>> embedding = {'a': {0, 1}}
        >>> print(dwave.embedding.chain_break_frequency(samples, embedding)['a'])
        0.5


    """
    if isinstance(samples_like, dimod.SampleSet):
        labels = samples_like.variables
        samples = samples_like.record.sample
        num_occurrences = samples_like.record.num_occurrences
    else:
        samples, labels = dimod.as_samples(samples_like)
        num_occurrences = np.ones(samples.shape[0])

    if not all(v == idx for idx, v in enumerate(labels)):
        labels_to_idx = {v: idx for idx, v in enumerate(labels)}
        embedding = {v: {labels_to_idx[u] for u in chain} for v, chain in embedding.items()}

    if not embedding:
        return {}

    variables, chains = zip(*embedding.items())

    broken = broken_chains(samples, chains)

    return {v: float(np.average(broken[:, cidx], weights=num_occurrences))
            for cidx, v in enumerate(variables)}


def edgelist_to_adjacency(edgelist):
    """Converts an iterator of edges to an adjacency dict.

    Args:
        edgelist (iterable):
            An iterator over 2-tuples where each 2-tuple is an edge.

    Returns:
        dict: The adjacency dict. A dict of the form `{v: Nv, ...}` where `v` is
        a node in a graph and `Nv` is the neighbors of `v` as an set.

    """
    adjacency = dict()
    for u, v in edgelist:
        if u in adjacency:
            adjacency[u].add(v)
        else:
            adjacency[u] = {v}
        if v in adjacency:
            adjacency[v].add(u)
        else:
            adjacency[v] = {u}
    return adjacency

def adjacency_to_edges(adjacency):
    """Converts an adjacency dict, networkx graph, or bqm to an edge iterator.

    Args:
        adjacency (dict/:class:`networkx.Graph`/:class:`dimod.BQM`):
            Should be a dict of the form {s: Ns, ...} where s is a variable
            in the graph and Ns is the set of neighbours of s.

    Yields:
        tuple: A 2-tuple, corresponding to an edge in the provided graph

    """
    if hasattr(adjacency, 'edges'):
        yield from adjacency.edges()

    elif hasattr(adjacency, 'quadratic'):
        yield from adjacency.quadratic

    elif hasattr(adjacency, 'items'):
        seen = set()
        for v, Nv in adjacency.items():
            seen.add(v)
            for u in Nv:
                if u not in seen:
                    yield (u, v)
    else:
        raise TypeError("unrecognized type for adjacency -- provide a dict, "
                        "Mapping, networkx.Graph or dimod.BQM")

class intlabel_disjointsets:
    """A disjoint sets implementation with size and path-halving, for graphs 
    labeled [0, ..., n-1]

    Args:
        n (int):
            The number of items in the disjoint sets

    """
    def __init__(self, n):
        self._parent = list(range(n))
        self._size = [1] * n

    def find(self, q):
        """Find the current root for q.

        Args:
            q (int):
                A number in range(n)

        Returns:
            int: the root of the set containing q

        """
        parent = self._parent
        p = parent[q]
        while q != p:
            r = parent[q] = parent[p]
            q, p = p, r
        return p

    def union(self, p, q):
        """Merges the sets containing p and q.

        Args:
            p (int):
                A number in range(n)
            q (int):
                A number in range(n)

        """
        p = self.find(p)
        q = self.find(q)
        a = self._size[p]
        b = self._size[q]
        if p == q:
            return
        if a > b:
            p, q = q, p
        self._parent[p] = q
        self._size[q] = a + b

    def size(self, q):
        """Returns the size of the set containing q.

        Args:
            p (int):
                A number in range(n)

        Returns:
            int: the size of the set containing q
        """
        return self._size[self.find(q)]


