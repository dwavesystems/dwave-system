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

"""Unembedding samples with broken chains."""

from collections.abc import Callable
from heapq import heapify, heappop

import numpy as np

import dimod

__all__ = ['broken_chains',
           'discard',
           'majority_vote',
           'weighted_random',
           'MinimizeEnergy',
           ]


def broken_chains(samples, chains):
    """Find the broken chains.

    Args:
        samples (array_like):
            Samples as a nS x nV array_like object where nS is the number of samples and nV is the
            number of variables. The values should all be 0/1 or -1/+1.

        chains (list[array_like]):
            List of chains of length nC where nC is the number of chains.
            Each chain should be an array_like collection of column indices in samples.

    Returns:
        :obj:`numpy.ndarray`: A nS x nC boolean array. If i, j is True, then chain j in sample i is
        broken.

    Examples:

        >>> import numpy as np
        ...
        >>> samples = np.array([[-1, +1, -1, +1], [-1, -1, +1, +1]], dtype=np.int8)
        >>> chains = [[0, 1], [2, 3]]
        >>> dwave.embedding.broken_chains(samples, chains)
        array([[True, True],
               [ False,  False]])

        >>> chains = [[0, 2], [1, 3]]
        >>> dwave.embedding.broken_chains(samples, chains)
        array([[False, False],
               [ True,  True]])

    """
    samples, labels = dimod.as_samples(samples)

    if labels != range(len(labels)):
        relabel = {v: idx for idx, v in enumerate(labels)}
        chains = [[relabel[v] for v in chain] for chain in chains]
    else:
        chains = list(map(list, chains))  # because we use them for indexing

    num_samples, num_variables = samples.shape
    num_chains = len(chains)

    broken = np.zeros((num_samples, num_chains), dtype=bool, order='F')

    for cidx, chain in enumerate(chains):
        if isinstance(chain, set):
            chain = list(chain)
        chain = np.asarray(chain)

        if chain.ndim > 1:
            raise ValueError("chains should be 1D array_like objects")

        # chains of length 1, or 0 cannot be broken
        if len(chain) <= 1:
            continue

        all_ = (samples[:, chain] == 1).all(axis=1)
        any_ = (samples[:, chain] == 1).any(axis=1)
        broken[:, cidx] = np.bitwise_xor(all_, any_)

    return broken


def discard(samples, chains):
    """Discard broken chains.

    Args:
        samples (samples_like):
            A collection of samples. `samples_like` is an extension of NumPy's
            array_like. See :func:`dimod.as_samples`.

        chains (list[array_like]):
            List of chains, where each chain is an array_like collection of
            the variables in the same order as their represention in the given
            samples.

    Returns:
        tuple: A 2-tuple containing:

            :obj:`numpy.ndarray`: Unembedded samples as an array of dtype 'int8'.
            Broken chains are discarded.

            :obj:`numpy.ndarray`: Indicies of rows with unbroken chains.

    Examples:
        This example unembeds two samples that chains nodes 0 and 1 to represent
        a single source node. The first sample has an unbroken chain, the second
        a broken chain.

        >>> import dimod
        >>> import numpy as np
        ...
        >>> chains = [(0, 1), (2,)]
        >>> samples = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.int8)
        >>> unembedded, idx = dwave.embedding.discard(samples, chains)
        >>> unembedded
        array([[1, 0]], dtype=int8)
        >>> print(idx)
        [0]

    """
    samples, labels = dimod.as_samples(samples)

    if labels != range(len(labels)):
        relabel = {v: idx for idx, v in enumerate(labels)}
        chains = [[relabel[v] for v in chain] for chain in chains]
    else:
        chains = list(map(list, chains))  # because we use them for indexing

    num_samples, num_variables = samples.shape
    num_chains = len(chains)

    broken = broken_chains(samples, chains)

    unbroken_idxs, = np.where(~broken.any(axis=1))

    chain_variables = np.fromiter((np.asarray(tuple(chain))[0] if isinstance(chain, set) else np.asarray(chain)[0]
                                   for chain in chains),
                                  count=num_chains, dtype=int)

    return samples[np.ix_(unbroken_idxs, chain_variables)], unbroken_idxs


def majority_vote(samples, chains):
    """Unembed samples using the most common value for broken chains.

    Args:
        samples (samples_like):
            A collection of samples. `samples_like` is an extension of NumPy's
            array_like. See :func:`dimod.as_samples`.

        chains (list[array_like]):
            List of chains, where each chain is an array_like collection of
            the variables in the same order as their represention in the given
            samples.

    Returns:
        tuple: A 2-tuple containing:

            :obj:`numpy.ndarray`: Unembedded samples as an nS-by-nC array of
            dtype 'int8', where nC is the number of chains and nS the number
            of samples. Broken chains are resolved by setting the sample value to
            that of most the chain's elements or, for chains without a majority,
            an arbitrary value.

            :obj:`numpy.ndarray`: Indicies of the samples. Equivalent to
            :code:`np.arange(nS)` because all samples are kept and none added.

    Examples:
        This example unembeds samples from a target graph that chains nodes 0 and 1 to
        represent one source node and nodes 2, 3, and 4 to represent another.
        Both samples have one broken chain, with different majority values.

        >>> import dimod
        >>> import numpy as np
        ...
        >>> chains = [(0, 1), (2, 3, 4)]
        >>> samples = np.array([[1, 1, 0, 0, 1], [1, 1, 1, 0, 1]], dtype=np.int8)
        >>> unembedded, idx = dwave.embedding.majority_vote(samples, chains)
        >>> print(unembedded)
        [[1 0]
         [1 1]]
        >>> print(idx)
        [0 1]

    """
    samples, labels = dimod.as_samples(samples)

    if labels != range(len(labels)):
        relabel = {v: idx for idx, v in enumerate(labels)}
        chains = [[relabel[v] for v in chain] for chain in chains]
    else:
        chains = list(map(list, chains))  # because we use them for indexing

    num_samples, num_variables = samples.shape
    num_chains = len(chains)

    unembedded = np.empty((num_samples, num_chains), dtype='int8', order='F')

    # determine if spin or binary. If samples are all 1, then either method works, so we use spin
    # because it is faster
    if samples.all():  # spin-valued
        for cidx, chain in enumerate(chains):
            # we just need the sign for spin. We don't use np.sign because in that can return 0
            # and fixing the 0s is slow.
            unembedded[:, cidx] = 2*(samples[:, chain].sum(axis=1) >= 0) - 1
    else:  # binary-valued
        for cidx, chain in enumerate(chains):
            mid = len(chain) / 2
            unembedded[:, cidx] = (samples[:, chain].sum(axis=1) >= mid)

    return unembedded, np.arange(num_samples)  # we keep all of the samples in this case


def weighted_random(samples, chains):
    """Unembed samples using weighed random choice for broken chains.

    Args:
        samples (samples_like):
            A collection of samples. `samples_like` is an extension of NumPy's
            array_like. See :func:`dimod.as_samples`.

        chains (list[array_like]):
            List of chains, where each chain is an array_like collection of
            the variables in the same order as their represention in the given
            samples.

    Returns:
        tuple: A 2-tuple containing:

            :obj:`numpy.ndarray`: Unembedded samples as an nS-by-nC array of
            dtype 'int8', where nC is the number of chains and nS the number
            of samples. Broken chains are resolved by setting the sample value to
            a random value weighted by frequency of the value in the chain.

            :obj:`numpy.ndarray`: Indicies of the samples. Equivalent to
            :code:`np.arange(nS)` because all samples are kept
            and no samples are added.

    Examples:
        This example unembeds samples from a target graph that chains nodes 0 and 1 to
        represent one source node and nodes 2, 3, and 4 to represent another.
        The sample has broken chains for both source nodes.

        >>> import dimod
        >>> import numpy as np
        ...
        >>> chains = [(0, 1), (2, 3, 4)]
        >>> samples = np.array([[1, 0, 1, 0, 1]], dtype=np.int8)
        >>> unembedded, idx = dwave.embedding.weighted_random(samples, chains)  # doctest: +SKIP
        >>> unembedded  # doctest: +SKIP
        array([[1, 1]], dtype=int8)
        >>> idx  # doctest: +SKIP
        array([0, 1])

    """
    samples, labels = dimod.as_samples(samples)

    if labels != range(len(labels)):
        relabel = {v: idx for idx, v in enumerate(labels)}
        chains = [[relabel[v] for v in chain] for chain in chains]
    else:
        chains = list(map(list, chains))  # because we use them for indexing

    # it sufficies to choose a random index from each chain and use that to construct the matrix
    idx = [np.random.choice(chain) for chain in chains]

    num_samples, num_variables = samples.shape
    return samples[:, idx], np.arange(num_samples)  # we keep all of the samples in this case


class MinimizeEnergy(Callable):
    """Unembed samples by minimizing local energy for broken chains.

    Args:
        bqm (:class:`~dimod.BinaryQuadraticModel`).
            Binary quadratic model associated with the source graph.

        embedding (dict):
            Mapping from source graph to target graph as a dict of form {s: [t, ...], ...},
            where s is a source-model variable and t is a target-model variable.

    Examples:
        This example embeds from a triangular graph to a square graph,
        chaining target-nodes 2 and 3 to represent source-node c, and unembeds minimizing the
        energy for the samples. The first two sample have unbroken chains, the second two have
        broken chains.

        >>> import dimod
        >>> import numpy as np
        ...
        >>> h = {'a': 0, 'b': 0, 'c': 0}
        >>> J = {('a', 'b'): 1, ('b', 'c'): 1, ('a', 'c'): 1}
        >>> bqm = dimod.BinaryQuadraticModel.from_ising(h, J)
        >>> embedding = {'a': [0], 'b': [1], 'c': [2, 3]}
        >>> cbm = dwave.embedding.MinimizeEnergy(bqm, embedding)
        >>> samples = np.array([[+1, -1, +1, +1],
        ...                     [-1, -1, -1, -1],
        ...                     [-1, -1, +1, -1],
        ...                     [+1, +1, -1, +1]], dtype=np.int8)
        >>> chains = [embedding['a'], embedding['b'], embedding['c']]
        >>> unembedded, idx = cbm(samples, chains)
        >>> unembedded
        array([[ 1, -1,  1],
               [-1, -1, -1],
               [-1, -1,  1],
               [ 1,  1, -1]], dtype=int8)
        >>> idx
        array([0, 1, 2, 3])

    """
    def __init__(self, bqm, embedding):
        # this is an awkward construction but we need it to maintain consistency with the other
        # chain break methods
        self.chain_to_var = {frozenset(chain): v for v, chain in embedding.items()}

        self.bqm = bqm

    def __call__(self, samples, chains):
        """
        Args:
            samples (samples_like):
                A collection of samples. `samples_like` is an extension of NumPy's
                array_like. See :func:`dimod.as_samples`.

            chains (list[array_like]):
                List of chains, where each chain is an array_like collection of
                the variables in the same order as their represention in the given
                samples.

        Returns:
            tuple: A 2-tuple containing:

                :obj:`numpy.ndarray`: Unembedded samples as an nS-by-nC array of
                dtype 'int8', where nC is the number of chains and nS the number
                of samples. Broken chains are resolved by greedy energy descent.

                :obj:`numpy.ndarray`: Indicies of the samples. Equivalent to
                :code:`np.arange(nS)` because all samples are kept and none added.

        """
        samples, labels = dimod.as_samples(samples)

        if labels != range(len(labels)):
            relabel = {v: idx for idx, v in enumerate(labels)}
            chains = [[relabel[v] for v in chain] for chain in chains]
        else:
            chains = list(map(list, chains))  # because we use them for indexing

        chain_to_var = self.chain_to_var
        variables = [chain_to_var[frozenset(labels[c] for c in chain)]
                     for chain in chains]

        # we want the bqm by index
        bqm = self.bqm.relabel_variables({v: idx for idx, v in enumerate(variables)}, inplace=False)

        num_chains = len(chains)

        if bqm.vartype is dimod.SPIN:
            ZERO = -1
        else:
            ZERO = 0

        def _minenergy(arr):

            unbroken_arr = np.zeros((num_chains,), dtype=np.int8)
            broken = []

            for cidx, chain in enumerate(chains):
                eq1 = (arr[chain] == 1)
                if not np.bitwise_xor(eq1.all(), eq1.any()):
                    # not broken
                    unbroken_arr[cidx] = arr[chain][0]
                else:
                    broken.append(cidx)

            energies = []
            for cidx in broken:
                en = bqm.linear[cidx] + sum(unbroken_arr[idx] * bqm.adj[cidx][idx] for idx in bqm.adj[cidx])
                energies.append([-abs(en), en, cidx])
            heapify(energies)

            while energies:
                _, e, i = heappop(energies)

                unbroken_arr[i] = val = ZERO if e > 0 else 1

                for energy_triple in energies:
                    k = energy_triple[2]
                    if k in bqm.adj[i]:
                        energy_triple[1] += val * bqm.adj[i][k]
                    energy_triple[0] = -abs(energy_triple[1])

                heapify(energies)

            return unbroken_arr

        num_samples, num_variables = samples.shape
        return np.apply_along_axis(_minenergy, 1, samples), np.arange(num_samples)
