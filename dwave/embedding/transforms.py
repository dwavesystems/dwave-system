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

from __future__ import division

import itertools

import numpy as np
import dimod

from six import iteritems, itervalues

from dwave.embedding.chain_breaks import majority_vote, broken_chains
from dwave.embedding.exceptions import MissingEdgeError, MissingChainError, InvalidNodeError
from dwave.embedding.utils import chain_to_quadratic


__all__ = ['embed_bqm',
           'embed_ising',
           'embed_qubo',
           'unembed_sampleset',
           ]


def embed_bqm(source_bqm, embedding, target_adjacency, chain_strength=1.0,
              smear_vartype=None):
    """Embed a binary quadratic model onto a target graph.

    Args:
        source_bqm (:obj:`.BinaryQuadraticModel`):
            Binary quadratic model to embed.

        embedding (dict):
            Mapping from source graph to target graph as a dict of form {s: {t, ...}, ...},
            where s is a source-model variable and t is a target-model variable.

        target_adjacency (dict/:obj:`networkx.Graph`):
            Adjacency of the target graph as a dict of form {t: Nt, ...},
            where t is a variable in the target graph and Nt is its set of neighbours.

        chain_strength (float, optional):
            Magnitude of the quadratic bias (in SPIN-space) applied between
            variables to create chains, with the energy penalty of chain breaks
            set to 2 * `chain_strength`.

        smear_vartype (:class:`.Vartype`, optional, default=None):
            Determines whether the linear bias of embedded variables is smeared
            (the specified value is evenly divided as biases of a chain in the
            target graph) in SPIN or BINARY space. Defaults to the
            :class:`.Vartype` of `source_bqm`.

    Returns:
        :obj:`.BinaryQuadraticModel`: Target binary quadratic model.

    Examples:
        This example embeds a triangular binary quadratic model representing
        a :math:`K_3` clique into a square target graph by mapping variable `c`
        in the source to nodes `2` and `3` in the target.

        >>> import networkx as nx
        ...
        >>> target = nx.cycle_graph(4)
        >>> # Binary quadratic model for a triangular source graph
        >>> h = {'a': 0, 'b': 0, 'c': 0}
        >>> J = {('a', 'b'): 1, ('b', 'c'): 1, ('a', 'c'): 1}
        >>> bqm = dimod.BinaryQuadraticModel.from_ising(h, J)
        >>> # Variable c is a chain
        >>> embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}
        >>> # Embed and show the chain strength
        >>> target_bqm = dwave.embedding.embed_bqm(bqm, embedding, target)
        >>> target_bqm.quadratic[(2, 3)]
        -1.0
        >>> print(target_bqm.quadratic)  # doctest: +SKIP
        {(0, 1): 1.0, (0, 3): 1.0, (1, 2): 1.0, (2, 3): -1.0}


    See also:
        :func:`.embed_ising`, :func:`.embed_qubo`

    """
    if smear_vartype is dimod.SPIN and source_bqm.vartype is dimod.BINARY:
        return embed_bqm(source_bqm.spin, embedding, target_adjacency,
                         chain_strength=chain_strength, smear_vartype=None).binary
    elif smear_vartype is dimod.BINARY and source_bqm.vartype is dimod.SPIN:
        return embed_bqm(source_bqm.binary, embedding, target_adjacency,
                         chain_strength=chain_strength, smear_vartype=None).spin

    # create a new empty binary quadratic model with the same class as source_bqm
    try:
        target_bqm = source_bqm.base.empty(source_bqm.vartype)
    except AttributeError:
        # dimod < 0.9.0
        target_bqm = source_bqm.empty(source_bqm.vartype)

    # add the offset
    target_bqm.add_offset(source_bqm.offset)

    # start with the linear biases, spreading the source bias equally over the target variables in
    # the chain
    for v, bias in iteritems(source_bqm.linear):

        if v in embedding:
            chain = embedding[v]
        else:
            raise MissingChainError(v)

        if any(u not in target_adjacency for u in chain):
            raise InvalidNodeError(v, next(u not in target_adjacency for u in chain))

        b = bias / len(chain)

        target_bqm.add_variables_from({u: b for u in chain})

    # next up the quadratic biases, spread the quadratic biases evenly over the available
    # interactions
    for (u, v), bias in iteritems(source_bqm.quadratic):
        available_interactions = {(s, t) for s in embedding[u] for t in embedding[v] if s in target_adjacency[t]}

        if not available_interactions:
            raise MissingEdgeError(u, v)

        b = bias / len(available_interactions)

        target_bqm.add_interactions_from((u, v, b) for u, v in available_interactions)

    for chain in itervalues(embedding):

        # in the case where the chain has length 1, there are no chain quadratic biases, but we
        # none-the-less want the chain variables to appear in the target_bqm
        if len(chain) == 1:
            v, = chain
            target_bqm.add_variable(v, 0.0)
            continue

        quadratic_chain_biases = chain_to_quadratic(chain, target_adjacency, chain_strength)
        # this is in spin, but we need to respect the vartype
        if target_bqm.vartype is dimod.SPIN:
            target_bqm.add_interactions_from(quadratic_chain_biases)
        else:
            # do the vartype converstion
            for (u, v), bias in quadratic_chain_biases.items():
                target_bqm.add_interaction(u, v, 4*bias)
                target_bqm.add_variable(u, -2*bias)
                target_bqm.add_variable(v, -2*bias)
                target_bqm.add_offset(bias)

        # add the energy for satisfied chains to the offset
        energy_diff = -sum(itervalues(quadratic_chain_biases))
        target_bqm.add_offset(energy_diff)

    return target_bqm


def embed_ising(source_h, source_J, embedding, target_adjacency, chain_strength=1.0):
    """Embed an Ising problem onto a target graph.

    Args:
        source_h (dict[variable, bias]/list[bias]):
            Linear biases of the Ising problem. If a list, the list's indices are used as
            variable labels.

        source_J (dict[(variable, variable), bias]):
            Quadratic biases of the Ising problem.

        embedding (dict):
            Mapping from source graph to target graph as a dict of form {s: {t, ...}, ...},
            where s is a source-model variable and t is a target-model variable.

        target_adjacency (dict/:obj:`networkx.Graph`):
            Adjacency of the target graph as a dict of form {t: Nt, ...},
            where t is a target-graph variable and Nt is its set of neighbours.

        chain_strength (float, optional):
            Magnitude of the quadratic bias (in SPIN-space) applied between
            variables to form a chain, with the energy penalty of chain breaks
            set to 2 * `chain_strength`.

    Returns:
        tuple: A 2-tuple:

            dict[variable, bias]: Linear biases of the target Ising problem.

            dict[(variable, variable), bias]: Quadratic biases of the target Ising problem.

    Examples:
        This example embeds a triangular Ising problem representing
        a :math:`K_3` clique into a square target graph by mapping variable `c`
        in the source to nodes `2` and `3` in the target.

        >>> import networkx as nx
        ...
        >>> target = nx.cycle_graph(4)
        >>> # Ising problem biases
        >>> h = {'a': 0, 'b': 0, 'c': 0}
        >>> J = {('a', 'b'): 1, ('b', 'c'): 1, ('a', 'c'): 1}
        >>> # Variable c is a chain
        >>> embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}
        >>> # Embed and show the resulting biases
        >>> th, tJ = dwave.embedding.embed_ising(h, J, embedding, target)
        >>> th  # doctest: +SKIP
        {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        >>> tJ  # doctest: +SKIP
        {(0, 1): 1.0, (0, 3): 1.0, (1, 2): 1.0, (2, 3): -1.0}


    See also:
        :func:`.embed_bqm`, :func:`.embed_qubo`

    """
    source_bqm = dimod.BinaryQuadraticModel.from_ising(source_h, source_J)
    target_bqm = embed_bqm(source_bqm, embedding, target_adjacency, chain_strength=chain_strength)
    target_h, target_J, __ = target_bqm.to_ising()
    return target_h, target_J


def embed_qubo(source_Q, embedding, target_adjacency, chain_strength=1.0):
    """Embed a QUBO onto a target graph.

    Args:
        source_Q (dict[(variable, variable), bias]):
            Coefficients of a quadratic unconstrained binary optimization (QUBO) model.

        embedding (dict):
            Mapping from source graph to target graph as a dict of form {s: {t, ...}, ...},
            where s is a source-model variable and t is a target-model variable.

        target_adjacency (dict/:obj:`networkx.Graph`):
            Adjacency of the target graph as a dict of form {t: Nt, ...},
            where t is a target-graph variable and Nt is its set of neighbours.

        chain_strength (float, optional):
            Magnitude of the quadratic bias (in SPIN-space) applied between
            variables to form a chain, with the energy penalty of chain breaks
            set to 2 * `chain_strength`.

    Returns:
        dict[(variable, variable), bias]: Quadratic biases of the target QUBO.

    Examples:
        This example embeds a triangular QUBO representing a :math:`K_3` clique
        into a square target graph by mapping variable `c` in the source to nodes
        `2` and `3` in the target.

        >>> import networkx as nx
        ...
        >>> target = nx.cycle_graph(4)
        >>> # QUBO
        >>> Q = {('a', 'b'): 1, ('b', 'c'): 1, ('a', 'c'): 1}
        >>> # Variable c is a chain
        >>> embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}
        >>> # Embed and show the resulting biases
        >>> tQ = dwave.embedding.embed_qubo(Q, embedding, target)
        >>> tQ  # doctest: +SKIP
        {(0, 1): 1.0,
         (0, 3): 1.0,
         (1, 2): 1.0,
         (2, 3): -4.0,
         (0, 0): 0.0,
         (1, 1): 0.0,
         (2, 2): 2.0,
         (3, 3): 2.0}

    See also:
        :func:`.embed_bqm`, :func:`.embed_ising`

    """
    source_bqm = dimod.BinaryQuadraticModel.from_qubo(source_Q)
    target_bqm = embed_bqm(source_bqm, embedding, target_adjacency, chain_strength=chain_strength)
    target_Q, __ = target_bqm.to_qubo()
    return target_Q


def unembed_sampleset(target_sampleset, embedding, source_bqm,
                      chain_break_method=None, chain_break_fraction=False,
                      return_embedding=False):
    """Unembed a samples set.

    Given samples from a target binary quadratic model (BQM), construct a sample
    set for a source BQM by unembedding.

    Args:
        target_sampleset (:obj:`dimod.SampleSet`):
            Sample set from the target BQM.

        embedding (dict):
            Mapping from source graph to target graph as a dict of form
            {s: {t, ...}, ...}, where s is a source variable and t is a target
            variable.

        source_bqm (:obj:`dimod.BinaryQuadraticModel`):
            Source BQM.

        chain_break_method (function, optional):
            Method used to resolve chain breaks.
            See :mod:`dwave.embedding.chain_breaks`.

        chain_break_fraction (bool, optional, default=False):
            Add a `chain_break_fraction` field to the unembedded :obj:`dimod.SampleSet`
            with the fraction of chains broken before unembedding.

        return_embedding (bool, optional, default=False):
            If True, the embedding is added to :attr:`dimod.SampleSet.info`
            of the returned sample set. Note that if an `embedding` key
            already exists in the sample set then it is overwritten.

    Returns:
        :obj:`.SampleSet`: Sample set in the source BQM.

    Examples:
       This example unembeds from a square target graph samples of a triangular
       source BQM.

        >>> # Triangular binary quadratic model and an embedding
        >>> J = {('a', 'b'): -1, ('b', 'c'): -1, ('a', 'c'): -1}
        >>> bqm = dimod.BinaryQuadraticModel.from_ising({}, J)
        >>> embedding = {'a': [0, 1], 'b': [2], 'c': [3]}
        >>> # Samples from the embedded binary quadratic model
        >>> samples = [{0: -1, 1: -1, 2: -1, 3: -1},  # [0, 1] is unbroken
        ...            {0: -1, 1: +1, 2: +1, 3: +1}]  # [0, 1] is broken
        >>> energies = [-3, 1]
        >>> embedded = dimod.SampleSet.from_samples(samples, dimod.SPIN, energies)
        >>> # Unembed
        >>> samples = dwave.embedding.unembed_sampleset(embedded, embedding, bqm)
        >>> samples.record.sample   # doctest: +SKIP
        array([[-1, -1, -1],
               [ 1,  1,  1]], dtype=int8)

    """

    if chain_break_method is None:
        chain_break_method = majority_vote

    variables = list(source_bqm.variables)  # need this ordered
    try:
        chains = [embedding[v] for v in variables]
    except KeyError:
        raise ValueError("given bqm does not match the embedding")

    record = target_sampleset.record

    unembedded, idxs = chain_break_method(target_sampleset, chains)

    reserved = {'sample', 'energy'}
    vectors = {name: record[name][idxs]
               for name in record.dtype.names if name not in reserved}

    if chain_break_fraction:
        vectors['chain_break_fraction'] = broken_chains(target_sampleset, chains).mean(axis=1)[idxs]

    info = target_sampleset.info.copy()

    if return_embedding:
        embedding_context = dict(embedding=embedding,
                                 chain_break_method=chain_break_method.__name__)
        info.update(embedding_context=embedding_context)

    return dimod.SampleSet.from_samples_bqm((unembedded, variables),
                                            source_bqm,
                                            info=info,
                                            **vectors)
