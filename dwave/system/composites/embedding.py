# coding: utf-8
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
# =============================================================================
import itertools

from warnings import warn

import dimod
import minorminer

from dwave.embedding import target_to_source, unembed_sampleset, embed_bqm

__all__ = ('EmbeddingComposite',
           'FixedEmbeddingComposite',
           'LazyFixedEmbeddingComposite',
           'LazyEmbeddingComposite',  # deprecated
           'AutoEmbeddingComposite',
           )


class EmbeddingComposite(dimod.ComposedSampler):
    """Composite that maps problems to a structured sampler.

    Enables quick incorporation of the D-Wave system as a sampler by handling
    minor-embedding of the problem into the D-Wave system's :term:`Chimera`
    graph. A new minor-embedding is calculated using the given `find_embedding`
    function each time one of its sampling methods is called.

    Args:
        child_sampler (:class:`dimod.Sampler`):
            A dimod sampler, such as a :obj:`.DWaveSampler`, that has a accepts
            only binary quadratic models of a particular structure.

        find_embedding (function, default=:func:`minorminer.find_embedding`):
            A function `find_embedding(S, T, **kwargs)` where `S` and `T`
            are edgelists. The function can accept additional keyword arguments.

        embedding_parameters (dict, optional):
            If provided, parameters are passed to the embedding method as
            keyword arguments.

    Examples:

       >>> from dwave.system import DWaveSampler, EmbeddingComposite
       ...
       >>> sampler = EmbeddingComposite(DWaveSampler())
       >>> h = {'a': -1., 'b': 2}
       >>> J = {('a', 'b'): 1.5}
       >>> sampleset = sampler.sample_ising(h, J)

    """
    def __init__(self, child_sampler,
                 find_embedding=minorminer.find_embedding,
                 embedding_parameters=None):

        self.children = [child_sampler]

        # keep any embedding parameters around until later, because we might
        # want to overwrite them
        self.embedding_parameters = embedding_parameters or {}
        self.find_embedding = find_embedding

        # set the parameters
        self.parameters = parameters = child_sampler.parameters.copy()
        parameters.update(chain_strength=[],
                          chain_break_method=[],
                          chain_break_fraction=[],
                          embedding_parameters=[],
                          )

        # set the properties
        self.properties = dict(child_properties=child_sampler.properties.copy())

        # track the child's structure. We use a dfs in case intermediate
        # composites are not structured. We could expose multiple different
        # searches but since (as of 14 june 2019) all composites have single
        # children, just doing dfs seems safe for now.
        self.target_structure = dimod.child_structure_dfs(child_sampler)

    parameters = None  # overwritten by init
    """dict[str, list]: Parameters in the form of a dict.

    For an instantiated composed sampler, keys are the keyword parameters
    accepted by the child sampler and parameters added by the composite.
    """

    children = None  # overwritten by init
    """list [child_sampler]: List containing the structured sampler."""

    properties = None  # overwritten by init
    """dict: Properties in the form of a dict.

    Contains the properties of the child sampler.
    """

    def sample(self, bqm, chain_strength=1.0,
               chain_break_method=None,
               chain_break_fraction=True,
               embedding_parameters=None,
               **parameters):
        """Sample from the provided binary quadratic model.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            chain_strength (float, optional, default=1.0):
                Magnitude of the quadratic bias (in SPIN-space) applied between
                variables to create chains. The energy penalty of chain breaks
                is 2 * `chain_strength`.

            chain_break_method (function, optional):
                Method used to resolve chain breaks during sample unembedding.
                See :func:`~dwave.embedding.unembed_sampleset`.

            chain_break_fraction (bool, optional, default=True):
                If True, the unembedded response contains a
                ‘chain_break_fraction’ field that reports the fraction of chains
                broken before unembedding.

            embedding_parameters (dict, optional):
                If provided, parameters are passed to the embedding method as
                keyword arguments. Overrides any `embedding_parameters` passed
                to the constructor.

            **parameters:
                Parameters for the sampling method, specified by the child
                sampler.

        Returns:
            :obj:`dimod.SampleSet`

        """

        # solve the problem on the child system
        child = self.child

        # apply the embedding to the given problem to map it to the child sampler
        __, target_edgelist, target_adjacency = self.target_structure

        # add self-loops to edgelist to handle singleton variables
        source_edgelist = list(bqm.quadratic) + [(v, v) for v in bqm.linear]

        # get the embedding
        if embedding_parameters is None:
            embedding_parameters = self.embedding_parameters
        else:
            # we want the parameters provided to the constructor, updated with
            # the ones provided to the sample method. To avoid the extra copy
            # we do an update, avoiding the keys that would overwrite the
            # sample-level embedding parameters
            embedding_parameters.update((key, val)
                                        for key, val in self.embedding_parameters
                                        if key not in embedding_parameters)

        embedding = self.find_embedding(source_edgelist, target_edgelist,
                                        **embedding_parameters)

        if bqm and not embedding:
            raise ValueError("no embedding found")

        bqm_embedded = embed_bqm(bqm, embedding, target_adjacency,
                                 chain_strength=chain_strength,
                                 smear_vartype=dimod.SPIN)

        if 'initial_state' in parameters:
            # if initial_state was provided in terms of the source BQM, we want
            # to modify it to now provide the initial state for the target BQM.
            # we do this by spreading the initial state values over the
            # chains
            state = parameters['initial_state']
            parameters['initial_state'] = {u: state[v]
                                           for v, chain in embedding.items()
                                           for u in chain}

        response = child.sample(bqm_embedded, **parameters)

        return unembed_sampleset(response, embedding, source_bqm=bqm,
                                 chain_break_method=chain_break_method,
                                 chain_break_fraction=chain_break_fraction)


class LazyFixedEmbeddingComposite(EmbeddingComposite, dimod.Structured):
    """Fixes itself to the structure of the first problem it samples.

    Args:
        sampler (dimod.Sampler):
            Structured dimod sampler.

        find_embedding (function, default=:func:`minorminer.find_embedding`):
            A function `find_embedding(S, T, **kwargs)` where `S` and `T`
            are edgelists. The function can accept additional keyword arguments.
            The function is used to find the embedding for the first problem
            solved.

        embedding_parameters (dict, optional):
            If provided, parameters are passed to the embedding method as keyword
            arguments.

    Examples:

        >>> from dwave.system import LazyFixedEmbeddingComposite, DWaveSampler
        ...
        >>> sampler = LazyFixedEmbeddingComposite(DWaveSampler())
        >>> sampler.nodelist is None  # no structure yet
        True
        >>> __ = sampler.sample_ising({}, {('a', 'b'): -1})
        >>> sampler.nodelist  # has structure based on given problem
        ['a', 'b']

    """

    @property
    def nodelist(self):
        """list: Nodes available to the composed sampler."""
        try:
            return self._nodelist
        except AttributeError:
            pass

        if self.adjacency is None:
            return None

        self._nodelist = nodelist = list(self.adjacency)

        # makes it a lot easier for the user if the list can be sorted, so we
        # try
        try:
            nodelist.sort()
        except TypeError:
            # python3 cannot sort unlike types
            pass

        return nodelist

    @property
    def edgelist(self):
        """list: Edges available to the composed sampler."""
        try:
            return self._edgelist
        except AttributeError:
            pass

        adj = self.adjacency

        if adj is None:
            return None

        # remove duplicates by putting into a set
        edges = set()
        for u in adj:
            for v in adj[u]:
                try:
                    edge = (u, v) if u <= v else (v, u)
                except TypeError:
                    # Py3 does not allow sorting of unlike types
                    if (v, u) in edges:
                        continue
                    edge = (u, v)

                edges.add(edge)

        self._edgelist = edgelist = list(edges)

        # makes it a lot easier for the user if the list can be sorted, so we
        # try
        try:
            edgelist.sort()
        except TypeError:
            # python3 cannot sort unlike types
            pass

        return edgelist

    @property
    def adjacency(self):
        """dict[variable, set]: Adjacency structure for the composed sampler."""
        try:
            return self._adjacency
        except AttributeError:
            pass

        if self.embedding is None:
            return None

        self._adjacency = adj = target_to_source(self.target_structure.adjacency,
                                                 self.embedding)

        return adj

    embedding = None
    """The embedding used to map bqms to the child sampler."""

    def _fix_embedding(self, embedding):
        # save the embedding and overwrite the find_embedding function
        self.embedding = embedding
        self.properties.update(embedding=embedding)

        def find_embedding(S, T):
            return embedding

        self.find_embedding = find_embedding

    def sample(self, bqm, **parameters):
        """Sample the binary quadratic model.

        Note: At the initial sample(..) call, it will find a suitable embedding
        and initialize the remaining attributes before sampling the bqm. All
        following sample(..) calls will reuse that initial embedding.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            chain_strength (float, optional, default=1.0):
                Magnitude of the quadratic bias (in SPIN-space) applied between
                variables to create chains. The energy penalty of chain breaks
                is 2 * `chain_strength`.

            chain_break_method (function, optional):
                Method used to resolve chain breaks during sample unembedding.
                See :func:`~dwave.embedding.unembed_sampleset`.

            chain_break_fraction (bool, optional, default=True):
                If True, the unembedded response contains a
                ‘chain_break_fraction’ field that reports the fraction of chains
                broken before unembedding.

            embedding_parameters (dict, optional):
                If provided, parameters are passed to the embedding method as
                keyword arguments. Overrides any `embedding_parameters` passed
                to the constructor. Only used on the first call.

            **parameters:
                Parameters for the sampling method, specified by the child
                sampler.

        Returns:
            :obj:`dimod.SampleSet`

        """
        if self.embedding is None:
            # get an embedding using the current find_embedding function
            embedding_parameters = parameters.pop('embedding_parameters', None)

            if embedding_parameters is None:
                embedding_parameters = self.embedding_parameters
            else:
                # update the base parameters with the new ones provided
                embedding_parameters.update((key, val)
                                            for key, val in self.embedding_parameters
                                            if key not in embedding_parameters)

            source_edgelist = list(itertools.chain(bqm.quadratic,
                                                   ((v, v) for v in bqm.linear)))

            target_edgelist = self.target_structure.edgelist

            embedding = self.find_embedding(source_edgelist, target_edgelist,
                                            **embedding_parameters)

            self._fix_embedding(embedding)

        return super(LazyFixedEmbeddingComposite, self).sample(bqm, **parameters)


class FixedEmbeddingComposite(LazyFixedEmbeddingComposite):
    """Uses a specified minor-embedding to map problems to a structured sampler.

    Args:
        sampler (dimod.Sampler):
            Structured dimod sampler such as a D-Wave system.

        embedding (dict[hashable, iterable], optional):
            Mapping from a source graph to the specified sampler’s graph (the
            target graph).

        source_adjacency (dict[hashable, iterable]):
            Deprecated. Dictionary to describe source graph. Ex. `{node:
            {node neighbours}}`.

        kwargs:
            See docs for :class:`.EmbeddingComposite` for additional keyword
            arguments. Note that `find_embedding` and `embedding_parameters`
            keyword arguments are ignored.

    Examples:

        >>> from dwave.system.samplers import DWaveSampler
        >>> from dwave.system.composites import FixedEmbeddingComposite
        ...
        >>> embedding = {'a': [0, 4], 'b': [1, 5], 'c': [2, 6]}
        >>> sampler = FixedEmbeddingComposite(DWaveSampler(), embedding)
        >>> sampler.nodelist
        ['a', 'b', 'c']
        >>> sampler.edgelist
        [('a', 'b'), ('a', 'c'), ('b', 'c')]
        >>> sampleset = sampler.sample_ising({'a': .5, 'c': 0}, {('a', 'c'): -1})

    """
    def __init__(self, child_sampler, embedding=None, source_adjacency=None,
                 **kwargs):
        super(FixedEmbeddingComposite, self).__init__(child_sampler, **kwargs)

        # dev note: this entire block is to support a deprecated feature and can
        # be removed in the next major release
        if embedding is None:

            warn(("The source_adjacency parameter is deprecated"),
                 DeprecationWarning)

            if source_adjacency is None:
                raise TypeError("either embedding or source_adjacency must be "
                                "provided")

            source_edgelist = [(u, v) for u in source_adjacency for v in source_adjacency[u]]

            embedding = self.find_embedding(source_edgelist,
                                            self.target_structure.edgelist)

        self._fix_embedding(embedding)


class LazyEmbeddingComposite(LazyFixedEmbeddingComposite):
    """Deprecated Class. 'LazyEmbeddingComposite' has been deprecated and renamed to 'LazyFixedEmbeddingComposite'.

    Takes an unstructured problem and maps it to a structured problem. This mapping is stored and gets reused
    for all following sample(..) calls.

    Args:
        sampler (dimod.Sampler):
            Structured dimod sampler.
    """
    def __init__(self, child_sampler):
        super(LazyEmbeddingComposite, self).__init__(child_sampler)
        warn("'LazyEmbeddingComposite' has been renamed to 'LazyFixedEmbeddingComposite'.", DeprecationWarning)


class AutoEmbeddingComposite(EmbeddingComposite):
    """Composite that maps problems to a structured sampler.

    Differs from :class:`.EmbeddingComposite` by not embedding binary quadratic
    models that already match the child sampler.

    Args:
        sampler (:class:`dimod.Sampler`):
            Structured dimod sampler, such as a
            :obj:`~dwave.system.samplers.DWaveSampler()`.

        find_embedding (function, default=:func:`minorminer.find_embedding`):
            A function `find_embedding(S, T, **kwargs)` where `S` and `T`
            are edgelists. The function can accept additional keyword arguments.

        embedding_parameters (dict, optional):
            If provided, parameters are passed to the embedding method as
            keyword arguments.

    """
    def __init__(self, child_sampler,
                 find_embedding=minorminer.find_embedding,
                 **kwargs):

        def auto_find_embedding(S, *args, **kw):
            # check if the problem already matches the target, in which case
            # don't embed
            adj = self.target_structure.adjacency

            if all(u in adj.get(v, []) if u != v else u in adj for u, v in S):
                # identity embedding
                return {v: [v] for pair in S for v in pair}

            return find_embedding(S, *args, **kw)

        super(AutoEmbeddingComposite, self).__init__(child_sampler,
                                                     find_embedding=auto_find_embedding,
                                                     **kwargs)
