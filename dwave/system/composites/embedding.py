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
           'ParamEmbeddingComposite',
           'LazyEmbeddingComposite',  # deprecated
           )


class EmbeddingComposite(dimod.ComposedSampler):
    """Composite that maps problems to a structured sampler.

    Enables quick incorporation of the D-Wave system as a sampler by handling
    minor-embedding of the problem into the D-Wave system's :term:`Chimera`
    graph. A new minor-embedding is calculated using the given `find_embedding`
    function each time one of its sampling methods is called.

    Args:
        sampler (:class:`dimod.Sampler`):
            Structured dimod sampler, such as a
            :obj:`~dwave.system.samplers.DWaveSampler()`.

        find_embedding (function, default=:func:`minorminer.find_embedding`):
            A function `find_embedding(S, T, **kwargs)` where `S` and `T`
            are edgelists. The function can accept addition keyword arguments.

        embedding_parameters (dict, optional):
            If provided, parameter are passed to the embedding method as keyword
            arguments.

    Examples:
       This example submits a simple Ising problem to a D-Wave solver selected by the user's
       default :std:doc:`D-Wave Cloud Client configuration file <cloud-client:intro>`.
       :class:`.EmbeddingComposite` maps the problem's variables 'a' and 'b'
       to qubits on the D-Wave system.

       >>> from dwave.system import DWaveSampler, EmbeddingComposite
       ...
       >>> sampler = EmbeddingComposite(DWaveSampler())
       >>> h = {'a': -1., 'b': 2}
       >>> J = {('a', 'b'): 1.5}
       >>> response = sampler.sample_ising(h, J)
       >>> for sample in response.samples():    # doctest: +SKIP
       ...     print(sample)
       {'a': 1, 'b': -1}

    See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_ for explanations
    of technical terms in descriptions of Ocean tools.

    """
    def __init__(self, child_sampler,
                 find_embedding=minorminer.find_embedding,
                 embedding_parameters=None):

        if not isinstance(child_sampler, dimod.Structured):
            raise dimod.InvalidComposition("EmbeddingComposite should only be applied to a Structured sampler")
        self.children = [child_sampler]

        # keep any embedding parameters around until later, because we might
        # want to overwrite them
        if embedding_parameters is None:
            self.embedding_parameters = {}
        else:
            self.embedding_parameters = embedding_parameters
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

        # track the child's structure
        self.target_structure = child_sampler.structure

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

        Also set parameters for handling a chain, the set of vertices in a target graph that
        represents a source-graph vertex; when a D-Wave system is the sampler, it is a set
        of qubits that together represent a variable of the binary quadratic model being
        minor-embedded.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            chain_strength (float, optional, default=1.0):
                Magnitude of the quadratic bias (in SPIN-space) applied between variables to create
                chains. The energy penalty of chain breaks is 2 * `chain_strength`.

            chain_break_method (function, optional, default=dwave.embedding.majority_vote):
                Method used to resolve chain breaks during sample unembedding.
                See :mod:`dwave.embedding.chain_breaks`.

            chain_break_fraction (bool, optional, default=True):
                If True, the unembedded response contains a ‘chain_break_fraction’ field that
                reports the fraction of chains broken before unembedding.

            embedding_parameters (dict, optional):
                If provided, parameter are passed to the embedding method as
                keyword arguments. Overrides any `embedding_parameters` passed
                to the constructor.

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :class:`dimod.SampleSet`: A `dimod` :obj:`~dimod.SampleSet` object.

        Examples:
            This example submits an triangle-structured Ising problem to a D-Wave solver, selected
            by the user's default
            :std:doc:`D-Wave Cloud Client configuration file <cloud-client:intro>`,
            by minor-embedding the problem's variables to physical qubits.

            >>> from dwave.system.samplers import DWaveSampler
            >>> from dwave.system.composites import EmbeddingComposite
            >>> import dimod
            ...
            >>> sampler = EmbeddingComposite(DWaveSampler())
            >>> bqm = dimod.BinaryQuadraticModel.from_ising({}, {'ab': 0.5, 'bc': 0.5, 'ca': 0.5})
            >>> response = sampler.sample(bqm, chain_strength=2)
            >>> response.first:    # doctest: +SKIP
            Sample(sample={'a': -1, 'b': 1, 'c': 1}, energy=-0.5,
                   num_occurrences=1, chain_break_fraction=0.0)

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_
        for explanations of technical terms in descriptions of Ocean tools.
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
            # update the base parameters with the new ones provided
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
            parameters['initial_state'] = _embed_state(embedding, parameters['initial_state'])

        response = child.sample(bqm_embedded, **parameters)

        return unembed_sampleset(response, embedding, source_bqm=bqm,
                                 chain_break_method=chain_break_method,
                                 chain_break_fraction=chain_break_fraction)


class LazyFixedEmbeddingComposite(EmbeddingComposite, dimod.Structured):
    """Takes an unstructured problem and maps it to a structured problem. This mapping is stored and gets reused
    for all following sample(..) calls.

    Args:
        sampler (dimod.Sampler):
            Structured dimod sampler.

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
            raise ValueError("no embedding has been set, so structure cannot "
                             "be determined")

        self._adjacency = adj = target_to_source(self.target_structure.adjacency,
                                                 self.embedding)

        return adj

    embedding = None
    """todo"""

    def _fix_embedding(self, embedding):
        # save the embedding and overwrite the find_embedding function
        self.embedding = embedding
        self.properties.update(embedding=embedding)

        def find_embedding(S, T):
            return embedding

        self.find_embedding = find_embedding

    def sample(self, bqm, **parameters):
        """Sample the binary quadratic model.

        Note: At the initial sample(..) call, it will find a suitable embedding and initialize the remaining attributes
        before sampling the bqm. All following sample(..) calls will reuse that initial embedding.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            chain_strength (float, optional, default=1.0):
                Magnitude of the quadratic bias (in SPIN-space) applied between variables to create
                chains. Note that the energy penalty of chain breaks is 2 * `chain_strength`.

            chain_break_method (function, optional, default=dwave.embedding.majority_vote):
                Method used to resolve chain breaks during sample unembedding.
                See :mod:`dwave.embedding.chain_breaks`.

            chain_break_fraction (bool, optional, default=True):
                If True, a ‘chain_break_fraction’ field is added to the unembedded response which report
                what fraction of the chains were broken before unembedding.

            **parameters:
                Parameters for the sampling method, specified by the child sampler.
        Returns:
            :class:`dimod.SampleSet`
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
    """Composite that uses a specified minor-embedding to map problems to a structured sampler.

    Enables incorporation of the D-Wave system as a sampler, given a precalculated minor-embedding.

    Args:
        sampler (dimod.Sampler):
            Structured dimod sampler such as a D-Wave system.

        embedding (dict[hashable, iterable]):
            Mapping from a source graph to the specified sampler’s graph (the target graph).

        source_adjacency (dict[hashable, iterable]):
            Dictionary to describe source graph. Ex. {node: {node neighbours}}

    Examples:
        This example submits an triangle-structured Ising problem to a D-Wave solver, selected
        by the user's default
        :std:doc:`D-Wave Cloud Client configuration file <cloud-client:intro>`,
        using a given minor-embedding of the problem's variables to physical qubits.

        >>> from dwave.system.samplers import DWaveSampler
        >>> from dwave.system.composites import FixedEmbeddingComposite
        ...
        >>> sampler = FixedEmbeddingComposite(DWaveSampler(), {'a': [0, 4], 'b': [1, 5], 'c': [2, 6]})
        >>> sampler.nodelist
        ['a', 'b', 'c']
        >>> sampler.edgelist
        [('a', 'b'), ('a', 'c'), ('b', 'c')]
        >>> resp = sampler.sample_ising({'a': .5, 'c': 0}, {('a', 'c'): -1})


    """
    def __init__(self, child_sampler, embedding):
        super(FixedEmbeddingComposite, self).__init__(child_sampler)
        self._fix_embedding(embedding)


def _embed_state(embedding, state):
    """Embed a single state/sample by spreading it's values over the chains in the embedding"""
    return {u: state[v] for v, chain in embedding.items() for u in chain}


class ParamEmbeddingComposite(dimod.ComposedSampler):
    """Composite that maps problems to a structured sampler using a
    parameterized embedding method.

    Args:
       sampler (dimod.Sampler)
            Structured dimod sampler.

       embedding_method (object, optional, default=minorminer)
            Any object with a find_embedding(S,T,**params) method.
            Where S and T are edgelists or NetworkX Graphs.

       embedding_parameters (dict, optional, default={}):
            Parameter dictionary to be passed to the embedding method.

    """
    def __init__(self, child_sampler, embedding_method=minorminer,
                 **embedding_parameters):
        if not isinstance(child_sampler, dimod.Structured):
            raise dimod.InvalidComposition("ParamEmbeddingComposite should "
                                           "only be applied to a "
                                           "Structured sampler")
        self._children = [child_sampler]
        self._embedding_method = embedding_method
        self._embedding_parameters = embedding_parameters

        self.embedding = None
        self.child_response = None

    @property
    def children(self):
        """list [child_sampler]: List containing the structured sampler."""
        return self._children

    @property
    def parameters(self):
        """dict[str, list]: Parameters in the form of a dict.

        #TODO: Ideally, the `embedding_method` object should also have a
        `parameters` attribute.
        """

        param = self.child.parameters.copy()
        param['force_embed'] = []
        param['chain_strength'] = []
        param['chain_break_method'] =  []
        param['chain_break_fraction'] = []
        # Ideally
        # param['embedding_parameters'] = self._embedding_method.parameters.copy()

        return param

    @property
    def properties(self):
        """dict: Properties in the form of a dict.
        """
        return {'child_properties': self.child.properties.copy()}

    def get_embedding(self, bqm, target_edgelist=None,
                      force_embed=False,
                      embedding_method=None,
                      **embedding_parameters):
        """Retrieve or create a minor-embedding from BinaryQuadraticModel

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            target_edgelist (list, optional, default=<Child Structure>):
                An iterable of label pairs representing the edges in the target graph.

            force_embed (bool, optional, default=False):
                If the sampler has an embedding return it. Otherwise, embed problem.

            **embedding_parameters:
                Parameters for the embedding method.

        Returns:
            embedding (dict):
                Dictionary that maps labels in S_edgelist to lists of labels in the
                graph of the structured sampler.
        """
        if not isinstance(bqm, dimod.BinaryQuadraticModel):
            raise ValueError("get_embedding() only takes "
                                   "dimod.BinaryQuadraticModel as input")

        embedding = self.embedding

        if not embedding_parameters:
            embedding_parameters = self._embedding_parameters

        if embedding_method is None:
            embedding_method = self._embedding_method

        if target_edgelist is None:
            _, target_edgelist, _ = self.child.structure

        # add self-loops to edgelist to handle singleton variables
        source_edgelist = list(bqm.quadratic) + [(v, v) for v in bqm.linear]

        if force_embed or not embedding:
            embedding = embedding_method.find_embedding(source_edgelist,
                                                        target_edgelist,
                                                        **embedding_parameters)
        if bqm and not embedding:
            raise ValueError("no embedding found")

        self.embedding = self.properties['embedding'] = embedding

        return embedding

    def sample(self, bqm, chain_strength=1.0, chain_break_method=None,
               chain_break_fraction=True, force_embed=False, **parameters):
        """Sample from the provided binary quadratic model.

        Also set parameters for handling a chain, the set of vertices in a target graph that
        represents a source-graph vertex; when a D-Wave system is the sampler, it is a set
        of qubits that together represent a variable of the binary quadratic model being
        minor-embedded.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            chain_strength (float, optional, default=1.0):
                Magnitude of the quadratic bias (in SPIN-space) applied between variables to create
                chains. The energy penalty of chain breaks is 2 * `chain_strength`.

            chain_break_method (function, optional, default=dwave.embedding.majority_vote):
                Method used to resolve chain breaks during sample unembedding.
                See :mod:`dwave.embedding.chain_breaks`.

            chain_break_fraction (bool, optional, default=True):
                If True, the unembedded response contains a ‘chain_break_fraction’ field that
                reports the fraction of chains broken before unembedding.

            force_embed (bool, optional, default=False):
                If True, regardless of `embedding`, a new embedding is obtained.

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :class:`dimod.SampleSet`: A `dimod` :obj:`~dimod.SampleSet` object.

        """

        # solve the problem on the child system
        child = self.child

        # apply the embedding to the given problem to map it to the child sampler
        __, target_edgelist, target_adjacency = child.structure

        # pass the corresponding parameters
        embedding_parameters = self._embedding_parameters

        # get the embedding
        embedding = self.get_embedding(bqm, target_edgelist=target_edgelist,
                                       force_embed=force_embed,
                                       **embedding_parameters)
        if bqm and not embedding:
            raise ValueError("no embedding found")

        bqm_embedded = embed_bqm(bqm, embedding, target_adjacency,
                                 chain_strength=chain_strength,
                                 smear_vartype=dimod.SPIN)

        if 'initial_state' in parameters:
            parameters['initial_state'] = _embed_state(embedding, parameters['initial_state'])

        response = child.sample(bqm_embedded, **parameters)

        # Store embedded response
        self.child_response = response

        return unembed_sampleset(response, embedding, source_bqm=bqm,
                                 chain_break_method=chain_break_method,
                                 chain_break_fraction=chain_break_fraction)


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
