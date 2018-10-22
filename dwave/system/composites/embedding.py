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
# ================================================================================================
"""
:std:doc:`dimod composites <dimod:reference/samplers>` that map problems
to a structured sampler. A structured sampler, such as :class:`~dwave.system.samplers.DWaveSampler()`,
solves problems that map to a specific graph: the :term:`Chimera` graph for a D-Wave system.

See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_ for explanations
of technical terms in descriptions of Ocean tools.

"""
from warnings import warn

import dimod
import minorminer

__all__ = ['EmbeddingComposite', 'FixedEmbeddingComposite', 'LazyFixedEmbeddingComposite', 'LazyEmbeddingComposite']


class EmbeddingComposite(dimod.ComposedSampler):
    """Composite that maps problems to a structured sampler.

    Enables quick incorporation of the D-Wave system as a sampler by handling minor-embedding
    of the problem into the D-Wave system's :term:`Chimera` graph. Minor-embedding is
    calculated using the heuristic :std:doc:`minorminer <minorminer:index>` library
    each time one of its sampling methods is called.

    Args:
       sampler (:class:`dimod.Sampler`):
            Structured dimod sampler such as a :class:`~dwave.system.samplers.DWaveSampler()`.

    Examples:
       This example submits a simple Ising problem to a D-Wave solver selected by the user's
       default :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`.
       :class:`.EmbeddingComposite` minor-embedds the problem's variables a and b
       to particular qubits on the D-Wave system.

       >>> from dwave.system.samplers import DWaveSampler
       >>> from dwave.system.composites import EmbeddingComposite
       ...
       >>> sampler = EmbeddingComposite(DWaveSampler())
       >>> h = {'a': -1., 'b': 2}
       >>> J = {('a', 'b'): 1.5}
       >>> response = sampler.sample_ising(h, J)
       >>> for sample in response.samples():    # doctest: +SKIP
       ...     print(sample)
       ...
       {'a': 1, 'b': -1}

    See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_ for explanations
    of technical terms in descriptions of Ocean tools.

    """
    def __init__(self, child_sampler):
        if not isinstance(child_sampler, dimod.Structured):
            raise dimod.InvalidComposition("EmbeddingComposite should only be applied to a Structured sampler")
        self._children = [child_sampler]

    @property
    def children(self):
        """list: Children property inherited from :class:`dimod.Composite` class.

        For an instantiated composed sampler, contains the single wrapped structured sampler.

        Examples:
            >>> sampler = EmbeddingComposite(DWaveSampler())
            >>> sampler.children   # doctest: +SKIP
            [<dwave.system.samplers.dwave_sampler.DWaveSampler at 0x7f45b20a8d50>]

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_ for explanations of technical
        terms in descriptions of Ocean tools.

        """
        return self._children

    @property
    def parameters(self):
        """dict[str, list]: Parameters in the form of a dict.

        For an instantiated composed sampler, keys are the keyword parameters accepted by the child sampler
        and parameters added by the composite such as those related to chains.

        Examples:
            This example views parameters of a composed sampler using a D-Wave system selected by
            the user's default
            :std:doc:`D-Wave Cloud Client configuration file. <cloud-client:reference/intro>`

            >>> from dwave.system.samplers import DWaveSampler
            >>> from dwave.system.composites import EmbeddingComposite
            ...
            >>> sampler = EmbeddingComposite(DWaveSampler())
            >>> sampler.parameters   # doctest: +SKIP
            {'anneal_offsets': ['parameters'],
             'anneal_schedule': ['parameters'],
             'annealing_time': ['parameters'],
             'answer_mode': ['parameters'],
             'auto_scale': ['parameters'],
            >>> # Snipped above response for brevity

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_ for explanations of technical
        terms in descriptions of Ocean tools.
        """

        param = self.child.parameters.copy()
        param['chain_strength'] = []
        param['chain_break_fraction'] = []
        return param

    @property
    def properties(self):
        """dict: Properties in the form of a dict.

        For an instantiated composed sampler, contains one key :code:`child_properties` that
        has a copy of the child sampler's properties.

        Examples:
            This example views properties of a composed sampler using a D-Wave system selected by
            the user's default
            :std:doc:`D-Wave Cloud Client configuration file. <cloud-client:reference/intro>`

            >>> from dwave.system.samplers import DWaveSampler
            >>> from dwave.system.composites import EmbeddingComposite
            ...
            >>> sampler = EmbeddingComposite(DWaveSampler())
            >>> sampler.properties   # doctest: +SKIP
            {'child_properties': {u'anneal_offset_ranges': [[-0.2197463755538704,
                0.03821687759418928],
               [-0.2242514597680286, 0.01718456460967399],
               [-0.20860153999435985, 0.05511969218508182],
            >>> # Snipped above response for brevity

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_ for explanations of technical
        terms in descriptions of Ocean tools.

        """
        return {'child_properties': self.child.properties.copy()}

    def sample(self, bqm, chain_strength=1.0, chain_break_fraction=True, **parameters):
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

            chain_break_fraction (bool, optional, default=True):
                If True, the unembedded response contains a ‘chain_break_fraction’ field that
                reports the fraction of chains broken before unembedding.

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :class:`dimod.Response`: A `dimod` :obj:`~dimod.Response` object.

        Examples:
            This example submits an triangle-structured Ising problem to a D-Wave solver, selected
            by the user's default
            :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`,
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
        __, target_edgelist, target_adjacency = child.structure

        # add self-loops to edgelist to handle singleton variables
        source_edgelist = list(bqm.quadratic) + [(v, v) for v in bqm.linear]

        # get the embedding
        embedding = minorminer.find_embedding(source_edgelist, target_edgelist)

        if bqm and not embedding:
            raise ValueError("no embedding found")

        bqm_embedded = dimod.embed_bqm(bqm, embedding, target_adjacency, chain_strength=chain_strength)

        if 'initial_state' in parameters:
            parameters['initial_state'] = _embed_state(embedding, parameters['initial_state'])

        response = child.sample(bqm_embedded, **parameters)

        return dimod.unembed_response(response, embedding, source_bqm=bqm,
                                      chain_break_fraction=chain_break_fraction)


class FixedEmbeddingComposite(dimod.ComposedSampler, dimod.Structured):
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
        :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`,
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

    def __init__(self, child_sampler, embedding=None, source_adjacency=None):
        self._set_child_related_init(child_sampler)
        self._set_graph_related_init(embedding, source_adjacency)

    def _set_child_related_init(self, child_sampler):
        if not isinstance(child_sampler, dimod.Structured):
            raise dimod.InvalidComposition("{} should only be applied to a Structured sampler"
                                           .format(type(self).__name__))

        self.children = [child_sampler]

        self.parameters = parameters = self.child.parameters.copy()
        parameters['chain_strength'] = []
        parameters['chain_break_fraction'] = []

        self.properties = {'child_properties': self.child.properties.copy()}

    def _set_graph_related_init(self, embedding=None, source_adjacency=None):
        # Must have embedding xor source_adjacency
        if (embedding is None) == (source_adjacency is None):
            raise TypeError('_set_graph_related_init() must take either an embedding or a source_adjacency argument,'
                            ' but not both.')

        # Populate embedding and adjacency attributes
        if embedding is not None:
            self.embedding = self.properties['embedding'] = embedding
            self.adjacency = dimod.embedding.target_to_source(self.child.adjacency, embedding)

        else:
            self.adjacency = source_adjacency

            # Find embedding with source_adjacency
            __, target_edgelist, target_adjacency = self.child.structure
            source_edgelist = []

            for k, edges in source_adjacency.items():
                source_edgelist.append((k, k))
                for e in edges:
                    source_edgelist.append((k, e))

            embedding = minorminer.find_embedding(source_edgelist, target_edgelist)
            self.embedding = self.properties['embedding'] = embedding

        # Populate nodelist and edgelist
        try:
            nodelist = sorted(self.adjacency)
            edgelist = sorted(_adjacency_to_edges(self.adjacency))
        except TypeError:
            # python3 does not allow sorting of unlike types, so if nodes have
            # different type names just choose an arbitrary order
            nodelist = list(self.adjacency)
            edgelist = list(_adjacency_to_edges(self.adjacency))
        self.nodelist = nodelist
        self.edgelist = edgelist

    nodelist = None
    """list: Nodes available to the composed sampler.

    Examples:
        >>> fixedsampler = FixedEmbeddingComposite(DWaveSampler(), {'a': [0, 4], 'b': [1, 5], 'c': [2, 6]})
        >>> fixedsampler.nodelist
        ['a', 'b', 'c']
    """

    edgelist = None
    """list: Edges available to the composed sampler.

    Examples:
        >>> fixedsampler = FixedEmbeddingComposite(DWaveSampler(), {'a': [0, 4], 'b': [1, 5], 'c': [2, 6]})
        >>> fixedsampler.edgelist
        [('a', 'b'), ('a', 'c'), ('b', 'c')]
    """

    adjacency = None
    """dict[variable, set]: Adjacency structure for the composed sampler.

    Examples:
        >>> fixedsampler = FixedEmbeddingComposite(DWaveSampler(), {'a': [0, 4], 'b': [1, 5], 'c': [2, 6]})
        >>> fixedsampler.adjacency    # doctest: +SKIP
        {'a': {'b', 'c'}, 'b': {'a', 'c'}, 'c': {'a', 'b'}}

    """

    children = None
    """list: List containing the wrapped sampler.

    See :obj:`.EmbeddingComposite.children` for detailed information.
    """

    parameters = None
    """dict[str, list]: Parameters in the form of a dict.

    See :obj:`.EmbeddingComposite.parameters` for detailed information.
    """

    properties = None
    """dict: Properties in the form of a dict.

    See :obj:`.EmbeddingComposite.properties` for detailed information.
    """

    @dimod.bqm_structured
    def sample(self, bqm, chain_strength=1.0, chain_break_fraction=True, **parameters):
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

            chain_break_fraction (bool, optional, default=True):
                If True, the unembedded response contains a ‘chain_break_fraction’ field
                that reports the fraction of chains broken before unembedding.

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :class:`dimod.Response`: A `dimod` :obj:`~dimod.Response` object.

        Examples:
            This example submits an triangle-structured problem to a D-Wave solver, selected
            by the user's default
            :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`,
            using a specified minor-embedding of the problem’s variables to physical qubits.

            >>> from dwave.system.samplers import DWaveSampler
            >>> from dwave.system.composites import FixedEmbeddingComposite
            >>> import dimod
            ...
            >>> sampler = FixedEmbeddingComposite(DWaveSampler(), {'a': [0, 4], 'b': [1, 5], 'c': [2, 6]})
            >>> response = sampler.sample_ising({}, {'ab': 0.5, 'bc': 0.5, 'ca': 0.5}, chain_strength=2)
            >>> response.first    # doctest: +SKIP
            Sample(sample={'a': 1, 'b': -1, 'c': 1}, energy=-0.5, num_occurrences=1, chain_break_fraction=0.0)

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_
        for explanations of technical terms in descriptions of Ocean tools.

        """

        # solve the problem on the child system
        child = self.child

        # apply the embedding to the given problem to map it to the child sampler
        __, __, target_adjacency = child.structure

        # get the embedding
        embedding = self.embedding

        bqm_embedded = dimod.embed_bqm(bqm, embedding, target_adjacency, chain_strength=chain_strength)

        if 'initial_state' in parameters:
            parameters['initial_state'] = _embed_state(embedding, parameters['initial_state'])

        response = child.sample(bqm_embedded, **parameters)

        return dimod.unembed_response(response, embedding, source_bqm=bqm,
                                      chain_break_fraction=chain_break_fraction)


def _adjacency_to_edges(adjacency):
    """determine from an adjacency the list of edges
    if (u, v) in edges, then (v, u) should not be"""
    edges = set()
    for u in adjacency:
        for v in adjacency[u]:
            try:
                edge = (u, v) if u <= v else (v, u)
            except TypeError:
                # Py3 does not allow sorting of unlike types
                if (v, u) in edges:
                    continue
                edge = (u, v)

            edges.add(edge)
    return edges


def _embed_state(embedding, state):
    """Embed a single state/sample by spreading it's values over the chains in the embedding"""
    return {u: state[v] for v, chain in embedding.items() for u in chain}


class LazyFixedEmbeddingComposite(FixedEmbeddingComposite):
    """Takes an unstructured problem and maps it to a structured problem. This mapping is stored and gets reused
    for all following sample(..) calls.

    Args:
        sampler (dimod.Sampler):
            Structured dimod sampler.
    """
    def __init__(self, child_sampler):
        self._set_child_related_init(child_sampler)
        self.embedding = None

    def sample(self, bqm, chain_strength=1.0, chain_break_fraction=True, **parameters):
        """Sample the binary quadratic model.

        Note: At the initial sample(..) call, it will find a suitable embedding and initialize the remaining attributes
        before sampling the bqm. All following sample(..) calls will reuse that initial embedding.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            chain_strength (float, optional, default=1.0):
                Magnitude of the quadratic bias (in SPIN-space) applied between variables to create
                chains. Note that the energy penalty of chain breaks is 2 * `chain_strength`.

            chain_break_fraction (bool, optional, default=True):
                If True, a ‘chain_break_fraction’ field is added to the unembedded response which report
                what fraction of the chains were broken before unembedding.

            **parameters:
                Parameters for the sampling method, specified by the child sampler.
        Returns:
            :class:`dimod.Response`
        """
        if self.embedding is None:
            # Find embedding
            child = self.child   # Solve the problem on the child system
            __, target_edgelist, target_adjacency = child.structure
            source_edgelist = list(bqm.quadratic) + [(v, v) for v in bqm.linear]  # Add self-loops for single variables
            embedding = minorminer.find_embedding(source_edgelist, target_edgelist)

            # Initialize properties that need embedding
            super(LazyFixedEmbeddingComposite, self)._set_graph_related_init(embedding=embedding)

        return super(LazyFixedEmbeddingComposite, self).sample(bqm, chain_strength=chain_strength,
                                                               chain_break_fraction=chain_break_fraction, **parameters)


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
