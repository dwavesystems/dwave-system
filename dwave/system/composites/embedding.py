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
A :std:doc:`dimod composite <dimod:reference/samplers>` that maps unstructured problems
to a structured sampler.

A structured sampler can only solve problems that map to a specific graph: the
D-Wave system's architecture is represented by a :std:doc:`Chimera <system:reference/intro>` graph.

The :class:`.EmbeddingComposite` uses the :std:doc:`minorminer <minorminer:index>` library
to map unstructured problems to a structured sampler such as a D-Wave system.

See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_ for explanations
of technical terms in descriptions of Ocean tools.

"""
import dimod
import minorminer

__all__ = ['EmbeddingComposite', 'FixedEmbeddingComposite', 'LazyEmbeddingComposite']


class EmbeddingComposite(dimod.ComposedSampler):
    """Composite to map unstructured problems to a structured sampler.

    Inherits from :class:`dimod.ComposedSampler`.

    Enables quick incorporation of the D-Wave system as a sampler in the D-Wave Ocean
    software stack by handling the minor-embedding of the problem into the D-Wave
    system's Chimera graph.

    Args:
       sampler (:class:`dimod.Sampler`):
            Structured dimod sampler.

    Examples:
       This example uses :class:`.EmbeddingComposite` to instantiate a composed sampler
       that submits a simple Ising problem to a D-Wave solver selected by the user's
       default :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`.
       The composed sampler handles
       minor-embedding of the problem's two generic variables, a and b, to physical
       qubits on the solver.

       >>> from dwave.system.samplers import DWaveSampler
       >>> from dwave.system.composites import EmbeddingComposite
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
            This example instantiates a composed sampler using a D-Wave solver selected by
            the user's default
            :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`
            and views the solver's parameters.

            >>> from dwave.system.samplers import DWaveSampler
            >>> from dwave.system.composites import EmbeddingComposite
            >>> sampler = EmbeddingComposite(DWaveSampler())
            >>> sampler.children   # doctest: +SKIP
            [<dwave.system.samplers.dwave_sampler.DWaveSampler at 0x7f45b20a8d50>]

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_ for explanations of technical terms in descriptions of Ocean tools.

        """
        return self._children

    @property
    def parameters(self):
        """dict[str, list]: Parameters in the form of a dict.

        For an instantiated composed sampler, keys are the keyword parameters accepted by the child sampler.

        Examples:
            This example instantiates a composed sampler using a D-Wave solver selected by
            the user's default
            :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`
            and views the solver's parameters.

            >>> from dwave.system.samplers import DWaveSampler
            >>> from dwave.system.composites import EmbeddingComposite
            >>> sampler = EmbeddingComposite(DWaveSampler())
            >>> sampler.parameters   # doctest: +SKIP
            {'anneal_offsets': ['parameters'],
             'anneal_schedule': ['parameters'],
             'annealing_time': ['parameters'],
             'answer_mode': ['parameters'],
             'auto_scale': ['parameters'],
            >>> # Snipped above response for brevity

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_ for explanations of technical terms in descriptions of Ocean tools.

        """
        # does not add or remove any parameters
        param = self.child.parameters.copy()
        param['chain_strength'] = []
        param['chain_break_fraction'] = []
        return param

    @property
    def properties(self):
        """dict: Properties in the form of a dict.

        For an instantiated composed sampler, contains one key :code:`'child_properties'` that
        has a copy of the child sampler's properties.

        Examples:
            This example instantiates a composed sampler using a D-Wave solver selected by
            the user's default
            :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`
            and views the solver's properties.

            >>> from dwave.system.samplers import DWaveSampler
            >>> from dwave.system.composites import EmbeddingComposite
            >>> sampler = EmbeddingComposite(DWaveSampler())
            >>> sampler.properties   # doctest: +SKIP
            {'child_properties': {u'anneal_offset_ranges': [[-0.2197463755538704,
                0.03821687759418928],
               [-0.2242514597680286, 0.01718456460967399],
               [-0.20860153999435985, 0.05511969218508182],
            >>> # Snipped above response for brevity

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_ for explanations of technical terms in descriptions of Ocean tools.

        """
        return {'child_properties': self.child.properties.copy()}

    def sample(self, bqm, chain_strength=1.0, chain_break_fraction=True, **parameters):
        """Sample from the provided binary quadratic model.

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

        Examples:
            This example uses :class:`.EmbeddingComposite` to instantiate a composed sampler
            that submits an unstructured Ising problem to a D-Wave solver, selected by the user's
            default
            :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`,
            while minor-embedding the problem's variables to physical qubits on the solver.

            >>> from dwave.system.samplers import DWaveSampler
            >>> from dwave.system.composites import EmbeddingComposite
            >>> import dimod
            >>> sampler = EmbeddingComposite(DWaveSampler())
            >>> h = {1: 1, 2: 2, 3: 3, 4: 4}
            >>> J = {(1, 2): 12, (1, 3): 13, (1, 4): 14,
            ...      (2, 3): 23, (2, 4): 24,
            ...      (3, 4): 34}
            >>> bqm = dimod.BinaryQuadraticModel.from_ising(h, J)
            >>> response = sampler.sample(bqm)
            >>> for sample in response.samples():    # doctest: +SKIP
            ...     print(sample)
            ...
            {1: -1, 2: 1, 3: 1, 4: -1}

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
    """Composite to alter the structure of a child sampler via an embedding.

    Inherits from :class:`dimod.ComposedSampler` and :class:`dimod.Structured`.

    Args:
        sampler (dimod.Sampler):
            Structured dimod sampler.

        embedding (dict[hashable, iterable]):
            Mapping from a source graph to the specified sampler’s graph (the target graph).

    Examples:

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
        if not isinstance(child_sampler, dimod.Structured):
            raise dimod.InvalidComposition("EmbeddingComposite should only be applied to a Structured sampler")

        self.children = [child_sampler]
        self._set_embedding_init(embedding)

    def _set_embedding_init(self, embedding):
        # Derive the structure of our composed sampler from the target graph and the embedding
        source_adjacency = dimod.embedding.target_to_source(self.child.adjacency, embedding)
        try:
            nodelist = sorted(source_adjacency)
            edgelist = sorted(_adjacency_to_edges(source_adjacency))
        except TypeError:
            # python3 does not allow sorting of unlike types, so if nodes have
            # different type names just choose an arbitrary order
            nodelist = list(source_adjacency)
            edgelist = list(_adjacency_to_edges(source_adjacency))
        self.nodelist = nodelist
        self.edgelist = edgelist
        self.adjacency = source_adjacency

        self.parameters = parameters = self.child.parameters.copy()
        parameters['chain_strength'] = []
        parameters['chain_break_fraction'] = []

        self.properties = {'child_properties': self.child.properties.copy()}

        self.embedding = self.properties['embedding'] = embedding

    nodelist = None
    """list:
           Nodes available to the composed sampler.
    """

    edgelist = None
    """list:
           Edges available to the composed sampler.
    """

    adjacency = None
    """dict[variable, set]:
           Adjacency structure for the composed sampler.

    """

    children = None
    """list: List containing the wrapped sampler."""

    parameters = None
    """dict[str, list]: Parameters in the form of a dict.

    The same as the child sampler with the addition of 'chain_strength'
    """

    properties = None
    """dict: Properties in the form of a dict.

    For an instantiated composed sampler, :code:`'child_properties'` has a copy of the child
    sampler's properties and :code:`'embedding'` contains the fixed embedding.

    """

    @dimod.bqm_structured
    def sample(self, bqm, chain_strength=1.0, chain_break_fraction=True, **parameters):
        """Sample from the provided binary quadratic model.

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

        Examples:
            This example uses :class:`.FixedEmbeddingComposite` to instantiate a composed sampler
            that submits an unstructured Ising problem to a D-Wave solver, selected by the user's
            default
            :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`,
            while minor-embedding the problem's variables to physical qubits on the solver.

            >>> from dwave.system.samplers import DWaveSampler
            >>> from dwave.system.composites import FixedEmbeddingComposite
            >>> import dimod
            >>> sampler = FixedEmbeddingComposite(DWaveSampler(), {'a': [0, 4], 'b': [1, 5], 'c': [2, 6]})
            >>> resp = sampler.sample_ising({'a': .5, 'c': 0}, {('a', 'c'): -1})

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


class LazyEmbeddingComposite(FixedEmbeddingComposite):
    """ Takes an unstructured problem and maps it to a structured problem. This mapping is stored and gets reused
    for all following sample(..) calls.

    Args:
        sampler (dimod.Sampler):
            Structured dimod sampler.
    """
    def __init__(self, child_sampler):
        if not isinstance(child_sampler, dimod.Structured):
            raise dimod.InvalidComposition('LazyEmbeddingComposite should only be applied to a Structured sampler')

        self.children = [child_sampler]
        self.embedding = None

    def sample(self, bqm, chain_strength=1.0, chain_break_fraction=True, **parameters):
        """ Sample the binary quadratic model.

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
            super(LazyEmbeddingComposite, self)._set_embedding_init(embedding)

        return super(LazyEmbeddingComposite, self).sample(bqm, chain_strength=chain_strength, chain_break_fraction=chain_break_fraction, **parameters)
