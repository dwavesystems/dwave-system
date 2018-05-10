# coding: utf-8
"""
A dimod composite_ that maps unstructured problems to a structured_ sampler.

A structured_ sampler can only solve problems that map to a specific graph: the
D-Wave system's architecture is represented by a Chimera_ graph.

The :class:`.EmbeddingComposite` uses the minorminer_ library to map unstructured
problems to a structured sampler such as a D-Wave system.

.. _composite: http://dimod.readthedocs.io/en/latest/reference/samplers.html
.. _minorminer: https://github.com/dwavesystems/minorminer
.. _structured: http://dimod.readthedocs.io/en/latest/reference/samplers.html#module-dimod.core.structured
.. _Chimera: http://dwave-system.readthedocs.io/en/latest/reference/intro.html#chimera

"""
import dimod
import minorminer

__all__ = ['EmbeddingComposite', 'FixedEmbeddingComposite']


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
       default D-Wave Cloud Client configuration_ file. The composed sampler handles
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

    .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config

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
            the user's default D-Wave Cloud Client configuration_ file and views the
            solver's parameters.

            >>> from dwave.system.samplers import DWaveSampler
            >>> from dwave.system.composites import EmbeddingComposite
            >>> sampler = EmbeddingComposite(DWaveSampler())
            >>> sampler.children   # doctest: +SKIP
            [<dwave.system.samplers.dwave_sampler.DWaveSampler at 0x7f45b20a8d50>]

        .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config

        """
        return self._children

    @property
    def parameters(self):
        """dict[str, list]: Parameters in the form of a dict.

        For an instantiated composed sampler, keys are the keyword parameters accepted by the child sampler.

        Examples:
            This example instantiates a composed sampler using a D-Wave solver selected by
            the user's default D-Wave Cloud Client configuration_ file and views the
            solver's parameters.

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

        .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config

        """
        # does not add or remove any parameters
        param = self.child.parameters.copy()
        param['chain_strength'] = []
        return param

    @property
    def properties(self):
        """dict: Properties in the form of a dict.

        For an instantiated composed sampler, contains one key :code:`'child_properties'` that
        has a copy of the child sampler's properties.

        Examples:
            This example instantiates a composed sampler using a D-Wave solver selected by
            the user's default D-Wave Cloud Client configuration_ file and views the
            solver's properties.

            >>> from dwave.system.samplers import DWaveSampler
            >>> from dwave.system.composites import EmbeddingComposite
            >>> sampler = EmbeddingComposite(DWaveSampler())
            >>> sampler.properties   # doctest: +SKIP
            {'child_properties': {u'anneal_offset_ranges': [[-0.2197463755538704,
                0.03821687759418928],
               [-0.2242514597680286, 0.01718456460967399],
               [-0.20860153999435985, 0.05511969218508182],
            >>> # Snipped above response for brevity

        .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config

        """
        return {'child_properties': self.child.properties.copy()}

    def sample(self, bqm, chain_strength=1.0, **parameters):
        """Sample from the provided binary quadratic model.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            chain_strength (float, optional, default=1.0):
                Magnitude of the quadratic bias (in SPIN-space) applied between variables to create
                chains. Note that the energy penalty of chain breaks is 2 * `chain_strength`.

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :class:`dimod.Response`

        Examples:
            This example uses :class:`.EmbeddingComposite` to instantiate a composed sampler
            that submits an unstructured Ising problem to a D-Wave solver, selected by the user's
            default D-Wave Cloud Client configuration_ file, while minor-embedding the problem's
            variables to physical qubits on the solver.

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

        response = child.sample(bqm_embedded, **parameters)

        return dimod.unembed_response(response, embedding, source_bqm=bqm)


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

        # Derive the structure of our composed sampler from the target graph and the embedding
        source_adjacency = dimod.embedding.target_to_source(child_sampler.adjacency, embedding)
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

        self.parameters = parameters = child_sampler.parameters.copy()
        parameters['chain_strength'] = []

        self.properties = {'child_properties': child_sampler.properties.copy()}

        self._embedding = embedding

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

    For an instantiated composed sampler, contains one key :code:`'child_properties'` that
    has a copy of the child sampler's properties.

    """

    @dimod.bqm_structured
    def sample(self, bqm, chain_strength=1.0, **parameters):
        """Sample from the provided binary quadratic model.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            chain_strength (float, optional, default=1.0):
                Magnitude of the quadratic bias (in SPIN-space) applied between variables to create
                chains. Note that the energy penalty of chain breaks is 2 * `chain_strength`.

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :class:`dimod.Response`

        Examples:
            This example uses :class:`.FixedEmbeddingComposite` to instantiate a composed sampler
            that submits an unstructured Ising problem to a D-Wave solver, selected by the user's
            default D-Wave Cloud Client configuration_ file, while minor-embedding the problem's
            variables to physical qubits on the solver.

            >>> from dwave.system.samplers import DWaveSampler
            >>> from dwave.system.composites import FixedEmbeddingComposite
            >>> import dimod
            >>> sampler = FixedEmbeddingComposite(DWaveSampler(), {'a': [0, 4], 'b': [1, 5], 'c': [2, 6]})
            >>> resp = sampler.sample_ising({'a': .5, 'c': 0}, {('a', 'c'): -1})

        """

        # solve the problem on the child system
        child = self.child

        # apply the embedding to the given problem to map it to the child sampler
        __, __, target_adjacency = child.structure

        # get the embedding
        embedding = self._embedding

        bqm_embedded = dimod.embed_bqm(bqm, embedding, target_adjacency, chain_strength=chain_strength)

        response = child.sample(bqm_embedded, **parameters)

        return dimod.unembed_response(response, embedding, source_bqm=bqm)


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
