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


class EmbeddingComposite(dimod.Sampler, dimod.Composite):
    """Composite to map unstructured problems to a structured sampler.

    Inherits from :class:`dimod.Sampler` and :class:`dimod.Composite`.

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
