"""
A dimod composite_ that tiles small problems multiple times to a Chimera-structured sampler.

The :class:`.TilingComposite` takes a problem that can fit on a small Chimera_ graph
and replicates it across a larger Chimera graph to obtain samples from multiple areas
of the solver in one call. For example, a 2x2 Chimera lattice could be tiled 64 times
(8x8) on a fully-yielded D-Wave 2000Q system (16x16).

.. _composite: http://dimod.readthedocs.io/en/latest/reference/samplers.html
.. _Chimera: http://dwave-system.readthedocs.io/en/latest/reference/intro.html#chimera

"""
from __future__ import division
from math import sqrt, ceil

import dimod
import dwave_networkx as dnx

__all__ = ['TilingComposite']


class TilingComposite(dimod.Sampler, dimod.Composite, dimod.Structured):
    """Composite to tile a small problem across a Chimera-structured sampler.

    Inherits from :class:`dimod.Sampler`, :class:`dimod.Composite`, and :class:`dimod.Structured`.

    Enables parallel sampling for small problems (problems that are minor-embeddable in
    a small part of a D-Wave solver's Chimera_ graph).

    The notation *CN* refers to a Chimera graph consisting of an NxN grid of unit cells.
    Each Chimera unit cell is itself a bipartite graph with shores of size t. The D-Wave 2000Q QPU
    supports a C16 Chimera graph: its 2048 qubits are logically mapped into a 16x16 matrix of
    unit cell of 8 qubits (t=4).

    A problem that can be minor-embedded in a single unit cell, for example, can therefore
    be tiled across the unit cells of a D-Wave 2000Q as 16x16 duplicates. This enables
    sampling 256 solutions in a single call.

    .. _Chimera: http://dwave-system.readthedocs.io/en/latest/reference/intro.html#chimera

    Args:
       sampler (:class:`dimod.Sampler`): Structured dimod sampler to be wrapped.
       sub_m (int): Number of rows of Chimera unit cells for minor-embedding the problem once.
       sub_n (int): Number of columns of Chimera unit cells for minor-embedding the problem once.
       t (int, optional, default=4): Size of the shore within each Chimera unit cell.

    Examples:
       This example instantiates a composed sampler using composite :class:`.TilingComposite`
       to tile a QUBO problem on a D-Wave solver, embedding it with composite
       :class:`.EmbeddingComposite` and selecting the D-Wave solver with the user's
       default D-Wave Cloud Client configuration_ file. The two-variable QUBO represents a
       logical NOT gate (two nodes with biases of -1 that are coupled with strength 2) and is
       easily minor-embedded in a single Chimera cell (it needs only any two coupled qubits) and
       so can be tiled multiple times across a D-Wave solver for parallel solution (the two
       nodes should typically have opposite values).

       >>> from dwave.system.samplers import DWaveSampler
       >>> from dwave.system.composites import EmbeddingComposite
       >>> from dwave.system.composites import TilingComposite
       >>> sampler = EmbeddingComposite(TilingComposite(DWaveSampler(), 1, 1, 4))
       >>> Q = {(1, 1): -1, (1, 2): 2, (2, 1): 0, (2, 2): -1}
       >>> response = sampler.sample_qubo(Q)
       >>> for sample in response.samples():    # doctest: +SKIP
       ...     print(sample)
       ...
       {1: 0, 2: 1}
       {1: 1, 2: 0}
       {1: 1, 2: 0}
       {1: 1, 2: 0}
       {1: 0, 2: 1}
       {1: 0, 2: 1}
       {1: 1, 2: 0}
       {1: 0, 2: 1}
       {1: 1, 2: 0}
       >>> # Snipped above response for brevity

    .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config

    """
    nodelist = None
    """list: List of active qubits for the structured solver.

    Examples:
       This example creates a :class:`.TilingComposite` for a problem that requires
       a 2x1 Chimera lattice to solve with a :class:`DWaveSampler` as the sampler.
       It prints the active qubits retrieved from a D-Wave solver selected by
       the user's default D-Wave Cloud Client configuration_ file.

       >>> from dwave.system.samplers import DWaveSampler
       >>> from dwave.system.composites import TilingComposite
       >>> sampler_tile = TilingComposite(DWaveSampler(), 2, 1, 4)
       >>> sampler_tile.nodelist   # doctest: +SKIP
       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config

    """

    edgelist = None
    """list: List of active couplers for the D-Wave solver.

    Examples:
       This example creates a :class:`.TilingComposite` for a problem that requires
       a 1x2 Chimera lattice to solve with a :class:`DWaveSampler` as the sampler.
       It prints the active couplers retrieved from a D-Wave solver selected by
       the user's default D-Wave Cloud Client configuration_ file.

       >>> from dwave.system.samplers import DWaveSampler
       >>> from dwave.system.composites import TilingComposite
       >>> sampler_tile = TilingComposite(DWaveSampler(), 1, 2, 4)
       >>> sampler_tile.edgelist   # doctest: +SKIP
       [[0, 4],
       [0, 5],
       [0, 6],
       [0, 7],
       [1, 4],
       [1, 5],
       [1, 6],
       [1, 7],
       [2, 4],
       [2, 5],
       [2, 6],
       [2, 7],
       [3, 4],
       [3, 5],
       [3, 6],
       [3, 7],
       [4, 12],
       [5, 13],
       [6, 14],
       [7, 15],
       [8, 12],
       [8, 13],
       [8, 14],
       [8, 15],
       [9, 12],
       [9, 13],
       [9, 14],
       [9, 15],
       [10, 12],
       [10, 13],
       [10, 14],
       [10, 15],
       [11, 12],
       [11, 13],
       [11, 14],
       [11, 15]]

    .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config

    """

    parameters = None
    """dict[str, list]: Parameters in the form of a dict.

    For an instantiated composed sampler, keys are the keyword parameters accepted by the
    child sampler.

    Examples:
       This example instantiates a :class:`.TilingComposite` sampler using a D-Wave solver
       selected by the user's default D-Wave Cloud Client configuration_ file and views the
       solver's parameters.

       >>> from dwave.system.samplers import DWaveSampler
       >>> from dwave.system.composites import TilingComposite
       >>> sampler_tile = TilingComposite(DWaveSampler(), 1, 1, 4)
       >>> sampler_tile.parameters   # doctest: +SKIP
       {u'anneal_offsets': ['parameters'],
        u'anneal_schedule': ['parameters'],
        u'annealing_time': ['parameters'],
        u'answer_mode': ['parameters'],
        u'auto_scale': ['parameters'],
       >>> # Snipped above response for brevity

    .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config

       """

    properties = None
    """dict: Properties in the form of a dict.

    For an instantiated composed sampler, contains one key :code:`'child_properties'` that
    has a copy of the child sampler's properties.

    Examples:
       This example instantiates a :class:`.TilingComposite` sampler using a D-Wave solver
       selected by the user's default D-Wave Cloud Client configuration_ file and views the
       solver's properties.

       >>> from dwave.system.samplers import DWaveSampler
       >>> from dwave.system.composites import TilingComposite
       >>> sampler_tile = TilingComposite(DWaveSampler(), 1, 1, 4)
       >>> sampler_tile.properties   # doctest: +SKIP
       {'child_properties': {u'anneal_offset_ranges': [[-0.2197463755538704,
           0.03821687759418928],
          [-0.2242514597680286, 0.01718456460967399],
          [-0.20860153999435985, 0.05511969218508182],
          [-0.2108920134230625, 0.056392603743884134],
          [-0.21788292874621265, 0.03360435584845211],
          [-0.21700680373359477, 0.005297355417068621],
       >>> # Snipped above response for brevity

    .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config

       """

    children = None
    """list: The single wrapped structured sampler."""

    def __init__(self, sampler, sub_m, sub_n, t=4):

        self.parameters = sampler.parameters.copy()
        self.properties = properties = {'child_properties': sampler.properties}

        tile = dnx.chimera_graph(sub_m, sub_n, t)
        self.nodelist = sorted(tile.nodes)
        self.edgelist = sorted(sorted(edge) for edge in tile.edges)
        # dimod.Structured abstract base class automatically populates adjacency and structure as
        # mixins based on nodelist and edgelist

        if not isinstance(sampler, dimod.Structured):
            # we could also just tile onto the unstructured sampler but in that case we would need
            # to know how many tiles to use
            raise ValueError("given child sampler should be structured")
        self.children = [sampler]

        nodes_per_cell = t * 2
        edges_per_cell = t * t
        m = n = int(ceil(sqrt(ceil(len(sampler.structure.nodelist) / nodes_per_cell))))  # assume square lattice shape
        system = dnx.chimera_graph(m, n, t, node_list=sampler.structure.nodelist, edge_list=sampler.structure.edgelist)
        c2i = {chimera_index: linear_index for (linear_index, chimera_index) in system.nodes(data='chimera_index')}
        sub_c2i = {chimera_index: linear_index for (linear_index, chimera_index) in tile.nodes(data='chimera_index')}

        # Count the connections between these qubits
        def _between(qubits1, qubits2):
            edges = [edge for edge in system.edges if edge[0] in qubits1 and edge[1] in qubits2]
            return len(edges)

        # Get the list of qubits in a cell
        def _cell_qubits(i, j):
            return [c2i[(i, j, u, k)] for u in range(2) for k in range(t) if (i, j, u, k) in c2i]

        # get a mask of complete cells
        cells = [[False for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                qubits = _cell_qubits(i, j)
                cells[i][j] = len(qubits) == nodes_per_cell and _between(qubits, qubits) == edges_per_cell

        # List of 'embeddings'
        self.embeddings = properties['embeddings'] = embeddings = []

        # For each possible chimera cell check if the next few cells are complete
        for i in range(m + 1 - sub_m):
            for j in range(n + 1 - sub_n):

                # Check if the sub cells are matched
                match = all(cells[i + sub_i][j + sub_j] for sub_i in range(sub_m) for sub_j in range(sub_n))

                # Check if there are connections between the cells.
                for sub_i in range(sub_m):
                    for sub_j in range(sub_n):
                        if sub_m > 1 and sub_i < sub_m - 1:
                            match &= _between(_cell_qubits(i + sub_i, j + sub_j),
                                              _cell_qubits(i + sub_i + 1, j + sub_j)) == t
                        if sub_n > 1 and sub_j < sub_n - 1:
                            match &= _between(_cell_qubits(i + sub_i, j + sub_j),
                                              _cell_qubits(i + sub_i, j + sub_j + 1)) == t

                if match:
                    # Pull those cells out into an embedding.
                    embedding = {}
                    for sub_i in range(sub_m):
                        for sub_j in range(sub_n):
                            cells[i + sub_i][j + sub_j] = False  # Mark cell as matched
                            for u in range(2):
                                for k in range(t):
                                    embedding[sub_c2i[sub_i, sub_j, u, k]] = {c2i[(i + sub_i, j + sub_j, u, k)]}

                    embeddings.append(embedding)

        if len(embeddings) == 0:
            raise ValueError("no tile embeddings found; is the sampler Chimera structured?")

    @dimod.bqm_structured
    def sample(self, bqm, **kwargs):
        """Sample from the provided binary quadratic model

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            **kwargs:
                Optional keyword arguments for the sampling method, specified per solver.

        Returns:
            :class:`dimod.Response`

        Examples:
            This example uses :class:`.TilingComposite` to instantiate a composed sampler
            that submits a simple Ising problem of just two variables that map to qubits 0 and 1
            on the D-Wave solver selected by the user's default D-Wave Cloud Client
            configuration_ file. (The simplicity of this example obviates the need for an embedding
            composite.) Because the problem fits in a single Chimera_ unit cell, it is tiled 
            across the solver's entire Chimera graph, resulting in multiple samples.

            >>> from dwave.system.samplers import DWaveSampler
            >>> from dwave.system.composites import EmbeddingComposite
            >>> samplertile = TilingComposite(DWaveSampler(), 1, 1, 4)
            >>> response = sampler_tile.sample_ising({0: -1, 1: 1}, {})
            >>> for sample in response.samples():    # doctest: +SKIP
            ...     print(sample)
            ...
            {0: 1, 1: -1}
            {0: 1, 1: -1}
            {0: 1, 1: -1}
            {0: 1, 1: -1}
            {0: 1, 1: -1}
            {0: 1, 1: -1}
            {0: 1, 1: -1}
            {0: 1, 1: -1}
            >>> # Snipped above response for brevity

        .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config
        .. _Chimera: http://dwave-system.readthedocs.io/en/latest/reference/intro.html#chimera

        """

        # apply the embeddings to the given problem to tile it across the child sampler
        embedded_bqm = dimod.BinaryQuadraticModel.empty(bqm.vartype)
        __, __, target_adjacency = self.child.structure
        for embedding in self.embeddings:
            embedded_bqm.update(dimod.embed_bqm(bqm, embedding, target_adjacency))

        # solve the problem on the child system
        response = self.child.sample(embedded_bqm, **kwargs)

        data_vectors = response.data_vectors.copy()

        source_response = None

        for embedding in self.embeddings:

            # filter for problem variables
            embedding = {v: chain for v, chain in embedding.items() if v in bqm.linear}

            tile_response = dimod.unembed_response(response, embedding, source_bqm=bqm)

            if source_response is None:
                source_response = tile_response
                source_response.info.update(response.info)  # overwrite the info
            else:
                source_response.update(tile_response)

        return source_response

    @property
    def num_tiles(self):
        return len(self.embeddings)


def draw_tiling(sampler, t=4):
    """Draw Chimera graph of sampler with colored tiles.

    Args:
        sampler (:class:`dwave_micro_client_dimod.TilingComposite`): A tiled dimod sampler to be drawn.
        t (int): The size of the shore within each Chimera cell.

    Uses `dwave_networkx.draw_chimera` (see draw_chimera_).
    Linear biases are overloaded to color the graph according to which tile each Chimera cell belongs to.

    .. _draw_chimera: http://dwave-networkx.readthedocs.io/en/latest/reference/generated/dwave_networkx.drawing.chimera_layout.draw_chimera.html

    """

    child = sampler.child
    nodes_per_cell = t * 2
    m = n = int(ceil(sqrt(ceil(len(child.structure.nodelist) / nodes_per_cell))))  # assume square lattice shape
    system = dnx.chimera_graph(m, n, t, node_list=child.structure.nodelist, edge_list=child.structure.edgelist)

    labels = {node: -len(sampler.embeddings) for node in system.nodes}  # unused cells are blue
    labels.update({node: i for i, embedding in enumerate(sampler.embeddings) for s in embedding.values() for node in s})
    dnx.draw_chimera(system, linear_biases=labels)
