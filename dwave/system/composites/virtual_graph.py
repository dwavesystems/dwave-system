"""
A dimod composite_ that uses the D-Wave virtual graph feature for improved minor-embedding_.

D-Wave *virtual graphs* simplify the process of minor-embedding by enabling you to more
easily create, optimize, use, and reuse an embedding for a given working graph. When you submit an
embedding and specify a chain strength using these tools, they automatically calibrate the qubits
in a chain to compensate for the effects of biases that may be introduced as a result of strong
couplings.

.. _composite: http://dimod.readthedocs.io/en/latest/reference/samplers.html
.. _minor-embedding: http://dwave-system.readthedocs.io/en/latest/reference/intro.html#minorEmbedding
"""

from six import iteritems

import dimod

from dwave.system.embedding import get_embedding_from_tag
from dwave.system.flux_bias_offsets import get_flux_biases


FLUX_BIAS_KWARG = 'flux_biases'

__all__ = ['VirtualGraphComposite']


class VirtualGraphComposite(dimod.ComposedSampler, dimod.Structured):
    """Composite to use the D-Wave virtual graph feature for minor-embedding.

    Inherits from :class:`dimod.ComposedSampler` and :class:`dimod.Structured`.

    Calibrates qubits in chains to compensate for the effects of biases and enables easy
    creation, optimization, use, and reuse of an embedding for a given working graph.

    Args:
        sampler (:class:`.DWaveSampler`):
            A dimod :class:`dimod.Sampler`. Typically a :obj:`.DWaveSampler` or
            derived composite sampler; other samplers may not work or make sense with
            this composite layer.

        embedding (dict[hashable, iterable]):
            Mapping from a source graph to the specified sampler's graph (the target graph).

        chain_strength (float, optional, default=None):
            Desired chain coupling strength. This is the magnitude of couplings between qubits
            in a chain. If None, uses the maximum available as returned by a SAPI query
            to the D-Wave solver.

        flux_biases (list/False/None, optional, default=None):
            Per-qubit flux bias offsets in the form of a list of lists, where each sublist
            is of length 2 and specifies a variable and the flux bias offset associated with
            that variable. Qubits in a chain with strong negative J values experience a
            J-induced bias; this parameter compensates by recalibrating to remove that bias.
            If False, no flux bias is applied or calculated.
            If None, flux biases are pulled from the database or calculated empirically.

        flux_bias_num_reads (int, optional, default=1000):
            Number of samples to collect per flux bias value.

        flux_bias_max_age (int, optional, default=3600):
            Maximum age (in seconds) allowed for a previously calculated flux bias offset to
            be considered valid.

    Examples:
       This example uses :class:`.VirtualGraphComposite` to instantiate a composed sampler
       that submits a QUBO problem to a D-Wave solver selected by the user's
       default D-Wave Cloud Client configuration_ file. The problem represents a logical
       AND gate using penalty function :math:`P = xy - 2(x+y)z +3z`, where variables x and y
       are the gate's inputs and z the output. This simple three-variable problem is manually
       minor-embedded to a single Chimera_ unit cell: variables x and y are represented by
       qubits 1 and 5, respectively, and z by a two-qubit chain consisting of qubits 0 and 4.
       The chain strength is set to the maximum allowed found from querying the solver's extended
       J range. In this example, the ten returned samples all represent valid states of
       the AND gate.

       >>> from dwave.system.samplers import DWaveSampler
       >>> from dwave.system.composites import VirtualGraphComposite
       >>> embedding = {'x': {1}, 'y': {5}, 'z': {0, 4}}
       >>> DWaveSampler().properties['extended_j_range']   # doctest: +SKIP
       [-2.0, 1.0]
       >>> sampler = VirtualGraphComposite(DWaveSampler(), embedding, chain_strength=2) # doctest: +SKIP
       >>> Q = {('x', 'y'): 1, ('x', 'z'): -2, ('y', 'z'): -2, ('z', 'z'): 3}
       >>> response = sampler.sample_qubo(Q, num_reads=10) # doctest: +SKIP
       >>> for sample in response.samples():    # doctest: +SKIP
       ...     print(sample)
       ...
       {'y': 0, 'x': 1, 'z': 0}
       {'y': 1, 'x': 0, 'z': 0}
       {'y': 1, 'x': 0, 'z': 0}
       {'y': 1, 'x': 1, 'z': 1}
       {'y': 0, 'x': 1, 'z': 0}
       {'y': 1, 'x': 0, 'z': 0}
       {'y': 0, 'x': 1, 'z': 0}
       {'y': 0, 'x': 1, 'z': 0}
       {'y': 0, 'x': 0, 'z': 0}
       {'y': 1, 'x': 0, 'z': 0}

    .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config
    .. _Chimera: http://dwave-system.readthedocs.io/en/latest/reference/intro.html#chimera

    """

    # override the abstract properties
    nodelist = None
    """list:
           Nodes available to the composed sampler.

    Examples:
       This example uses :class:`.VirtualGraphComposite` to instantiate a composed sampler
       that uses a D-Wave solver selected by the user's default D-Wave Cloud Client configuration_ file.
       Because qubits 0, 1, 4, 5 are active on the selected D-Wave solver, the three nodes, x, y, and z,
       specified by the embedding, are all available to problems using this composed sampler.

       >>> from dwave.system.samplers import DWaveSampler
       >>> from dwave.system.composites import VirtualGraphComposite
       >>> embedding = {'x': {1}, 'y': {5}, 'z': {0, 4}}
       >>> sampler = VirtualGraphComposite(DWaveSampler(), embedding)  # doctest: +SKIP
       >>> sampler.nodelist  # doctest: +SKIP
       ['x', 'y', 'z']

    .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config

    """

    edgelist = None
    """list:
           Edges available to the composed sampler.

    Examples:
       This example uses :class:`.VirtualGraphComposite` to instantiate a composed sampler
       that uses a D-Wave solver selected by the user's default D-Wave Cloud Client configuration_ file.
       Because qubits 0, 5, and coupled qubits {0, 4} are all coupled on the selected D-Wave solver, edges
       between three nodes, x, y, and z, as specified by the embedding, are available to problems using this
       composed sampler. However, qubit 8 is in an adjacent unit cell on the D-Wave solver and not directly
       connected to the other four qubits, so node `a` does not share an edge with any other nodes.

       >>> from dwave.system.samplers import DWaveSampler
       >>> from dwave.system.composites import VirtualGraphComposite
       >>> embedding = {'x': {1}, 'y': {5}, 'z': {0, 4}, 'a': {8}}
       >>> sampler = VirtualGraphComposite(DWaveSampler(), embedding)  # doctest: +SKIP
       >>> sampler.edgelist  # doctest: +SKIP
       [('x', 'y'), ('x', 'z'), ('y', 'z')]

    .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config

    """

    adjacency = None
    """dict[variable, set]:
           Adjacency structure for the composed sampler.

    Examples:
       This example uses :class:`.VirtualGraphComposite` to instantiate a composed sampler
       that uses a D-Wave solver selected by the user's default D-Wave Cloud Client configuration_ file.
       Because qubits 0, 5, and coupled qubits {0, 4} are all coupled on the selected D-Wave solver, edges
       between three nodes, x, y, and z, as specified by the embedding, are available to problems using this
       composed sampler. However, qubit 8 is in an adjacent unit cell on the D-Wave solver and not directly
       connected to the other four qubits, so node `a` does not share an edge with any other nodes.

       >>> from dwave.system.samplers import DWaveSampler
       >>> from dwave.system.composites import VirtualGraphComposite
       >>> embedding = {'x': {1}, 'y': {5}, 'z': {0, 4}, 'a': {8}}
       >>> sampler = VirtualGraphComposite(DWaveSampler(), embedding)  # doctest: +SKIP
       >>> sampler.adjacency  # doctest: +SKIP
       {'a': set(), 'x': {'y', 'z'}, 'y': {'x', 'z'}, 'z': {'x', 'y'}}

    .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config

    """

    children = None
    """list: List containing the wrapped sampler."""

    parameters = None
    """dict[str, list]: Parameters in the form of a dict.

    For an instantiated composed sampler, keys are the keyword parameters accepted by the child
    sampler with an additional parameter, 'apply_flux_bias_offsets'.

    Examples:
       This example uses :class:`.VirtualGraphComposite` to instantiate a composed sampler
       that uses a D-Wave solver selected by the user's default D-Wave Cloud Client configuration_ file
       and views the composed sampler's parameters.

       >>> from dwave.system.samplers import DWaveSampler
       >>> from dwave.system.composites import VirtualGraphComposite
       >>> embedding = {'x': {1}, 'y': {5}, 'z': {0, 4}}
       >>> sampler = VirtualGraphComposite(DWaveSampler(), embedding)  # doctest: +SKIP
       >>> sampler.parameters  # doctest: +SKIP
       {u'anneal_offsets': ['parameters'],
        u'anneal_schedule': ['parameters'],
        u'annealing_time': ['parameters'],
        u'answer_mode': ['parameters'],
        'apply_flux_bias_offsets': [],
        u'auto_scale': ['parameters'],
       >>>  # Snipped above response for brevity

    .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config

    """

    properties = None
    """dict: Properties in the form of a dict.

    For an instantiated composed sampler, contains one key :code:`'child_properties'` that
    has a copy of the child sampler's properties.

    Examples:
       This example uses :class:`.VirtualGraphComposite` to instantiate a composed sampler
       that uses a D-Wave solver selected by the user's default D-Wave Cloud Client configuration_ file
       and views the composed sampler's properties.

       >>> from dwave.system.samplers import DWaveSampler
       >>> from dwave.system.composites import VirtualGraphComposite
       >>> embedding = {'x': {1}, 'y': {5}, 'z': {0, 4}}
       >>> sampler = VirtualGraphComposite(DWaveSampler(), embedding)  # doctest: +SKIP
       >>> sampler.properties  # doctest: +SKIP
       {'child_properties': {u'anneal_offset_ranges': [[-0.2197463755538704,
           0.03821687759418928],
          [-0.2242514597680286, 0.01718456460967399],
          [-0.20860153999435985, 0.05511969218508182],
          [-0.2108920134230625, 0.056392603743884134],
       >>>  # Snipped above response for brevity

    .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config

    """

    def __init__(self, sampler, embedding,
                 chain_strength=None,
                 flux_biases=None, flux_bias_num_reads=1000, flux_bias_max_age=3600):
        self.children = [sampler]

        self.parameters = parameters = {'apply_flux_bias_offsets': []}
        parameters.update(sampler.parameters)

        self.properties = {'child_properties': sampler.properties.copy()}

        #
        # Get the adjacency of the child sampler (this is the target for our embedding)
        #
        try:
            target_nodelist, target_edgelist, target_adjacency = sampler.structure
        except:
            # todo, better exception catching
            raise

        #
        # Validate the chain strength, or obtain it from J-range if chain strength is not provided.
        #
        self.chain_strength = self._validate_chain_strength(chain_strength)

        #
        # We want to track the persistent embedding so that we can map input problems
        # to the child sampler.
        #
        if isinstance(embedding, str):
            embedding = get_embedding_from_tag(embedding, target_nodelist, target_edgelist)
        elif not isinstance(embedding, dict):
            raise TypeError("expected input `embedding` to be a dict.")
        self.embedding = embedding

        #
        # Derive the structure of our composed from the target graph and the embedding
        #
        source_adjacency = dimod.embedding.target_to_source(target_adjacency, embedding)
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

        #
        # If the sampler accepts flux bias offsets, we'll want to set them
        #
        if flux_biases is None and FLUX_BIAS_KWARG in sampler.parameters:
            # If nothing is provided, then we either get them from the cache or generate them
            flux_biases = get_flux_biases(sampler, embedding, num_reads=flux_bias_num_reads,
                                          max_age=flux_bias_max_age)
        elif flux_biases:
            if FLUX_BIAS_KWARG not in sampler.accepted_kwargs:
                raise ValueError("Given child sampler does not accept flux_biases.")
            # something provided, error check
            if not isinstance(flux_biases, list):
                flux_biases = list(flux_biases)  # cast to a list
        else:
            # disabled, empty or not available for this sampler so do nothing
            flux_biases = None
        self.flux_biases = flux_biases

    @dimod.bqm_structured
    def sample(self, bqm, apply_flux_bias_offsets=True, **kwargs):
        """Sample from the given Ising model.

        Args:

            h (list/dict):
                Linear biases of the Ising model. If a list, the list's indices
                are used as variable labels.

            J (dict of (int, int):float):
                Quadratic biases of the Ising model.

            apply_flux_bias_offsets (bool, optional):
                If True, use the calculated flux_bias offsets (if available).

            **kwargs:
                Optional keyword arguments for the sampling method, specified per solver.

        Examples:
           This example uses :class:`.VirtualGraphComposite` to instantiate a composed sampler
           that submits an Ising problem to a D-Wave solver selected by the user's
           default D-Wave Cloud Client configuration_ file. The problem represents a logical
           NOT gate using penalty function :math:`P = xy`, where variable x is the gate's input
           and y the output. This simple two-variable problem is manually
           minor-embedded to a single Chimera_ unit cell: each variable is represented by a
           chain of half the cell's qubits, x as qubits 0, 1, 4, 5, and y as qubits 2, 3, 6, 7.
           The chain strength is set to half the maximum allowed found from querying the solver's extended
           J range. In this example, the ten returned samples all represent valid states of
           the NOT gate.

           >>> from dwave.system.samplers import DWaveSampler
           >>> from dwave.system.composites import VirtualGraphComposite
           >>> embedding = {'x': {0, 4, 1, 5}, 'y': {2, 6, 3, 7}}
           >>> DWaveSampler().properties['extended_j_range']   # doctest: +SKIP
           [-2.0, 1.0]
           >>> sampler = VirtualGraphComposite(DWaveSampler(), embedding, chain_strength=1) # doctest: +SKIP
           >>> h = {}
           >>> J = {('x', 'y'): 1}
           >>> response = sampler.sample_ising(h, J, num_reads=10) # doctest: +SKIP
           >>> for sample in response.samples():    # doctest: +SKIP
           ...     print(sample)
           ...
           {'y': -1, 'x': 1}
           {'y': 1, 'x': -1}
           {'y': -1, 'x': 1}
           {'y': -1, 'x': 1}
           {'y': -1, 'x': 1}
           {'y': 1, 'x': -1}
           {'y': 1, 'x': -1}
           {'y': 1, 'x': -1}
           {'y': -1, 'x': 1}
           {'y': 1, 'x': -1}

        .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config
        .. _Chimera: http://dwave-system.readthedocs.io/en/latest/reference/intro.html#chimera

        """

        # apply the embedding to the given problem to map it to the child sampler
        __, __, target_adjacency = self.child.structure
        embedding = self.embedding
        embedded_bqm = dimod.embed_bqm(bqm, self.embedding, target_adjacency, self.chain_strength)

        # solve the problem on the child system
        child = self.child

        if apply_flux_bias_offsets and self.flux_biases is not None:
            # If self.flux_biases is in the old format (list of lists) convert it to the new format (flat list).
            if isinstance(self.flux_biases[0], list):
                flux_bias_dict = dict(self.flux_biases)
                kwargs[FLUX_BIAS_KWARG] = [flux_bias_dict.get(v, 0.) for v in range(child.properties['num_qubits'])]
            else:
                kwargs[FLUX_BIAS_KWARG] = self.flux_biases
            assert len(kwargs[FLUX_BIAS_KWARG]) == child.properties['num_qubits'], \
                "{} must have length {}, the solver's num_qubits."\
                .format(FLUX_BIAS_KWARG, child.properties['num_qubits'])

        # Embed arguments providing initial states for reverse annealing, if applicable.
        kwargs = _embed_initial_state_kwargs(kwargs, self.embedding, self.child.structure[0])

        response = child.sample(embedded_bqm, **kwargs)

        return dimod.unembed_response(response, embedding, source_bqm=bqm)

    def _validate_chain_strength(self, chain_strength):
        """Validate the provided chain strength, checking J-ranges of the sampler's children.

        Args:
            chain_strength (float) The provided chain strength.  Use None to use J-range.

        Returns (float):
            A valid chain strength, either provided or based on available J-range.  Positive finite float.
        """
        child = self.child

        j_range_minimum = None  # Minimum value allowed in J-range
        try:
            # Try to get extended_j_range, just use j_range if that doesn't exist.
            j_range_minimum = min(child.properties.get('extended_j_range', child.properties['j_range']))
        except (AttributeError, KeyError):
            pass

        if chain_strength is not None:
            try:
                chain_strength = float(chain_strength)
            except TypeError:
                raise ValueError("chain_strength could not be converted to float.")
            if not 0. < chain_strength < float('Inf'):
                raise ValueError("chain_strength is not a finite positive number.")

        if j_range_minimum is not None:
            try:
                j_range_minimum = float(j_range_minimum)
            except TypeError:
                raise ValueError("j_range_minimum could not be converted to float.")
            if not 0. < -j_range_minimum < float('Inf'):
                raise ValueError("j_range_minimum is not a finite negative number.")

        if j_range_minimum is None and chain_strength is None:
            raise ValueError("Could not find valid j_range property.  chain_strength must be provided explicitly.")

        if j_range_minimum is None:
            return chain_strength

        if chain_strength is None:
            return -j_range_minimum

        if chain_strength > -j_range_minimum:
            raise ValueError("chain_strength ({}) is too great (larger than -j_range_minimum ({})).".format(chain_strength, -j_range_minimum))
        return chain_strength


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


def _embed_initial_state(initial_state, embedding, qubits):
    """Embed the states provided by the initial_state parameter used for reverse annealing.

    Args:

        initial_state (list of lists): Logical initial state as it would be passed to SAPI for reverse annealing.

        embedding (dict): The embedding used to embed the initial state.  Maps logical indices to chains.

        qubits (list): A list of qubits on the target topology.


    Returns (list of lists):

        The initial_state, embedded according to the provided embedding.
    """

    # Initialize by setting all qubits to 1 (these will be overwritten for active qubits).
    embedded_state = {q: 1 for q in qubits}

    for logical_idx, logical_value in initial_state:  # Iterate through the logical qubit, state pairs.
        for embedded_idx in embedding[logical_idx]:  # For each embedded qubit in the corresponding chain...
            embedded_state[embedded_idx] = int(logical_value)  # make the embedded state equal to the logical state.

    # Convert dictionary to a list of lists.
    embedded_state_list_of_lists = [[q_emb, embedded_state[q_emb]] for q_emb in sorted(embedded_state.keys())]

    return embedded_state_list_of_lists


def _embed_initial_state_kwargs(kwargs, embedding, qubits):
    """Embed the state(s) used for reverse annealing.

    The keyword argument storing the state(s) will be detected by name and handled appropriately.

    Args:

        kwargs (dict): Dictionary of keyword arguments, one of which must end with "initial_state" or "initial_states".

        embedding (dict): The embedding used to embed the initial state.  Maps logical indices to chains.

        qubits (list): A list of qubits on the target topology.

    Returns (list of lists):

        The initial_state(s), embedded according to the provided embedding.
    """

    initial_state_kwargs = {k: v for k, v in iteritems(kwargs)
                            if k.endswith('initial_state') or k.endswith('initial_states')}

    if len(initial_state_kwargs) == 0:
        return kwargs

    if len(initial_state_kwargs) > 1:
        raise ValueError("Multiple arguments providing initial states to sample_ising (only one allowed): "
                         "{}.".format(initial_state_kwargs.keys()))

    initial_state_kwarg_key, initial_state_kwarg_val = next(iteritems(initial_state_kwargs))

    # If it is a single state, embed the single state.
    if initial_state_kwarg_key.endswith('initial_state'):
        kwargs[initial_state_kwarg_key] = _embed_initial_state(initial_state_kwarg_val, embedding, qubits)

    # If it is multiple states, embed each one.
    elif initial_state_kwarg_key.endswith('initial_states'):
        kwargs[initial_state_kwarg_key] = \
            [_embed_initial_state(initial_state, embedding, qubits) for initial_state in initial_state_kwarg_val]

    else:
        raise AssertionError("kwarg should end with 'initial_state' or 'initial_states' "
                             "but it is {}.".format(initial_state_kwarg_key))

    return kwargs
