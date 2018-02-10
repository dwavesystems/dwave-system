import itertools

import dimod

import dwave_embedding_utilities as embutil

from dwave_virtual_graph.compatibility23 import iteritems
from dwave_virtual_graph.embedding import get_embedding_from_tag
from dwave_virtual_graph.flux_bias_offsets import get_flux_biases


FLUX_BIAS_KWARG = 'flux_biases'


class VirtualGraph(dimod.Composite):
    def __init__(self, sampler, embedding,
                 chain_strength=None,
                 flux_biases=None, flux_bias_num_reads=1000, flux_bias_max_age=3600):
        """Apply the VirtualGraph composite layer to the given solver.

        Args:
            sampler (:class:`dwave_micro_client_dimod.DWaveSampler`):
                A dimod_ sampler. Normally :class:`dwave_micro_client_dimod.DWaveSampler`, or a
                derived composite sampler. Other samplers in general will not work or will not make
                sense with this composite layer.

            embedding (dict[hashable, iterable]):
                A mapping from a source graph to the given sampler's graph (the target graph).

            chain_strength (float, optional, default=None):
                The desired chain strength. If None, will use the maximum available from the
                processor.

            flux_biases (list/False/None, optional, default=None):
                The per-qubit flux bias offsets. If given, should be a list of lists. Each sublist
                should be of length 2 and is the variable and the flux bias
                offset associated with the variable. If `flux_biases` evaluates False, then no
                flux bias is applied or calculated. If None if given, the the flux biases are
                pulled from the database or calculated empirically.

            flux_bias_num_reads (int, optional, default=1000):
                The number of samplers to collect per flux bias value.

            flux_bias_max_age (int, optional, default=3600):
                The maximum age (in seconds) allowed for a previously calculated flux bias offset.

        Returns:
            `dimod.TemplateComposite`

        """
        dimod.Composite.__init__(self, sampler)

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
        source_adjacency = embutil.target_to_source(target_adjacency, embedding)
        try:
            nodelist = sorted(source_adjacency)
            edgelist = sorted(_adjacency_to_edges(source_adjacency))
        except TypeError:
            # python3 does not allow sorting of unlike types, so if nodes have
            # different type names just choose an arbitrary order
            nodelist = list(source_adjacency)
            edgelist = list(_adjacency_to_edges(source_adjacency))
        self.structure = (nodelist, edgelist, source_adjacency)

        #
        # If the sampler accepts flux bias offsets, we'll want to set them
        #
        if flux_biases is None and FLUX_BIAS_KWARG in sampler.sample_kwargs:
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

    def sample_ising(self, h, J, apply_flux_bias_offsets=True, **kwargs):
        """Sample from the given Ising model.

            h (list/dict): Linear terms of the model.

            J (dict of (int, int):float): Quadratic terms of the model.

            apply_flux_bias_offsets (bool, optional):
                If True, use the calculated flux_bias offsets (if available).

            **kwargs: Parameters for the sampling method, specified by the child sampler.

        """

        __, __, adjacency = self.structure
        if not all(v in adjacency for v in h):
            raise ValueError("nodes in linear bias do not map to the structure")
        if not all(u in adjacency[v] for u, v in J):
            raise ValueError("edges in linear bias do not map to the structure")

        # apply the embedding to the given problem to map it to the child sampler
        __, __, target_adjacency = self.child.structure
        h_emb, J_emb, J_chain = embutil.embed_ising(h, J, self.embedding, target_adjacency, self.chain_strength)
        J_emb.update(J_chain)

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

        response = child.sample_ising(h_emb, J_emb, **kwargs)

        # unembed the problem and save to a new response object
        samples = embutil.unembed_samples(response, self.embedding,
                                          chain_break_method=embutil.minimize_energy,
                                          linear=h, quadratic=J)  # needed by minimize_energy

        source_response = dimod.Response(dimod.SPIN)

        for sample, (__, data) in zip(samples, response.df_data.iterrows()):
            data['energy'] = dimod.ising_energy(sample, h, J)
            source_response.add_sample(sample, **data.to_dict())

        return source_response

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
            j_range_minimum = min(child.solver.properties.get('extended_j_range',
                                                              child.solver.properties['j_range']))
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

        else:
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

    initial_state_kwarg_key, initial_state_kwarg_val = next(iteritems(kwargs))

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
