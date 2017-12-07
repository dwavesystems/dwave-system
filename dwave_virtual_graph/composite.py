import itertools

import dimod

import dwave_embedding_utilities as embutil

from dwave_virtual_graph.flux_bias_offsets import get_flux_biases
from dwave_virtual_graph.embedding import get_embedding, get_embedding_from_tag


class VirtualGraph(dimod.TemplateComposite):
    def __init__(self, sampler,
                 embedding_tag=None, embedding=None):
        """Apply the VirtualGraph composite layer to the given solver.

        Args:
            sampler (:class:`dimod.TemplateSampler`): A dimod sampler.
            embedding_tag (str): A tag that can be used to access a
                cached embedding.
            embedding (dict[hashable, iterable]): A mapping from a source
                graph to the given sampler's graph (the target graph).

        Returns:
            `dimod.TemplateComposite`

        """
        # NB: dimod should be updated to provide general types for sampler and composite,
        # this will happen in a future version of dimod.

        # The composite __init__ adds the sampler into self.children
        dimod.TemplateComposite.__init__(self, sampler)
        self._child = sampler  # faster access than self.children[0]

        #
        # Get the adjacency of the child sampler (this is the target for our embedding)
        #
        try:
            target_nodelist, target_edgelist, target_adjacency = sampler.structure
        except:
            # todo, better exception catching
            raise

        #
        # We want to track the persistent embedding so that we can map input problems
        # to the child sampler.
        #
        if embedding is None:
            if embedding_tag is None:
                raise TyperError("either 'embedding' or 'embedding_tag' must be provided")
            elif not isinstance(embedding_tag, (str, unicode)):
                raise TypeError("expected input 'embedding_tag' to be a str")
            embedding = get_embedding_from_tag(embedding_tag)
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

    @dimod.decorators.ising(1, 2)
    def sample_ising(self, h, J, apply_flux_bias_offsets=True, **kwargs):
        """todo

            apply_flux_bias_offsets (bool, optional): Whether to pass
                the `flux_bias` parameter to child samplers if they
                request it.

        """

        __, __, adjacency = self.structure
        if not all(v in adjacency for v in h):
            raise ValueError("nodes in linear bias do not map to the structure")
        if not all(u in adjacency[v] for u, v in J):
            raise ValueError("edges in linear bias do not map to the structure")

        # apply the embedding to the given problem to map it to the child sampler
        __, __, target_adjacency = self._child.structure
        h_emb, J_emb, J_chain = embutil.embed_ising(h, J, self.embedding, target_adjacency)
        J_emb.update(J_chain)

        # solve the problem on the child system
        child = self._child

        if apply_flux_bias_offsets and 'flux_biases' in child.accepted_kwargs:
            kwargs['flux_biases'] = get_flux_biases(self.chain_biases, J_emb)

        response = child.sample_ising(h_emb, J_emb, **kwargs)

        # unembed the problem and save to a new response object
        samples = embutil.unembed_samples(response, self.embedding,
                                          chain_break_method=embutil.minimize_energy,
                                          linear=h, quadratic=J)  # needed by minimize_energy
        source_response = dimod.SpinResponse()
        source_response.add_samples_from(samples,
                                         sample_data=(data for __, data in response.samples(data=True)),
                                         h=h, J=J)
        return source_response

    @property
    def chain_biases(self):
        return {v: 0. for v in itertools.chain(*self.embedding.values())}


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
