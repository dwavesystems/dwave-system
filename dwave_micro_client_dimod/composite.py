"""
EmbeddingComposite
==================
"""
import itertools

import dimod
import dwave_embedding_utilities as embutil
import minorminer


class EmbeddingComposite(dimod.TemplateComposite):
    """Composite to map unstructured problems to a structured sampler.

    Args:
        sampler (:class:`dimod.TemplateSampler`):
            A structured dimod sampler to be wrapped.

    """
    def __init__(self, sampler):
        # The composite __init__ adds the sampler into self.children
        dimod.TemplateComposite.__init__(self, sampler)
        self._child = sampler  # faster access than self.children[0]

    @dimod.decorators.ising(1, 2)
    def sample_ising(self, h, J, **kwargs):
        """Sample from the provided unstructured Ising model.

        Args:
            h (list/dict): Linear terms of the model.
            J (dict of (int, int):float): Quadratic terms of the model.
            **kwargs: Parameters for the sampling method, specified per solver.

        Returns:
            :class:`dimod.SpinResponse`

        """
        # solve the problem on the child system
        child = self._child

        # apply the embedding to the given problem to map it to the child sampler
        __, target_edgelist, target_adjacency = child.structure

        # get the embedding
        embedding = minorminer.find_embedding(J, target_edgelist)

        # this should change in later versions
        if isinstance(embedding, list):
            embedding = dict(enumerate(embedding))

        h_emb, J_emb, J_chain = embutil.embed_ising(h, J, embedding, target_adjacency)
        J_emb.update(J_chain)

        response = child.sample_ising(h_emb, J_emb, **kwargs)

        # unembed the problem and save to a new response object
        samples = embutil.unembed_samples(response, embedding,
                                          chain_break_method=embutil.minimize_energy,
                                          linear=h, quadratic=J)  # needed by minimize_energy
        source_response = dimod.SpinResponse()
        source_response.add_samples_from(samples,
                                         sample_data=(data for __, data in response.samples(data=True)),
                                         h=h, J=J)
        return source_response
