"""
Because the D-Wave System is Chimera-structured but most problems of application interest are not,
it is convenient to be able to map from a structured sampler to an unstructured one.

A structured sampler is one that can only solver problems that map to a specific graph (see structured_)

The :class:`.EmbeddingComposite` uses the minorminer_ library to map unstructured problems to a
structured sampler.

.. _minorminer: https://github.com/dwavesystems/minorminer
.. _structured: http://dimod.readthedocs.io/en/latest/reference/samplers.html#module-dimod.core.structured

"""

import dimod
import dwave_embedding_utilities as embutil
import minorminer


class EmbeddingComposite(dimod.Sampler, dimod.Composite):
    """Composite to map unstructured problems to a structured sampler.

    Args:
        sampler (:class:`dimod.Sampler`):
            A structured dimod sampler.

    """
    def __init__(self, child_sampler):
        if not isinstance(child_sampler, dimod.Structured):
            raise dimod.InvalidComposition("EmbeddingComposite should only be applied to a Structured sampler")
        self._children = [child_sampler]

    @property
    def children(self):
        """list: Contains the single wrapped structured sampler."""
        return self._children

    @property
    def parameters(self):
        """dict[str, list]: The keys are the keyword parameters accepted by the child sampler."""
        # does not add or remove any parameters
        return self.child.parameters.copy()

    @property
    def properties(self):
        """dict: Contains one key :code:`'child_properties'` which has a copy of the child sampler's properties."""
        return {'child_properties': self.child.properties.copy()}

    def sample_ising(self, h, J, **parameters):
        """Sample from the provided unstructured Ising model.

        Args:
            h (list/dict): Linear terms of the model.
            J (dict of (int, int):float): Quadratic terms of the model.
            **parameters: Parameters for the sampling method, specified by the child sampler.

        Returns:
            :class:`dimod.Response`

        """
        if isinstance(h, list):
            h = dict(enumerate(h))

        # solve the problem on the child system
        child = self.child

        # apply the embedding to the given problem to map it to the child sampler
        __, target_edgelist, target_adjacency = child.structure

        # get the embedding
        embedding = minorminer.find_embedding(J, target_edgelist)

        if J and not embedding:
            raise ValueError("no embedding found")

        # this should change in later versions
        if isinstance(embedding, list):
            embedding = dict(enumerate(embedding))

        h_emb, J_emb, J_chain = embutil.embed_ising(h, J, embedding, target_adjacency)
        J_emb.update(J_chain)

        response = child.sample_ising(h_emb, J_emb, **parameters)

        # unembed the problem and save to a new response object
        samples = embutil.unembed_samples(response, embedding,
                                          chain_break_method=embutil.minimize_energy,
                                          linear=h, quadratic=J)  # needed by minimize_energy

        # source_response = dimod.Response(dimod.SPIN)
        data_vectors = response.data_vectors
        data_vectors['energy'] = [dimod.ising_energy(sample, h, J) for sample in samples]

        return dimod.Response.from_dicts(samples, data_vectors, info=response.info, vartype=dimod.SPIN)
