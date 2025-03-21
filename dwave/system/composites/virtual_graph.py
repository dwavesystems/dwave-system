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

"""Deprecated and functionality removed. Behaves identically to
:class:`~dwave.system.composites.embedding.FixedEmbeddingComposite`, and will
be completely removed in dwave-system 2.0.

Virtual graphs are deprecated due to improved calibration of newer QPUs; to
calibrate chains for residual biases, follow the instructions in the
`shimming tutorial <https://github.com/dwavesystems/shimming-tutorial>`_.

A :ref:`dimod composite <concept_samplers_composites>` that
uses the D-Wave virtual graph feature for improved
:term:`minor-embedding`.

D-Wave *virtual graphs* simplify the process of minor-embedding by enabling you
to more easily create, optimize, use, and reuse an embedding for a given working
graph. When you submit an embedding and specify a chain strength using these
tools, they automatically calibrate the qubits in a chain to compensate for the
effects of biases that may be introduced as a result of strong couplings.

See the :ref:`index_concepts` section
for explanations of technical terms in descriptions of Ocean tools.
"""

import warnings

import dimod

from dwave.system.composites.embedding import FixedEmbeddingComposite

__all__ = ['VirtualGraphComposite']


class VirtualGraphComposite(FixedEmbeddingComposite):
    """Removed. Used to provide access to the D-Wave virtual graph feature for
    minor-embedding, but now is just a thin wrapper around the
    :class:`~dwave.system.composites.embedding.FixedEmbeddingComposite`.

    .. deprecated:: 1.25.0
        This class is deprecated due to improved calibration of newer QPUs and
        will be removed in 1.27.0; to calibrate chains for residual biases, 
        follow the instructions in the 
        `shimming tutorial <https://github.com/dwavesystems/shimming-tutorial>`_.

    .. versionremoved:: 1.28.0
        This class is now only a pass-through wrapper around the
        :class:`~dwave.system.composites.embedding.FixedEmbeddingComposite`.

        It will be completely removed in dwave-system 2.0.

        For removal reasons and alternatives, see the deprecation note above.
    """

    def __init__(self, sampler, embedding, chain_strength=None, **kwargs):
        super(VirtualGraphComposite, self).__init__(sampler, embedding)

        warnings.warn(
            "'VirtualGraphComposite' functionality is removed due to improved "
            "calibration of newer QPUs and in future will raise an exception. "
            "Currently it's equivalent to 'FixedEmbeddingComposite'. If needed, "
            "follow the instructions in the shimming tutorial at "
            "https://github.com/dwavesystems/shimming-tutorial instead.",
            DeprecationWarning, stacklevel=2
        )

        if chain_strength is not None:
            warnings.warn(
                "'chain_strength' parameter is ignored since dwave-system 1.28.",
                DeprecationWarning, stacklevel=2)

        # for API backwards compatibility
        self.parameters.update(apply_flux_bias_offsets=[])
        self.chain_strength = chain_strength
        self.flux_biases = None

    @dimod.bqm_structured
    def sample(self, bqm, apply_flux_bias_offsets=True, **kwargs):
        return super(VirtualGraphComposite, self).sample(bqm, **kwargs)
