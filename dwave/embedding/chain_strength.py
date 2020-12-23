# Copyright 2020 D-Wave Systems Inc.
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

"""Utility functions for calculating chain strength.

Examples:
    This example uses :func:`uniform_torque_compensation`, given a prefactor of 2, 
    to calculate a chain strength that :class:`EmbeddingComposite` then uses.

    >>> from functools import partial
    >>> from dwave.system import EmbeddingComposite, DWaveSampler
    >>> from dwave.embedding.chain_strength import uniform_torque_compensation
    ...
    >>> Q = {(0,0): 1, (1,1): 1, (2,3): 2, (1,2): -2, (0,3): -2}
    >>> sampler = EmbeddingComposite(DWaveSampler())
    >>> # partial() can be used when the BQM or embedding is not accessible
    >>> chain_strength = partial(uniform_torque_compensation, prefactor=2)
    >>> sampleset = sampler.sample_qubo(Q, chain_strength=chain_strength, return_embedding=True)
    >>> sampleset.info['embedding_context']['chain_strength']
    1.224744871391589

"""
import math

__all__ = ['uniform_torque_compensation', 'scaled']

def uniform_torque_compensation(bqm, embedding=None, prefactor=1.414):
    """Chain strength that attempts to compensate for torque that would break
    the chain.

    The RMS of the problem's quadratic biases is used for calculation.

    Args: 
        bqm (:obj:`.BinaryQuadraticModel`):
            A binary quadratic model.

        embedding (dict/:class:`.EmbeddedStructure`, default=None):
            Included to satisfy the `chain_strength` callable specifications 
            for `embed_bqm`. 

        prefactor (float, optional, default=1.414):
            Prefactor used for scaling. For non-pathological problems, the recommended 
            range of prefactors to try is [0.5, 2].

    Returns:
        float: The chain strength, or 1 if chain strength is not applicable.

    """
    if bqm.num_interactions > 0:
        squared_j = (j ** 2 for j in bqm.quadratic.values())
        rms = math.sqrt(sum(squared_j)/bqm.num_interactions)
        avg_degree = bqm.degrees(array=True).mean()

        return prefactor * rms * math.sqrt(avg_degree)
    else:
        return 1    # won't matter (chain strength isn't needed to embed this problem)

def scaled(bqm, embedding=None, prefactor=1.0):
    """Chain strength that is scaled to the problem bias range.

    Args:
        bqm (:obj:`.BinaryQuadraticModel`):
            A binary quadratic model.

        embedding (dict/:class:`.EmbeddedStructure`, default=None):
            Included to satisfy the `chain_strength` callable specifications 
            for `embed_bqm`. 

        prefactor (float, optional, default=1.0):
            Prefactor used for scaling. 

    Returns:
        float: The chain strength, or 1 if chain strength is not applicable.

    """  
    if bqm.num_interactions > 0:
        max_bias = max(max(bqm.linear.max(), -bqm.linear.min()), 
                       max(bqm.quadratic.max(), -bqm.quadratic.min()))
        return prefactor * max_bias
    else:
        return 1    # won't matter (chain strength isn't needed to embed this problem)
