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

import math

__all__ = ['uniform_torque_compensation']

def uniform_torque_compensation(bqm, embedding=None, prefactor=1.414):
    """Calculates chain strength using the RMS of the problem's quadratic biases.
    Attempts to compensate for the torque that would break the chain. 

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
        float: The chain strength, or 1 if chain strength is not applicable
               to the problem. 
            
    """
    if bqm.num_interactions > 0:
        squared_j = (j ** 2 for j in bqm.quadratic.values())
        rms = math.sqrt(sum(squared_j)/bqm.num_interactions)
        avg_degree = bqm.degrees(array=True).mean()

        return prefactor * rms * math.sqrt(avg_degree)
    else:
        return 1    # won't matter (chain strength isn't needed to embed this problem)
