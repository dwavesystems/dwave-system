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
    This example uses :func:`uniform_torque_compensation`, given a prefactor of 3,
    to calculate a chain strength that :class:`EmbeddingComposite` then uses.

    >>> import numpy as np
    >>> from functools import partial
    >>> from dwave.system import EmbeddingComposite, DWaveSampler
    >>> from dwave.embedding.chain_strength import uniform_torque_compensation
    ...
    >>> Q = np.triu(np.ones((5, 5)))        # K5 with all biases set to 1.0
    >>> sampler = EmbeddingComposite(DWaveSampler())
    >>> # partial() can be used when the BQM or embedding is not accessible
    >>> chain_strength = partial(uniform_torque_compensation, prefactor=3)
    >>> sampleset = sampler.sample_qubo(Q, chain_strength=chain_strength, return_embedding=True)
    >>> sampleset.info['embedding_context']['chain_strength']
    1.5

"""
import math
import numpy as np

__all__ = ['uniform_torque_compensation', 'scaled']

def uniform_torque_compensation(bqm, embedding=None, prefactor=1.414):
    r"""Chain strength that attempts to compensate for chain-breaking torque.

    The problem's connectivity\ [#]_ and quadratic biases are used to calculate
    a value of chain strength that preforms reasonably well on many problems.

    As the quantum annealing progresses, and the amplitude of the transverse
    field (:math:`A(s)` in the :ref:`qpu_qa_implementation` section) decreases,
    the wavefunction representing the QPU's quantum state develops long-range
    order. A chain strength that increases the correlation of the chains' qubits
    together with the development of this long-range order produces efficient
    quantum dynamics [Ray2020]_. For many hard, frustrated problems (such as
    spin glasses) with typical chain topology (path or tree-like chains), the
    optimal chain strength is proportional to the root of typical variable
    connectivity and the root mean square (RMS) of coupling strength
    (:math:`\sqrt{\text{connectivity}} \times \text{coupling}_{RMS}`).

    This chain strength, chosen for its dynamically-efficient scaling, also
    meets another requirement, even in challenging models: it must be large
    enough so that, toward the end of the quantum annealing, intact chains of
    the embedded problem (the programmed Hamiltonian) have lower energy than
    broken chains for the ground state. This allows the embedded problem to
    recover the ground state in the adiabatic limit. Consider the following
    observation: in a frustrated problem, such as a spin glass, qubits in chains
    are subject to random energy signals from neighbors. If you split the chain
    in two with equal numbers of neighboring chains, the central limit theorem
    dictates that the signal in each half has zero mean and variance
    proportional to the connectivity and the RMS of the coupling values. In
    combination, these can create a random torque on the chain, favouring
    misalignment (a broken chain). For the central coupling to maintain
    alignment of the two halves, it needs an energy penalty larger than the
    torque signal, and so scales as
    :math:`\sqrt{\text{connectivity}} \times \text{coupling}_{RMS}`.

    .. [#]
        A measure of the density of interactions between variables; for example
        the average degree (number of edges per node in the graph representing
        the problem).

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
    num_interactions = bqm.num_interactions

    # NumPy arrays improves performance through vectorization
    quadratic_array = np.fromiter(bqm.quadratic.values(), dtype=float, count=num_interactions)

    if num_interactions:
        squared_j = quadratic_array**2

        rms = math.sqrt(squared_j.sum() / num_interactions)
        avg_degree = bqm.degrees(array=True).mean()

        return prefactor * rms * math.sqrt(avg_degree)

    # won't matter (chain strength isn't needed to embed this problem)
    return 1

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
        max_bias = max(
            bqm.linear.max(), -bqm.linear.min(), bqm.quadratic.max(), -bqm.quadratic.min()
        )
        return prefactor * max_bias

    # won't matter (chain strength isn't needed to embed this problem)
    return 1
