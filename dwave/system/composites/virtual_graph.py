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

"""
A :std:doc:`dimod composite <oceandocs:docs_dimod/reference/samplers>` that uses the D-Wave virtual
graph feature for improved :std:doc:`minor-embedding <oceandocs:docs_system/intro>`.

D-Wave *virtual graphs* simplify the process of minor-embedding by enabling you to more
easily create, optimize, use, and reuse an embedding for a given working graph. When you submit an
embedding and specify a chain strength using these tools, they automatically calibrate the qubits
in a chain to compensate for the effects of biases that may be introduced as a result of strong
couplings.

See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/stable/concepts/index.html>`_
for explanations of technical terms in descriptions of Ocean tools.
"""

import dimod

from dwave.system.composites.embedding import FixedEmbeddingComposite
from dwave.system.flux_bias_offsets import get_flux_biases


FLUX_BIAS_KWARG = 'flux_biases'

__all__ = ['VirtualGraphComposite']


class VirtualGraphComposite(FixedEmbeddingComposite):
    """Composite to use the D-Wave virtual graph feature for minor-embedding.

    Calibrates qubits in chains to compensate for the effects of biases and enables easy
    creation, optimization, use, and reuse of an embedding for a given working graph.

    Inherits from :class:`dimod.ComposedSampler` and :class:`dimod.Structured`.

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
            Number of samples to collect per flux bias value to calculate calibration
            information.

        flux_bias_max_age (int, optional, default=3600):
            Maximum age (in seconds) allowed for a previously calculated flux bias offset to
            be considered valid.

    .. attention::
       D-Wave's *virtual graphs* feature can require many seconds of D-Wave system time to calibrate
       qubits to compensate for the effects of biases. If your account has limited
       D-Wave system access, consider using :class:`.FixedEmbeddingComposite` instead.

    Examples:
       This example uses :class:`.VirtualGraphComposite` to instantiate a composed sampler
       that submits a QUBO problem to a D-Wave solver.
       The problem represents a logical
       AND gate using penalty function :math:`P = xy - 2(x+y)z +3z`, where variables x and y
       are the gate's inputs and z the output. This simple three-variable problem is manually
       minor-embedded to a single :std:doc:`Chimera <oceandocs:docs_system/intro>` unit cell:
       variables x and y are represented by qubits 1 and 5, respectively, and z by a
       two-qubit chain consisting of qubits 0 and 4.
       The chain strength is set to the maximum allowed found from querying the solver's extended
       J range. In this example, the ten returned samples all represent valid states of
       the AND gate.

       >>> from dwave.system import DWaveSampler, VirtualGraphComposite
       >>> embedding = {'x': {1}, 'y': {5}, 'z': {0, 4}}
       >>> qpu_2000q = DWaveSampler(solver={'topology__type': 'chimera'})
       >>> qpu_2000q.properties['extended_j_range']
       [-2.0, 1.0]
       >>> sampler = VirtualGraphComposite(qpu_2000q, embedding, chain_strength=2) # doctest: +SKIP
       >>> Q = {('x', 'y'): 1, ('x', 'z'): -2, ('y', 'z'): -2, ('z', 'z'): 3}
       >>> sampleset = sampler.sample_qubo(Q, num_reads=10) # doctest: +SKIP
       >>> print(sampleset)    # doctest: +SKIP
          x  y  z energy num_oc. chain_.
       0  1  0  0    0.0       2     0.0
       1  0  1  0    0.0       3     0.0
       2  1  1  1    0.0       3     0.0
       3  0  0  0    0.0       2     0.0
       ['BINARY', 4 rows, 10 samples, 3 variables]

    See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/stable/concepts/index.html>`_
    for explanations of technical terms in descriptions of Ocean tools.

    """

    def __init__(self, sampler, embedding,
                 chain_strength=None,
                 flux_biases=None,
                 flux_bias_num_reads=1000,
                 flux_bias_max_age=3600):

        super(VirtualGraphComposite, self).__init__(sampler, embedding)
        self.parameters.update(apply_flux_bias_offsets=[])

        # Validate the chain strength, or obtain it from J-range if chain strength is not provided.
        self.chain_strength = _validate_chain_strength(sampler, chain_strength)

        if flux_biases is False:  # use 'is' because bool(None) is False
            # in this case we are done
            self.flux_biases = None
            return

        if FLUX_BIAS_KWARG not in sampler.parameters:
            raise ValueError("Given child sampler does not accept flux_biases.")

        # come back as a dict
        flux_biases = get_flux_biases(sampler, embedding,
                                      num_reads=flux_bias_num_reads,
                                      chain_strength=self.chain_strength,
                                      max_age=flux_bias_max_age)

        self.flux_biases = [flux_biases.get(v, 0.0) for v in range(sampler.properties['num_qubits'])]

        return

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
           that submits an Ising problem to a D-Wave solver.
           The problem represents a logical
           NOT gate using penalty function :math:`P = xy`, where variable x is the gate's input
           and y the output. This simple two-variable problem is manually minor-embedded
           to a single :std:doc:`Chimera <oceandocs:docs_system/intro>` unit cell: each variable
           is represented by a chain of half the cell's qubits, x as qubits 0, 1, 4, 5,
           and y as qubits 2, 3, 6, 7.
           The chain strength is set to half the maximum allowed found from querying the solver's extended
           J range. In this example, the ten returned samples all represent valid states of
           the NOT gate.

           >>> from dwave.system import DWaveSampler, VirtualGraphComposite
           >>> embedding = {'x': {0, 4, 1, 5}, 'y': {2, 6, 3, 7}}
           >>> qpu_2000q = DWaveSampler(solver={'topology__type': 'chimera'})
           >>> qpu_2000q.properties['extended_j_range']
           [-2.0, 1.0]
           >>> sampler = VirtualGraphComposite(qpu_2000q, embedding, chain_strength=1) # doctest: +SKIP
           >>> h = {}
           >>> J = {('x', 'y'): 1}
           >>> sampleset = sampler.sample_ising(h, J, num_reads=10) # doctest: +SKIP
           >>> print(sampleset)    # doctest: +SKIP
              x  y energy num_oc. chain_.
           0 -1 +1   -1.0       6     0.0
           1 +1 -1   -1.0       4     0.0
           ['SPIN', 2 rows, 10 samples, 2 variables]

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/stable/concepts/index.html>`_
        for explanations of technical terms in descriptions of Ocean tools.

        """

        if apply_flux_bias_offsets:
            if self.flux_biases is not None:
                kwargs[FLUX_BIAS_KWARG] = self.flux_biases

        kwargs.setdefault('chain_strength', self.chain_strength)

        return super(VirtualGraphComposite, self).sample(bqm, **kwargs)


def _validate_chain_strength(sampler, chain_strength):
    """Validate the provided chain strength, checking J-ranges of the sampler's children.

    Args:
        chain_strength (float) The provided chain strength.  Use None to use J-range.

    Returns (float):
        A valid chain strength, either provided or based on available J-range.  Positive finite float.

    """
    properties = sampler.properties

    if 'extended_j_range' in properties:
        max_chain_strength = - min(properties['extended_j_range'])
    elif 'j_range' in properties:
        max_chain_strength = - min(properties['j_range'])
    else:
        raise ValueError("input sampler should have 'j_range' and/or 'extended_j_range' property.")

    if chain_strength is None:
        chain_strength = max_chain_strength
    elif chain_strength > max_chain_strength:
        raise ValueError("Provided chain strength exceedds the allowed range.")

    return chain_strength
