# coding: utf-8
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

import warnings

import dimod

from dwave.system.cache.database_manager import get_flux_biases_from_cache, cache_connect, insert_flux_bias
from dwave.system.exceptions import MissingFluxBias
from dwave.system.samplers.dwave_sampler import DWaveSampler


def get_flux_biases(sampler, embedding, chain_strength, num_reads=1000, max_age=3600):
    """Get the flux bias offsets for sampler and embedding.

    Args:
        sampler (:obj:`.DWaveSampler`):
            A D-Wave sampler.

        embedding (dict[hashable, iterable]):
            Mapping from a source graph to the specified sampler’s graph (the target graph). The
            keys of embedding should be nodes in the source graph, the values should be an iterable
            of nodes in the target graph.

        chain_strength (number):
            Desired chain coupling strength. This is the magnitude of couplings between qubits
            in a chain.

        num_reads (int, optional, default=1000):
            The number of reads per system call if new flux biases need to be calculated.

        max_age (int, optional, default=3600):
            The maximum age (in seconds) allowed for previously calculated flux bias offsets.

    Returns:
        dict: A dict where the keys are the nodes in the chains and the values are the flux biases.

    """

    if not isinstance(sampler, dimod.Sampler):
        raise TypeError("input sampler should be DWaveSampler")

    # try to read the chip_id, otherwise get the name
    system_name = sampler.properties.get('chip_id', str(sampler.__class__))

    try:
        with cache_connect() as cur:
            fbo = get_flux_biases_from_cache(cur, embedding.values(), system_name,
                                             chain_strength=chain_strength,
                                             max_age=max_age)
        return fbo
    except MissingFluxBias:
        pass

    # if dwave-drivers is not available, then we can't calculate the biases
    try:
        import dwave.drivers as drivers
    except ImportError:
        msg = ("dwave-drivers not found, cannot calculate flux biases. dwave-drivers can be "
               "installed with "
               "'pip install dwave-drivers --extra-index-url https://pypi.dwavesys.com/simple'. "
               "See documentation for dwave-drivers license.")
        raise RuntimeError(msg)

    fbo = drivers.oneshot_flux_bias(sampler, embedding.values(), num_reads=num_reads,
                                    chain_strength=chain_strength,
                                    label='VirtualGraph flux bias measurements')

    # store them in the cache
    with cache_connect() as cur:
        for chain in embedding.values():
            v = next(iter(chain))
            flux_bias = fbo.get(v, 0.0)
            insert_flux_bias(cur, chain, system_name, flux_bias, chain_strength)

    return fbo
