import dwave_system_tuning as dst

from dwave_virtual_graph.compatibility23 import itervalues
from dwave_virtual_graph.cache.database_manager import get_flux_biases_from_cache, cache_connect, insert_flux_bias
from dwave_virtual_graph.exceptions import MissingFluxBias


def get_flux_biases(sampler, embedding, num_reads, chain_strength=1, max_age=3600):
    system_name = sampler.solver.properties['chip_id']

    try:
        with cache_connect() as cur:
            fbo = get_flux_biases_from_cache(cur, embedding.values(), system_name,
                                             chain_strength=chain_strength)
    except MissingFluxBias:
        fbo = dst.oneshot_flux_bias(sampler, embedding.values(),
                                    num_reads=num_reads, chain_strength=chain_strength)

        # store them in the cache
        fbo_dict = dict(fbo)
        with cache_connect() as cur:
            for chain in embedding.values():
                flux_bias = fbo_dict[next(iter(chain))]
                insert_flux_bias(cur, chain, system_name, flux_bias, chain_strength)

    return fbo
