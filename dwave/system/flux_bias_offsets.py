from dwave.system.cache.database_manager import get_flux_biases_from_cache, cache_connect, insert_flux_bias
from dwave.system.exceptions import MissingFluxBias


def get_flux_biases(sampler, embedding, num_reads, chain_strength=1, max_age=3600):
    # try to read the chip_id, otherwise get the name
    system_name = sampler.properties.get('chip_id', str(sampler.__class__))

    try:
        with cache_connect() as cur:
            fbo = get_flux_biases_from_cache(cur, embedding.values(), system_name,
                                             chain_strength=chain_strength)
    except MissingFluxBias:

        # if dwave-system-tuning is not available, then we can't calculate the biases
        try:
            import dwave.system.tuning as dst
        except ImportError:
            return []

        fbo = dst.oneshot_flux_bias(sampler, embedding.values(),
                                    num_reads=num_reads, chain_strength=chain_strength)

        # store them in the cache
        fbo_dict = dict(fbo)
        with cache_connect() as cur:
            for chain in embedding.values():
                flux_bias = fbo_dict[next(iter(chain))]
                insert_flux_bias(cur, chain, system_name, flux_bias, chain_strength)

    return fbo
