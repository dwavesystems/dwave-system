import dwave_system_tuning as dst

from dwave_virtual_graph.compatibility23 import itervalues


def get_flux_biases(sampler, embedding, num_reads):
    fbo = dst.oneshot_flux_bias(sampler, embedding.values(), num_reads=num_reads)
    return fbo
