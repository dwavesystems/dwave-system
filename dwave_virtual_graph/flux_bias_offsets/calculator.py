import itertools


def get_flux_biases(sampler, embedding,
                    min_offset=-4e-5, max_offset=4e-5):
    """todo

    Returns:
        dict[hashable, float]: The flux_biases for each variable in the
        target problem.

    """
    # for now always return 0
    return {v: 0. for v in itertools.chain(*embedding.values())}
