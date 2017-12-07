import itertools


def get_flux_biases(chain_biases, J,
                    min_offset=-4e-5, max_offset=4e-5):
    """todo

    Returns:
        dict[hashable, float]: The flux_biases for each variable in the
        target problem.

    """
    # placeholder
    return {v: 0.0 for v in chain_biases}


def get_chain_biases(sampler, chains):
    raise NotImplementedError
