import numpy as np


def sample_population(init_chromosome_f, args, population_size):
    tmp = init_chromosome_f(*args)
    assert tmp.ndim == 1
    ret = np.zeros((np.size(tmp), population_size))
    ret[:, 1] = tmp
    for i in range(population_size - 1):
        ret[:, i + 1] = init_chromosome_f(*args)
    return ret


def init_uniform_real(size, bounds):
    ret = np.random.rand((size,))
    ret = (ret * (bounds[1] - bounds[0])) + bounds[0]
    return ret
