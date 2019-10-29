import numpy as np


def sample_population(init_chromosome_f, args, population_size):
    """
    creates population from single chromosome init function
    :param init_chromosome_f:
    :param args:
    :param population_size:
    :return:
    """
    tmp = init_chromosome_f(*args)
    assert tmp.ndim == 1
    ret = np.zeros((np.size(tmp), population_size))
    ret[:, 0] = tmp
    for i in range(population_size - 1):
        ret[:, i + 1] = init_chromosome_f(*args)
    return ret


def init_uniform_real(size, bounds):
    ret = np.random.rand(size)
    ret = (ret * (bounds[1] - bounds[0])) + bounds[0]
    return ret


def init_shuffled_integer_array(size):
    ret = np.arange(size)
    np.random.shuffle(ret)
    return ret


def init_conv_nn(length, width, variance):
    return np.random.normal(0, variance, length*width)
