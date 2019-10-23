import numpy as np


# ------ selectors -----

def pointer_wheel_selector(population, fitness, pointers_num, spins_num):
    """
    pointer wheel selection type
    :param population: all chromosomes
    :param fitness: array with values of fitness
    :param pointers_num: number of equidistant pointers
    :param spins_num: number of spins
    :return: chosen parents [chrom_size, pointers_num*spins_num]
    """
    def single_spin(population, cumulative_fitness, pointers_num):
        ret = np.empty((population.shape[0], pointers_num))
        max_fit = cum_fit[-1]
        spin = (np.random.rand(1) * max_fit)/pointers_num
        incr = max_fit/pointers_num
        for i in range(pointers_num):
            cum_sum = i * incr + spin
            idx = np.searchsorted(cum_fit, cum_sum)
            ret[:, i] = population[:, idx].reshape((population.shape[0], ))
        return ret
    ret = np.empty((np.shape(population)[0], pointers_num * spins_num))
    # normalise above 0
    fit_min = fitness.min()
    my_fitness = fitness
    if fit_min < 0:
        my_fitness = fitness - fit_min
    cum_fit = np.cumsum(my_fitness)
    for i in range(spins_num):
        idx = i*pointers_num
        ret[:, idx:idx+pointers_num] = single_spin(population, cum_fit, pointers_num)
    return ret


# ------ replacement strategy ------

def pointer_wheel_replacement(pop, fit, new_pop, new_fit, ret_size):
    whole_pop = np.hstack((pop, new_pop))
    whole_fit = np.hstack((fit, new_fit))
    return pointer_wheel_selector(whole_pop, whole_fit, ret_size, 1)
