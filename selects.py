import numpy as np
import utilities as ut
import random


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
    def single_spin(population, cum_fit, pointers_num):
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
    cum_fit = np.cumsum(ut.fitness_scaling(fitness))
    for i in range(spins_num):
        idx = i*pointers_num
        ret[:, idx:idx+pointers_num] = single_spin(population, cum_fit, pointers_num)
    return ret


def multiple_times_selector(population, fitness, multiplier):
    pop_size = population.shape[1]
    new_pop = np.empty((population.shape[0], pop_size*multiplier))
    for i in range(multiplier):
        start = i*pop_size
        new_pop[:, start:start+pop_size] = population
    return new_pop


def binary_tournament_front_selector(population, front, c_d):
    # always halves the population
    def one_comparison(front1, front2, c_d1, c_d2):
        # determines wheter candidate1 is better than candidate2
        if front1 > front2:  # higher is better!
            return True
        else:
            return c_d1 > c_d2

    pop_size = population.shape[1]
    ret_pop = np.empty((population.shape[0], pop_size//2))
    shuffled_arr = [i for i in range(pop_size)]
    random.shuffle(shuffled_arr)
    for i in range(0, len(shuffled_arr), 2):
        idxs = [shuffled_arr[i], shuffled_arr[i + 1]]
        if one_comparison(front[idxs[0]], front[idxs[1]], c_d[idxs[0]], c_d[idxs[1]]):
            ret_pop[:, i//2] = population[:, idxs[0]]
        else:
            ret_pop[:, i//2] = population[:, idxs[1]]
    return ret_pop


def binary_tournament_sorted_selector(sorted_population):
    # always halves the population
    def one_comparison(id1, id2):
        return id1 < id2

    pop_size = sorted_population.shape[1]
    ret_pop = np.empty((sorted_population.shape[0], pop_size//2))
    shuffled_arr = [i for i in range(pop_size)]
    random.shuffle(shuffled_arr)
    for i in range(0, len(shuffled_arr), 2):
        idxs = [shuffled_arr[i], shuffled_arr[i + 1]]
        if one_comparison(idxs[0], idxs[1]):
            ret_pop[:, i//2] = sorted_population[:, idxs[0]]
        else:
            ret_pop[:, i//2] = sorted_population[:, idxs[1]]
    return ret_pop


# ------ replacement strategy ------

def pointer_wheel_replacement(pop, fit, new_pop, new_fit, ret_size, sharing=None):
    whole_pop = np.hstack((pop, new_pop))
    whole_fit = np.hstack((fit, new_fit))
    if sharing is None:
        return pointer_wheel_selector(whole_pop, ut.fitness_scaling(whole_fit), ret_size, 1)  # only scale
    else:
        return pointer_wheel_selector(whole_pop,
                                      ut.fitness_sharing(ut.fitness_scaling(whole_fit), sharing[0], sharing[1]),
                                      ret_size, 1)  # scale and share


def replace_the_best(pop, fit, new_pop, new_fit, ret_size):
    whole_pop = np.hstack((pop, new_pop))
    whole_fit = np.hstack((fit, new_fit))
    idxs = whole_fit.argsort()[-ret_size:][::-1]
    return whole_pop[:, idxs]