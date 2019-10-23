import numpy as np
from scipy.stats import cauchy
import itertools
import math


# ------------- MUTATION -----------


def perturb_bin(population, probability=0.5):  # OK
    prob_bools = np.random.rand(np.shape(population)) > probability
    return np.abs(prob_bools - population)


def map_bin_real(population, bounds=(0, 1)):
    def map_bin_real_single(chromosome, bounds):  # OK
        """
        maps binary (little endian) to real representation
        :param chromosome:
        :param bounds: [lower_b, upper_b]
        :return: real chromosome
        """
        c = list(reversed(chromosome))
        size = 2**len(chromosome)
        step = (bounds[1] - bounds[0])/float(size)
        int_val = 0
        for i in range(len(chromosome)):
            int_val += c[i]*2**i
        print(int_val)
        return step*int_val + bounds[0]
    return np.apply_along_axis(map_bin_real_single, 1, population, bounds)


def perturb_real_normal(population, sigma=1):  # OK
    pop = np.asarray(population)
    return pop + np.random.normal(0, sigma, np.shape(pop))


def perturb_real_cauchy(population, gamma=1):  # OK
    pop = np.asarray(population)
    return pop + np.asarray(cauchy.rvs(0, gamma, np.shape(pop)))


def swap_two_random(population):
    rnd = np.floor(np.random.rand(2, population.shape[1]) * population.shape[0])
    pop = np.asarray(population)
    pop[rnd[1, :]] = population[rnd[0, :]]
    pop[rnd[0, :]] = population[rnd[1, :]]
    return pop


# ------------- CROSSOVER -----------


def two_point_crossover(population, parents_num):

    def single_crossover(parents):
        assert len(parents) > 1
        l = [i for i in range(parents.shape[1])]
        perm = itertools.permutations(l)
        ret = np.empty((population.shape[0], math.factorial(parents.shape[1])))
        for idx, order in enumerate(perm):
            tmp = parents[:, order[0]]
            for i in range(1, len(order)):
                points = np.floor(np.random.rand(2)*len(parents[0]))
                if points[1] > points[0]:
                    tmp[points[0]:points[1]] = parents[i][points[0]:points[1]]
                else:
                    tmp[points[0]:-1] = parents[i][points[0]:-1]
                    tmp[0:points[1]] = parents[i][0:points[1]]
            ret[:, idx] = tmp
        return ret

    perms = math.factorial(parents_num)
    new_size = np.shape(population)[1]*(perms/parents_num)
    ret = np.empty((np.shape(population)[0], new_size))
    for i in range(0, new_size, perms):
        idx = parents_num*i
        ret[:, i:i+perms] = single_crossover(population[:, idx:idx+parents_num])
    return ret


def replace_sequence(population):

    def single_replacement(parents):
        ret = np.empty(parents.shape)
        assert parents.shape[1] == 2
        parent1 = list(parents[:, 0])
        parent2 = list(parents[:, 1])
        interval = np.floor(np.random.rand(2) * parents.shape[0])
        if interval[0] > interval[1]:
            interval = np.flip(interval)
        tmp = parent1.copy()
        parent1.insert(interval[0], parent2[interval[0]:interval[1]])
        parent2.insert(interval[0], tmp[interval[0]:interval[1]])
        check_list1 = np.arange(0, interval[0])
        check_list2 = np.arange(interval[1], parents.shape[0])
        check_list = [*check_list1, *check_list2]
        for i in check_list:
            if parent1[i] in parent2[interval[0]:interval[1]]:
                parent1.pop(i)
            if parent2[i] in parent1[interval[0]:interval[1]]:
                parent2.pop(i)
        ret[:, 0] = parent1
        ret[:, 0] = parent2
        return ret

    ret = np.empty(population.shape)
    for i in range(0, population.shape[1]):
        ret[:, i:i+1] = single_replacement(population[:, i:i+1])
    return ret
