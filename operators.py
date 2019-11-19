import numpy as np
from scipy.stats import cauchy
import itertools
import math
import random


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
        return step*int_val + bounds[0]
    return np.apply_along_axis(map_bin_real_single, 0, population, bounds)


def perturb_real_normal(population, sigma=1, p_chrom=1, p_gene=None):  # OK
    pop = np.asarray(population).copy()
    if p_gene is None:
        if p_chrom == 1:
            pop += np.random.normal(0, sigma, population.shape)
        else:
            num = int(population.shape[1] * p_chrom)
            chrom_idxs = np.random.randint(0, population.shape[1], num)
            pop[:, chrom_idxs] += np.random.normal(0, sigma, (population.shape[0], chrom_idxs.size))
    else:
        a = np.zeros(population.size, dtype=int)
        num = int(population.size * p_gene)
        a[:num] = 1
        np.random.shuffle(a)
        gene_mask = a.astype(bool).resize(population.shape)
        pop[gene_mask] += np.random.normal(0, sigma, population.shape)[gene_mask]
    return pop


def perturb_real_cauchy(population, gamma=1, p_chrom=1, p_gene=None):  # OK
    pop = np.asarray(population).copy()
    if p_gene is None:
        if p_chrom == 1:
            pop += np.asarray(cauchy.rvs(0, gamma, population.shape))
        else:
            num = int(population.shape[1] * p_chrom)
            chrom_idxs = np.random.randint(0, population.shape[1], num)
            pop[:, chrom_idxs] += np.asarray(cauchy.rvs(0, gamma, (population.shape[0], chrom_idxs.size)))
    else:
        a = np.zeros(population.size, dtype=int)
        num = int(population.size * p_gene)
        a[:num] = 1
        np.random.shuffle(a)
        gene_mask = a.astype(bool).resize(population.shape)
        pop[gene_mask] += np.asarray(cauchy.rvs(0, gamma, population.shape)[gene_mask])
    return pop


def swap_two_random(population, prob=1):  # maybe add probability here
    do_it = np.random.rand(population.shape[1])
    booleans = (do_it < prob).astype(int)
    rnd = np.floor(np.random.rand(2, population.shape[1]) * population.shape[0]).astype(int)
    rnd[0, :] = booleans*rnd[0, :]
    rnd[1, :] = booleans * rnd[1, :]
    pop = np.asarray(population)
    l = np.arange(population.shape[1])
    tmp = pop[rnd[1, :], l]
    pop[rnd[1, :], l] = population[rnd[0, :], l]
    pop[rnd[0, :], l] = tmp
    return pop


def make_multiple_swaps(population, scale):
    pop = np.asarray(population)
    swaps_num = np.round(np.random.exponential(scale, (pop.shape[1])))
    for i in range(swaps_num.size):
        swaps = np.floor(np.random.rand(swaps_num[i].astype(int)) * (pop.shape[0] - 1)).astype(int)
        for swap in swaps:
            tmp = pop[swap, i]
            pop[swap, i] = pop[swap + 1, i]
            pop[swap + 1, i] = tmp
    return pop


def swap_order(population, prob=1):
    def single_swap(parent, prob):
        if np.random.rand(1) < prob:
            interval = np.floor(np.random.rand(2) * parent.shape[0]).astype(int)
            if interval[0] > interval[1]:
                interval = np.flip(interval, axis=0)
            parent[interval[0]:interval[1]] = np.flip(parent[interval[0]:interval[1]], axis=0)
            return parent
        else:
            return parent
    return np.apply_along_axis(single_swap, 0, population, prob)


def flip_filter(population, filter_per_ts, ts_num, prob):
    width = filter_per_ts*ts_num
    num_of_flips = int(population.shape[1] * prob)
    pop_r = np.resize(population, (int(population.shape[0]/width), population.shape[1]*width))
    pop_r_flip = np.flip(pop_r, axis=0)
    idxs = np.floor(np.random.rand(num_of_flips) * pop_r_flip.shape[1]).astype(int)
    pop_r[:, idxs] = pop_r_flip[:, idxs]
    return np.resize(pop_r, population.shape)


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
                points = np.floor(np.random.rand(2) * parents.shape[0]).astype(int)
                if points[0] - points[1] == 0:
                    return parents
                if points[1] > points[0]:
                    tmp[points[0]:points[1]] = parents[points[0]:points[1], i]
                else:
                    tmp[points[0]:] = parents[points[0]:, i]
                    tmp[:points[1]] = parents[:points[1], i]
            ret[:, idx] = tmp
        return ret

    perms = math.factorial(parents_num)
    new_size = int(np.shape(population)[1]*(perms/parents_num))
    ret = np.empty((np.shape(population)[0], new_size))
    idx = 0
    for i in range(0, new_size, perms):
        ret[:, i:i+perms] = single_crossover(population[:, idx:idx+parents_num])
        idx += parents_num
    return ret


def replace_sequence(population, prob):

    def single_replacement(parents, prob):
        if np.random.rand(1) < prob:
            ret = np.empty(parents.shape)
            assert parents.shape[1] == 2
            parent1 = list(parents[:, 0])
            parent2 = list(parents[:, 1])
            interval = np.floor(np.random.rand(2) * parents.shape[0]).astype(int)
            if interval[0] > interval[1]:
                interval = np.flip(interval, axis=0)
            insert_list1 = parent1[interval[0]:interval[1]].copy()
            insert_list2 = parent2[interval[0]:interval[1]].copy()
            parent1[interval[0]:interval[0]] = insert_list2
            parent2[interval[0]:interval[0]] = insert_list1
            check_list1 = np.arange(0, interval[0])
            check_list2 = np.arange(interval[1], len(parent1))
            check_list = [*check_list1, *check_list2]
            for idx in reversed(check_list):
                if parent1[idx] in insert_list2:
                    parent1.pop(idx)
                if parent2[idx] in insert_list1:
                    parent2.pop(idx)
            ret[:, 0] = parent1
            ret[:, 1] = parent2
            return ret
        else:
            return parents

    ret = np.empty(population.shape)
    order = np.arange(population.shape[1])
    np.random.shuffle(order)
    for i in range(0, population.shape[1], 2):
        ret[:, order[i:i+2]] = single_replacement(population[:, i:i+2], prob)
    return ret


def ordered_crossover(population, prob):
    def single_ordered(parents, prob):
        if np.random.rand(1) < prob:
            ind1 = parents[:, 0].astype(int).tolist()
            ind2 = parents[:, 1].astype(int).tolist()
            ret = np.empty(parents.shape)
            size = min(len(ind1), len(ind2))
            a, b = random.sample(range(size), 2)
            if a > b:
                a, b = b, a

            holes1, holes2 = [True] * size, [True] * size
            for i in range(size):
                if i < a or i > b:
                    holes1[ind2[i]] = False
                    holes2[ind1[i]] = False

            # We must keep the original values somewhere before scrambling everything
            temp1, temp2 = ind1, ind2
            k1, k2 = b + 1, b + 1
            for i in range(size):
                if not holes1[temp1[(i + b + 1) % size]]:
                    ind1[k1 % size] = temp1[(i + b + 1) % size]
                    k1 += 1

                if not holes2[temp2[(i + b + 1) % size]]:
                    ind2[k2 % size] = temp2[(i + b + 1) % size]
                    k2 += 1

            # Swap the content between a and b (included)
            for i in range(a, b + 1):
                ind1[i], ind2[i] = ind2[i], ind1[i]

            ret[:, 0] = ind1
            ret[:, 1] = ind2
            return ret
        else:
            return parents

    ret = np.empty(population.shape)
    order = np.arange(population.shape[1])
    np.random.shuffle(order)
    for i in range(0, population.shape[1], 2):
        ret[:, order[i:i+2]] = single_ordered(population[:, i:i+2], prob)
    return ret


def trading_crossover(population, filter_per_ts, ts_num, prob):
    width = filter_per_ts*ts_num
    num_of_swaps = int(population.shape[1] * prob)
    pop_r = np.resize(population, (int(population.shape[0]/width), population.shape[1]*width))
    pop_r_copy = pop_r.copy()
    for i in range(num_of_swaps):
        chrom = np.floor(np.random.rand(2) * population.shape[1]).astype(int)
        idx1 = chrom[0] * width
        idx2 = chrom[1] * width
        tmp = np.arange(width)
        np.random.shuffle(tmp)
        idxs = tmp[:int(width/2)]
        pop_r[:, idx1+idxs] = pop_r_copy[:, idx2+idxs]
        pop_r[:, idx2+idxs] = pop_r_copy[:, idx1+idxs]
    return np.resize(pop_r, population.shape)
