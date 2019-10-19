import numpy as np
from scipy.stats import cauchy


# ------------- BINARY -----------


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


# ------------- REAL -----------


def perturb_real_normal(population, sigma=1):  # OK
    pop = np.asarray(population)
    return pop + np.random.normal(0, sigma, np.shape(pop))


def perturb_real_cauchy(population, gamma=1):  # OK
    pop = np.asarray(population)
    return pop + np.asarray(cauchy.rvs(0, gamma, np.shape(pop)))


def two_point_crossover(population, parents_num):
    def single_crossover(parents):
        assert len(parents) > 1
        ret = parents[0]
        for i in range(len(parents) - 1):
            points = np.floor(np.random.rand((2,))*len(parents[0]))
            if points[1] > points[0]:
                ret[points[0]:points[1]] = parents[i + 1][points[0]:points[1]]
            else:
                ret[points[0]:-1] = parents[i + 1][points[0]:-1]
                ret[0:points[1]] = parents[i + 1][0:points[1]]
        return ret
    ret = np.zeros((np.shape(population)[0], np.shape(population)[1]/parents_num))
    for i in range(population/parents_num):
        idx = parents_num*i
        ret[:, i] = single_crossover(population[:, idx:idx+parents_num])
    return ret
