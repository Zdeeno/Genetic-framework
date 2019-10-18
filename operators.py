import numpy as np
from scipy.stats import cauchy


# ------------- BINARY -----------


def perturb_bin(chromosome, probability=0.5):  # OK
    prob_bools = np.random.rand(len(chromosome)) > probability
    return np.abs(prob_bools - chromosome)


def map_bin_real(chromosome, bounds=(0, 1)):  # OK
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


# ------------- REAL -----------


def perturb_real_normal(chromosome, sigma=1):  # OK
    chrom = np.asarray(chromosome)
    return chrom + np.random.normal(0, sigma, len(chrom))


def perturb_real_cauchy(chromosome, gamma=1):  # OK
    chrom = np.asarray(chromosome)
    return chrom + np.asarray(cauchy.rvs(0, gamma, len(chrom)))


