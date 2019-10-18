import numpy as np


def one_max(chromosome):  # OK
    return np.sum(chromosome)


def labs(chromosome):  # Probably OK
    if len(chromosome) == 1:
        return 0
    chrom = np.asarray(chromosome)
    chrom[chrom == 0] = -1
    h = 0
    for g in range(1, len(chrom) - 1):
        c = sum(a*b for a, b in zip(chrom, chrom[g:]))
        h += c**2
    return h + 1


def sphere(chromosome, minimum):  # OK
    chrom = np.asarray(chromosome)
    minim = np.asarray(minimum)
    return np.sum(np.power((chrom - minim), 2))


def rosenbrock(chromosome):  # OK
    chrom = np.asarray(chromosome)
    tmp = 100*np.power((chrom[1:] - np.power(chrom[:-1], 2)), 2) + np.power((1 - chrom[:-1]), 2)
    return np.sum(tmp)


def linear(chromosome, sloap, bias):  # OK
    chrom = np.asarray(chromosome)
    a = np.asarray(sloap)
    return a.dot(chrom) + bias


def step(chromosome, sloap, bias):  # OK
    chrom = np.asarray(chromosome)
    a = np.asarray(sloap)
    tmp_vec = np.floor(a*chrom)
    return np.sum(tmp_vec) + bias


def rastrigin(chromosome):  # OK
    chrom = np.asarray(chromosome)
    chrom = chrom**2 - 10*np.cos(2*np.pi*chrom)
    return 10*len(chrom) + np.sum(chrom)


def griewank(chromosome):  # OK
    chrom = np.asarray(chromosome)
    tmp = np.arange(len(chrom)) + 1
    tmp = np.prod(np.cos(chrom/np.sqrt(tmp)))
    return 1 + (np.sum(chrom**2)/4000) - tmp


def schwefel(chromosome):  # OK
    chrom = np.asarray(chromosome)
    norm_tmp = np.abs(chrom)
    return -np.sum(chrom * np.sin(np.sqrt(norm_tmp)))
