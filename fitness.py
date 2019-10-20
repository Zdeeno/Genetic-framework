import numpy as np


def one_max(population):  # OK
    def one_max_single(chromosome):
        return np.sum(chromosome)
    return np.apply_along_axis(one_max_single, 1, population)


def labs(population):  # Probably OK
    def labs_single(chromosome):
        if len(chromosome) == 1:
            return 0
        chrom = np.asarray(chromosome)
        chrom[chrom == 0] = -1
        h = 0
        for g in range(1, len(chrom) - 1):
            c = sum(a*b for a, b in zip(chrom, chrom[g:]))
            h += c**2
        return h + 1
    return np.apply_along_axis(labs_single, 1, population)


def sphere(population, minimum):  # OK
    def sphere_single(chromosome, minimum):
        chrom = np.asarray(chromosome)
        minim = np.asarray(minimum)
        return -np.sum(np.power((chrom - minim), 2))
    return np.apply_along_axis(sphere_single, 1, population, minimum)


def rosenbrock(population):  # OK
    def rosenbrock_single(chromosome):
        chrom = np.asarray(chromosome)
        tmp = 100*np.power((chrom[1:] - np.power(chrom[:-1], 2)), 2) + np.power((1 - chrom[:-1]), 2)
        return np.sum(tmp)
    return np.apply_along_axis(rosenbrock_single, 1, population)


def linear(population, sloap, bias):  # OK
    def linear_single(chromosome, sloap, bias):
        chrom = np.asarray(chromosome)
        a = np.asarray(sloap)
        return a.dot(chrom) + bias
    return np.apply_along_axis(linear_single, 1, population, sloap, bias)


def step(population, sloap, bias):  # OK
    def step_single(chromosome, sloap, bias):
        chrom = np.asarray(chromosome)
        a = np.asarray(sloap)
        tmp_vec = np.floor(a*chrom)
        return np.sum(tmp_vec) + bias
    return np.apply_along_axis(step_single, 1, population, sloap, bias)


def rastrigin(population):  # OK
    def rastrigin_single(chromosome):
        chrom = np.asarray(chromosome)
        chrom = chrom**2 - 10*np.cos(2*np.pi*chrom)
        return 10*len(chrom) + np.sum(chrom)
    return np.apply_along_axis(rastrigin_single, 1, population)


def griewank(population):  # OK
    def griewank_single(chromosome):
        chrom = np.asarray(chromosome)
        tmp = np.arange(len(chrom)) + 1
        tmp = np.prod(np.cos(chrom/np.sqrt(tmp)))
        return 1 + (np.sum(chrom**2)/4000) - tmp
    return np.apply_along_axis(griewank_single, 1, population)


def schwefel_single(population):  # OK
    def schwefel_single(chromosome):
        chrom = np.asarray(chromosome)
        norm_tmp = np.abs(chrom)
        return -np.sum(chrom * np.sin(np.sqrt(norm_tmp)))
    return np.apply_along_axis(schwefel_single, 1, population)

