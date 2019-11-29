import numpy as np


# ---------------------------------- BINARY
def one_max(population):  # OK

    def one_max_single(chromosome):
        return np.sum(chromosome)

    return np.apply_along_axis(one_max_single, 0, population)


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

    return np.apply_along_axis(labs_single, 0, population)


# ---------------------------------- REAL
def sphere(population, minimum):  # OK

    def sphere_single(chromosome, minimum):
        chrom = np.asarray(chromosome)
        minim = np.asarray(minimum)
        return -np.sum(np.power((chrom - minim), 2))

    return np.apply_along_axis(sphere_single, 0, population, minimum)


def rosenbrock(population):  # OK

    def rosenbrock_single(chromosome):
        chrom = np.asarray(chromosome)
        tmp = 100*np.power((chrom[1:] - np.power(chrom[:-1], 2)), 2) + np.power((1 - chrom[:-1]), 2)
        return np.sum(tmp)

    return np.apply_along_axis(rosenbrock_single, 0, population)


def linear(population, sloap, bias):  # OK

    def linear_single(chromosome, sloap, bias):
        chrom = np.asarray(chromosome)
        a = np.asarray(sloap)
        return a.dot(chrom) + bias

    return np.apply_along_axis(linear_single, 0, population, sloap, bias)


def step(population, sloap, bias):  # OK

    def step_single(chromosome, sloap, bias):
        chrom = np.asarray(chromosome)
        a = np.asarray(sloap)
        tmp_vec = np.floor(a*chrom)
        return np.sum(tmp_vec) + bias

    return np.apply_along_axis(step_single, 0, population, sloap, bias)


def rastrigin(population):  # OK

    def rastrigin_single(chromosome):
        chrom = np.asarray(chromosome)
        chrom = chrom**2 - 10*np.cos(2*np.pi*chrom)
        return 10*len(chrom) + np.sum(chrom)

    return np.apply_along_axis(rastrigin_single, 0, population)


def griewank(population):  # OK

    def griewank_single(chromosome):
        chrom = np.asarray(chromosome)
        tmp = np.arange(len(chrom)) + 1
        tmp = np.prod(np.cos(chrom/np.sqrt(tmp)))
        return 1 + (np.sum(chrom**2)/4000) - tmp

    return np.apply_along_axis(griewank_single, 0, population)


def schwefel(population):  # OK

    def schwefel_single(chromosome):
        chrom = np.asarray(chromosome)
        norm_tmp = np.abs(chrom)
        return -np.sum(chrom * np.sin(np.sqrt(norm_tmp)))

    return np.apply_along_axis(schwefel_single, 0, population)


def traveled_distance(population, distance_matrix):  # for HW1

    def traveled_distance_single(chromosome, dist_matrix):
        from_idxs = chromosome[:-1].astype(int)
        to_idxs = chromosome[1:].astype(int)
        return -np.sum(dist_matrix[from_idxs, to_idxs]) - dist_matrix[int(chromosome[0]), int(chromosome[-1])]

    return np.apply_along_axis(traveled_distance_single, 0, population, distance_matrix)


# MULTI-CRITERION ---------------------------------------------------------


def g6(population):

    def single_g6_fit(chromosome):
        c = chromosome
        f = (c[0] - 10.0)**3.0 + (c[1] - 20.0)**3.0
        return f

    def single_g6_err(chromosome):
        c = chromosome
        g1 = -(c[0] - 5.0)**2.0 - (c[1] - 5)**2.0 + 100.0
        if g1 <= 0:
            g1 = 0.0
        g2 = (c[0] - 6.0)**2.0 + (c[1] - 5)**2.0 - 82.81
        if g2 <= 0:
            g2 = 0.0
        return g1 + g2

    fs = np.apply_along_axis(single_g6_fit, 0, population)
    gs = np.apply_along_axis(single_g6_err, 0, population)
    return fs, gs


def g8(population):

    def single_g8_fit(chromosome):
        c = chromosome
        f = -(np.sin(2.0*np.pi*c[0])**3.0 * np.sin(2.0*np.pi*c[1]))/(c[0]**3.0 * (c[0] + c[1]))
        return f

    def single_g8_err(chromosome):
        c = chromosome
        g1 = c[0]**2.0 - c[1] + 1.0
        if g1 <= 0:
            g1 = 0.0
        g2 = 1.0 - c[0] + (c[1] - 4.0)**2.0
        if g2 <= 0:
            g2 = 0.0
        return g1 + g2

    fs = np.apply_along_axis(single_g8_fit, 0, population)
    gs = np.apply_along_axis(single_g8_err, 0, population)
    return fs, gs


def g11(population):

    def single_g11_fit(chromosome):
        c = chromosome
        f = c[0]**2.0 + (c[1] - 1.0)**2.0
        return f

    def single_g11_err(chromosome):
        c = chromosome
        g = c[1] - c[0]**2.0
        if g <= 0:
            g = 0.0
        return g

    fs = np.apply_along_axis(single_g11_fit, 0, population)
    gs = np.apply_along_axis(single_g11_err, 0, population)
    return fs, gs


def g24(population):

    def single_g24_fit(chromosome):
        c = chromosome
        f = -c[0] - c[1]
        return f

    def single_g24_err(chromosome):
        c = chromosome
        g1 = -2.0*c[0]**4.0 + 8.0*c[0]**3.0 - 8.0*c[0]**2.0 + c[1] - 2.0
        if g1 <= 0:
            g1 = 0.0
        g2 = -4.0*c[0]**4.0 + 32.0*c[0]**3.0 - 88.0*c[0]**2.0 + 96.0*c[0] + c[1] - 36.0
        if g2 <= 0:
            g2 = 0.0
        return g1 + g2

    fs = np.apply_along_axis(single_g24_fit, 0, population)
    gs = np.apply_along_axis(single_g24_err, 0, population)
    return fs, gs