import numpy as np


def local_search(init_vec, fitness_f, perturb_f, finish_f):
    """
    The simplest local search method
    :param init_vec: init vector
    :param fitness_f: fitness function
    :param perturb_f: operator to perturb vector
    :param finish_f: boolean function which provides final condition
    :return: [result, stats]
    """
    stats = []
    opt_vec = np.asarray(init_vec)
    last_fitness = fitness_f(opt_vec)
    while not finish_f(opt_vec):
        new_vec = perturb_f(opt_vec)
        new_fitness = fitness_f(new_vec)
        if new_fitness > last_fitness:
            stats.append([opt_vec, last_fitness])
            opt_vec = new_vec
            last_fitness = new_fitness
    stats.append([opt_vec, last_fitness])
    return opt_vec, stats


def local_search_adaptive(init_vec, init_const, fitness_f, perturb_f, finish_f, gen_size):
    """
    Local search with adaptive perturbation size using 1/5 rule
    :param init_vec: init vector
    :param init_const:
    :param fitness_f: fitness function
    :param perturb_f: operator to perturb vector
    :param finish_f: boolean function which provides final condition
    :param gen_size: generation size
    :return: [result, stats]
    """
    stats = []
    opt_vec = np.asarray(init_vec)
    last_fitness = fitness_f(opt_vec)
    adaptive_const = init_const
    while not finish_f(opt_vec):
        new_generation = np.asarray([perturb_f(opt_vec, adaptive_const) for i in range(gen_size)])
        new_genfitness = np.asarray([fitness_f(new_generation[i]) for i in range(gen_size)])
        max_id = np.argmax(new_genfitness)
        if new_genfitness[max_id] > last_fitness:
            stats.append([opt_vec, last_fitness])
            opt_vec = new_generation[max_id]
            last_fitness = new_genfitness[max_id]
            adaptive_const = adaptive_const * np.power((np.exp(0.8)), 1/gen_size)
            # TODO: this sounds weird, check if it works! I think it should be switched
        else:
            adaptive_const = adaptive_const * np.power((np.exp(-0.2)), 1/gen_size)
    stats.append([opt_vec, last_fitness])
    return opt_vec, stats