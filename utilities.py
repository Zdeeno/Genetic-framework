import numpy as np
import os

RULE = 1/5
SCALE = 1.2


def one_fifth_rule_decay(old_fitness, new_fitness, arg, decay_arg, decay_pace, max):
    for i in range(len(decay_arg)):
        for j in range(len(decay_arg[i])):
            tmp = np.sum(((new_fitness > old_fitness).astype(int)))/(old_fitness.size + new_fitness.size)
            if tmp > RULE:
                arg[i][j] = arg[i][j] * decay_pace
            else:
                if arg[i][j] < max[i][j]:
                    arg[i][j] = arg[i][j] / decay_pace


def my_print(string, verbose):
    if verbose:
        print(string)


def update_top_solution(top_solution, population, generation):
    if population.fitness.max() > top_solution["fitness"]:
        top_solution["fitness"] = population.fitness.max()
        top_solution["generation"] = generation
        top_solution["chromosome"] = population.population[:, population.fitness.argmax()]


def fitness_scaling(fitness):
    fit_min = fitness.min()
    my_fitness = fitness.copy()
    my_fitness = my_fitness - fit_min
    old_max = my_fitness.max()
    avg = np.mean(my_fitness)
    if old_max == avg:
        print("!all chromosomes have same fitness!")
    new_max = SCALE*avg
    new_min = (2-SCALE)*avg
    scale = new_max - new_min
    new_fitness = (my_fitness/old_max)*scale + new_min
    return new_fitness


def dump_population(population, filename):
    if os.path.isfile(filename):
        i = 0
        while os.path.isfile(filename + str(i)):
            i += 1
        np.savetxt(filename + str(i), population, delimiter=',')
    else:
        np.savetxt(filename, population, delimiter=',')


def load_population(filename):
    assert os.path.isfile(filename), filename + " - this is not a file!!!"
    return np.genfromtxt(filename, delimiter=',')
