import numpy as np
import inits
import objects
import utilities


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
            adaptive_const = adaptive_const * np.power((np.exp(0.8)), 1 / gen_size)
            # TODO: this sounds weird, check if it works! I think it should be switched
        else:
            adaptive_const = adaptive_const * np.power((np.exp(-0.2)), 1 / gen_size)
    stats.append([opt_vec, last_fitness])
    return opt_vec, stats


def genetic_algorithm(init_f, init_args, population_size,
                      fitness_f, fitness_args,
                      selection_f, selection_args,
                      mutation_f, mutation_args, decay_args,
                      crossover_f, crossover_args,
                      replacement_f, replacement_args,
                      condition_f):
    pop = inits.sample_population(init_f, init_args, population_size)
    population = objects.Population(pop, fitness_f, selection_f, replacement_f)
    generation = 0
    while condition_f(generation, population.fitness):
        print("----- GENERATION " + str(generation) + " -----")
        population.evaluate_population(fitness_args)
        print("Population evaluated")
        population.select_parents(selection_args)
        print("Parents selected")
        population.run_operators(mutation_f, mutation_args)
        population.run_operators(crossover_f, crossover_args)
        print("Children created")
        population.evaluate_children(fitness_args)
        utilities.one_fifth_rule_decay(population.fitness, population.children_fitness, mutation_args, decay_args)
        population.do_replacement(replacement_args)
        print("Population replaced")
        generation += 1
        print(mutation_args)
    return population.population, population.fitness
