import numpy as np
import inits
import objects
import utilities


def genetic_algorithm(init_f, init_args, population_size,
                      fitness_f, fitness_args,
                      selection_f, selection_args,
                      mutation_f, mutation_args, decay_args, decay_pace, max_args,
                      crossover_f, crossover_args,
                      replacement_f, replacement_args,
                      condition_f, verbose):
    pop = inits.sample_population(init_f, init_args, population_size)
    population = objects.Population(pop, fitness_f, selection_f, replacement_f)
    generation = 0
    log = []
    top_solution = {"fitness": -np.Inf, "generation": 0, "chromosome": None}
    while condition_f(generation, population.fitness):
        utilities.my_print("----- GENERATION " + str(generation) + " -----", verbose)
        population.evaluate_population(fitness_args)
        utilities.my_print("Population evaluated", verbose)
        population.select_parents(selection_args)
        utilities.my_print("Parents selected", verbose)
        population.run_operators(mutation_f, mutation_args)
        population.run_operators(crossover_f, crossover_args)
        utilities.my_print("Children created", verbose)
        population.evaluate_children(fitness_args)
        utilities.one_fifth_rule_decay(population.fitness, population.children_fitness,
                                       mutation_args, decay_args, decay_pace, max_args)
        population.do_replacement(replacement_args)
        utilities.my_print("Population replaced", verbose)
        generation += 1
        log.append(population.fitness.max())
        utilities.my_print("Decay args:" + str(mutation_args), verbose)
        utilities.update_top_solution(top_solution, population, generation)

    return top_solution, log, population
