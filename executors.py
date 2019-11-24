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


def multiobjective_genetic_algorithm(init_f, init_args, pop_size,
                                     fitness_f, fitness_args,
                                     preprocess_f, preprocess_args,
                                     selection_f, selection_args,
                                     mutation_f, mutation_args,
                                     crossover_f, crossover_args,
                                     condition_f, verbose):
    population = inits.sample_population(init_f, *init_args, pop_size)
    generation = 0
    log = []
    fitness = -np.inf
    error = np.inf
    top_solution = {"fitness": fitness, "error": error, "generation": 0, "chromosome": None}
    while condition_f(generation, fitness, error):
        utilities.my_print("----- GENERATION " + str(generation) + " -----", verbose)

        fitness, error = fitness_f(population, *fitness_args)
        utilities.my_print("Population evaluated", verbose)

        # UPDATE BEST SOLUTION
        best_arg = np.argmax(fitness)
        log.append([fitness[best_arg], error[best_arg]])
        if fitness[best_arg] > top_solution["fitness"] and error[best_arg] < top_solution["error"]:
            top_solution["fitness"] = fitness[best_arg]
            top_solution["error"] = error[best_arg]
            top_solution["generation"] = generation
            top_solution["chromosome"] = population[best_arg]

        processed = preprocess_f(population, fitness, error, *preprocess_args)

        parents = selection_f(*processed, *selection_args)
        utilities.my_print("Parents selected", verbose)

        children = mutation_f(parents, *mutation_args)
        children = crossover_f(children, *crossover_args)
        utilities.my_print("Children created", verbose)

        ch_fitness, ch_error = fitness_f(children, fitness_args)
        all_pop = np.hstack([population, children])
        all_fit = np.hstack([fitness, ch_fitness])
        all_err = np.hstack([error, ch_error])
        processed = preprocess_f(all_pop, all_fit, all_err, *preprocess_args)
        population = selection_f(*processed, *selection_args)
        utilities.my_print("Population replaced", verbose)

        generation += 1

    return top_solution, log, population
