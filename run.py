import numpy as np
import fitness as fit
import operators as op
import executors as ex
import selects as sel
import inits
import tsp.tsp_parser as parse
import matplotlib.pyplot as plt


def run_shifted_sphere():
    def condition(generation, fitness):
        if generation > 10000 or np.any(fitness > - 0.1):
            return False
        else:
            return True

    population = 200
    chrom_size = 5
    goal = [(10, 10, 5, 100, 1000)]

    top_solution, log, pop = ex.genetic_algorithm(inits.init_uniform_real, [chrom_size, (0, 10)], population,
                                                  fit.sphere, goal,
                                                  sel.pointer_wheel_selector, [population, 1],
                                                  [op.perturb_real_cauchy], [[100]], [[0]], 0.99,
                                                  [op.two_point_crossover], [[2]],
                                                  sel.pointer_wheel_replacement, [population],
                                                  condition, True)

    print(top_solution)
    plt.plot(-log)
    plt.grid()
    plt.show()


def run_local_search_tsp(file, chrom_size, searches, operators, operators_args, max_gen, max_fit):
    def condition(generation, fitness):
        if fitness.max() == max_fit or generation > max_gen:
            return False
        else:
            return True

    top_solution, log, pop = ex.genetic_algorithm(inits.init_shuffled_integer_array, [chrom_size], 1,
                                                  fit.traveled_distance, [parse.distance_matrix(file)],
                                                  sel.multiple_times_selector, [searches],
                                                  operators, operators_args,
                                                  [[0]], 0.999, [[3]],  # decay for arguments
                                                  [], [[]],  # no crossover
                                                  sel.replace_the_best, [1],
                                                  condition, False)

    print("Algorithm finished.")
    log = np.asarray(log)
    return top_solution, -log


def run_ea_tsp(file, chrom_size, population, crossover, max_gen, max_fit):
    def condition(generation, fitness):
        if fitness.max() == max_fit or generation > max_gen:
            return False
        else:
            return True

    top_solution, log, pop = ex.genetic_algorithm(inits.init_shuffled_integer_array, [chrom_size], population,
                                                  fit.traveled_distance, [parse.distance_matrix(file)],
                                                  sel.pointer_wheel_selector, [population, 1],
                                                  [op.swap_order], [[1]],
                                                  [[]], 0.99, [[1]],  # decay for arguments
                                                  crossover, [[0.1]],
                                                  sel.replace_the_best, [population],
                                                  condition, False)

    print("Algorithm finished.")
    log = np.asarray(log)
    return top_solution, -log


def run_all_local_searches():
    problem_file = "tsp/berlin52.tsp"
    chrom_size = 52
    population = 50
    max_gen = 1000
    max_fit = -7542

    # LOCAL SEARCHES

    top_sols = []
    for i in range(5):
        ts, log = run_local_search_tsp(problem_file, chrom_size, population,
                                       [op.make_multiple_swaps], [[1]], max_gen, max_fit)
        plt.plot(log, 'r')
        top_sols.append(ts)
    print("EA:", top_sols)

    top_sols = []
    for i in range(5):
        ts, log = run_local_search_tsp(problem_file, chrom_size, population,
                                       [op.swap_two_random], [[1]], max_gen, max_fit)
        plt.plot(log, 'g')
        top_sols.append(ts)
    print("EA:", top_sols)

    top_sols = []
    for i in range(5):
        ts, log = run_local_search_tsp(problem_file, chrom_size, population,
                                       [op.swap_order], [[1]], max_gen, max_fit)
        plt.plot(log, 'b')
        top_sols.append(ts)
    print("EA:", top_sols)

    plt.title("Local search berlin52")
    plt.xlabel("generation")
    plt.ylabel("distance")
    plt.legend(["Prohození sousedů", "Vzdálené prohození", "Otočení pořadí", "Optimální cesta"])
    plt.hlines(-max_fit, 0, max_gen)
    plt.grid()
    ax = plt.gca()
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('red')
    leg.legendHandles[1].set_color('green')
    leg.legendHandles[2].set_color('blue')
    leg.legendHandles[3].set_color('black')
    # plt.savefig("pr136.png")
    plt.show()


def run_all_eas():
    problem_file = "tsp/berlin52.tsp"
    chrom_size = 52
    population = 100
    max_gen = 1000
    max_fit = -7542

    # EA
    top_sols = []
    for i in range(5):
        ts, log = run_ea_tsp(problem_file, chrom_size, population, [op.replace_sequence], max_gen, max_fit)
        plt.plot(log, 'b')
        top_sols.append(ts)

    for i in range(5):
        ts, log = run_ea_tsp(problem_file, chrom_size, population, [op.ordered_crossover], max_gen, max_fit)
        plt.plot(log, 'r')
        top_sols.append(ts)

    print("EA:", top_sols)
    plt.title("EA berlin52")
    plt.xlabel("generation")
    plt.ylabel("distance")
    plt.legend(["Vložit a smazat", "Uspořádané křížení", "Optimální cesta"])
    plt.hlines(-max_fit, 0, max_gen)
    plt.grid()
    ax = plt.gca()
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('blue')
    leg.legendHandles[1].set_color('red')
    leg.legendHandles[2].set_color('black')
    # plt.savefig("st70_ea.png")
    plt.show()


if __name__ == '__main__':

    # run_all_eas()
    # run_all_local_searches()
    pass
