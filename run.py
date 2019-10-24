import numpy as np
import fitness as fit
import operators as op
import executors as ex
import selects as sel
import inits
import tcp_parser as parse
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


def run_tsp(file, chrom_size, population, max_gen, max_fit):
    def condition(generation, fitness):
        if fitness.max() == max_fit or generation > max_gen:
            return False
        else:
            return True

    top_solution, log, pop = ex.genetic_algorithm(inits.init_shuffled_integer_array, [chrom_size], population,
                                                  fit.traveled_distance, [parse.distance_matrix(file)],
                                                  sel.pointer_wheel_selector, [population, 1],
                                                  [op.make_multiple_swaps, op.swap_two_random], [[5], [0.75]],
                                                  [[0], []], 0.999,  # decay for arguments
                                                  [op.ordered_crossover], [[0.25]],
                                                  sel.pointer_wheel_replacement, [population],
                                                  condition, False)

    print("Algorithm finished.")
    log = np.asarray(log)
    return top_solution, -log


if __name__ == '__main__':
    problem_file = "tsp/berlin52.tsp"
    chrom_size = 52
    population = 50
    max_gen = 5000
    max_fit = -7542
    top_sols = []
    for i in range(5):
        ts, log = run_tsp(problem_file, chrom_size, population, max_gen, max_fit)
        plt.plot(log, 'b', linewidth=0.5, alpha=0.7)
        top_sols.append(ts)
    print(top_sols)
    plt.grid()
    plt.show()
    plt.title("pr136")
