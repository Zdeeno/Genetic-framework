import executors as exec
import fitness as fit
import selects as sel
import operators as op
import inits as init
import utilities as ut
import numpy as np


def run_storchastic_ranking(generations, pop_size, problem_f, lower_bound, upper_bound, max_fit):
    def condition_f(generation, fitness, error):
        if generation >= generations:
            return False
        return not np.any([(fitness[i] < max_fit and error[i] < 0.1) for i in range(len(fitness))])

    output = exec.multiobjective_genetic_algorithm(init.init_uniform_real, [2, lower_bound, upper_bound], pop_size,
                                                   problem_f, [],
                                                   ut.stochastic_ranking, [],
                                                   sel.binary_tournament_sorted_selector, [],
                                                   op.perturb_real_normal, [lower_bound, upper_bound],
                                                   op.two_point_crossover, [2, 0.25],
                                                   condition_f, True)

    print(*output)


def run_NSGA2(generations, pop_size, problem_f, lower_bound, upper_bound, max_fit):
    def condition_f(generation, fitness, error):
        if generation >= generations:
            return False
        return not np.any([(fitness[i] < max_fit and error[i] < 0.1) for i in range(len(fitness))])

    output = exec.multiobjective_genetic_algorithm(init.init_uniform_real, [2, lower_bound, upper_bound], pop_size,
                                                   problem_f, [],
                                                   ut.fronts_and_crowding, [],
                                                   sel.binary_tournament_front_selector, [],
                                                   op.perturb_real_normal, [lower_bound, upper_bound],
                                                   op.two_point_crossover, [2, 0.25],
                                                   condition_f, True)

    print(*output)


if __name__ == '__main__':
    # for i in range(3):
    run_storchastic_ranking(2000, 100, fit.g6, np.asarray([13, 0]), np.asarray([100, 100]), -6900)
    # run_NSGA2(500, 100, fit.g6, np.asarray([13, 0]), np.asarray([100, 100]), -6900)
