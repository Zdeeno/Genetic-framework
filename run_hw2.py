import executors as exec
import fitness as fit
import selects as sel
import operators as op
import inits as init
import utilities as ut
import numpy as np
import matplotlib.pyplot as plt


def plot_pareto_optimal(pop, fitness_f, name):
    # plot whole pareto-optimal front
    f, err = fitness_f(pop)
    _, fronts, _, _ = ut.fronts_and_crowding(pop, f, err)
    idxs = np.where(fronts == 1)[0]
    for idx in idxs:
        plt.scatter(f[idx], err[idx])
    plt.grid()
    plt.title(name)
    plt.ylabel("error")
    plt.xlabel("fitness")
    plt.show()


def run_storchastic_ranking(generations, pop_size, problem_f, lower_bound, upper_bound, min_fit, min_err, init_sigma):

    print("----- stochastic ranking on problem: " + str(problem_f)[1:13])

    def condition_f(fitness, error):
        return np.any(error[fitness <= min_fit] <= min_err)

    pop = exec.multiobjective_genetic_algorithm(init.init_uniform_real, [2, lower_bound, upper_bound], pop_size,
                                                problem_f, [],
                                                ut.stochastic_ranking, [],
                                                sel.binary_tournament_sorted_selector, [],
                                                op.perturb_real_normal, [lower_bound, upper_bound],
                                                op.two_point_crossover, [2, 0.25],
                                                condition_f, False, generations, init_sigma)

    plot_pareto_optimal(pop, problem_f, "Stochastic ranking " + str(problem_f)[1:13])


def run_NSGA2(generations, pop_size, problem_f, lower_bound, upper_bound, min_fit, min_err, init_sigma):

    print("----- NSGA II on problem: " + str(problem_f)[1:13])

    def condition_f(fitness, error):
        return np.any(error[fitness <= min_fit] <= min_err)

    pop = exec.multiobjective_genetic_algorithm(init.init_uniform_real, [2, lower_bound, upper_bound], pop_size,
                                                problem_f, [],
                                                ut.fronts_and_crowding, [],
                                                sel.binary_tournament_front_constrained_selector, [],
                                                op.perturb_real_normal, [lower_bound, upper_bound],
                                                op.two_point_crossover, [2, 0.25],
                                                condition_f, False, generations, init_sigma)

    plot_pareto_optimal(pop, problem_f, "NSGA II " + str(problem_f)[1:13])


if __name__ == '__main__':
    # for i in range(3):
    pop_size = 200
    max_iterations = 1000
    max_iterations_sr = 2000

    # problem g6
    bottom_limit = np.asarray([13.0, 0.0])
    top_limit = np.asarray([100.0, 100.0])
    perturb_sigma = 1
    run_storchastic_ranking(max_iterations_sr, pop_size, fit.g6, bottom_limit, top_limit, -6900, 0.1, perturb_sigma)
    run_NSGA2(max_iterations, pop_size, fit.g6, bottom_limit, top_limit, -6900, 0.1, perturb_sigma)

    # problem g8
    bottom_limit = np.asarray([0.1e-6, 0.1e-6])     # dividing by zero is not good idea ....
    top_limit = np.asarray([10.0, 10.0])
    perturb_sigma = 1
    run_storchastic_ranking(max_iterations_sr, pop_size, fit.g8, bottom_limit, top_limit, -0.0955, 0.01, perturb_sigma)
    run_NSGA2(max_iterations, pop_size, fit.g8, bottom_limit, top_limit, -0.0955, 0.01, perturb_sigma)

    # problem g11
    bottom_limit = np.asarray([-1.0, -1.0])
    top_limit = np.asarray([1.0, 1.0])
    perturb_sigma = 0.2
    run_storchastic_ranking(max_iterations_sr, pop_size, fit.g11, bottom_limit, top_limit, 0.7501, 0.001, perturb_sigma)
    run_NSGA2(max_iterations, pop_size, fit.g11, bottom_limit, top_limit, 0.7501, 0.01, perturb_sigma)

    # problem g24
    bottom_limit = np.asarray([0.0, 0.0])
    top_limit = np.asarray([3.0, 4.0])
    perturb_sigma = 0.3
    run_storchastic_ranking(max_iterations_sr, pop_size, fit.g24, bottom_limit, top_limit, -5.5, 0.01, perturb_sigma)
    run_NSGA2(max_iterations, pop_size, fit.g24, bottom_limit, top_limit, -5.5, 0.01, perturb_sigma)

