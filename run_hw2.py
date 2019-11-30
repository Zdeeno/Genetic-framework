import executors as exec
import fitness as fit
import selects as sel
import operators as op
import inits as init
import utilities as ut
import numpy as np
import matplotlib.pyplot as plt


ITERATIONS = 3


def plot_pareto_optimal(pops, fitness_f, name, opt):
    # plot whole pareto-optimal front
    best_dists = np.zeros(ITERATIONS)
    for pop_id, pop in enumerate(pops):
        f, err = fitness_f(pop)
        _, fronts, _, _ = ut.fronts_and_crowding(pop, f, err)
        idxs = np.where(fronts == 1)[0]
        best_dist = np.inf
        for idx in idxs:
            plt.scatter(f[idx], err[idx])
            dst = np.linalg.norm((f[idx] - opt, err[idx]))
            if dst < best_dist:
                best_dist = dst
        best_dists[pop_id] = best_dist
    plt.scatter(opt, 0.0, marker='x', c="red", s=100)
    plt.grid()
    plt.title(name)
    plt.ylabel("error")
    plt.xlabel("fitness")
    plt.show()
    print("Average best distance is: " + "{0:.4f}".format(np.mean(best_dists)))


def run_storchastic_ranking(generations, pop_size, problem_f, lower_bound, upper_bound, opt_fit, max_dist, init_sigma):

    def condition_f(fitness, error):
        return np.linalg.norm((fitness - opt_fit, error)) < max_dist

    pops = []

    for i in range(ITERATIONS):
        print("----- Stochastic ranking on problem: " + str(problem_f)[1:13] + " - iteration " + str(i))

        pop = exec.multiobjective_genetic_algorithm(init.init_uniform_real, [2, lower_bound, upper_bound], pop_size,
                                                    problem_f, [],
                                                    ut.stochastic_ranking, [],
                                                    sel.binary_tournament_sorted_selector, [],
                                                    op.perturb_real_normal, [lower_bound, upper_bound],
                                                    op.two_point_crossover, [2, 0.25],
                                                    condition_f, False, generations, init_sigma)
        pops.append(pop)

    plot_pareto_optimal(pops, problem_f, "Stochastic ranking " + str(problem_f)[1:13], opt_fit)


def run_NSGA2(generations, pop_size, problem_f, lower_bound, upper_bound, opt_fit, max_dist, init_sigma):

    def condition_f(fitness, error):
        return np.linalg.norm((fitness - opt_fit, error)) < max_dist

    pops = []

    for i in range(ITERATIONS):
        print("----- NSGA II on problem: " + str(problem_f)[1:13] + " - iteration " + str(i))

        pop = exec.multiobjective_genetic_algorithm(init.init_uniform_real, [2, lower_bound, upper_bound], pop_size,
                                                    problem_f, [],
                                                    ut.fronts_and_crowding, [],
                                                    sel.binary_tournament_front_constrained_selector, [],
                                                    # sel.binary_tournament_front_selector, [],
                                                    op.perturb_real_normal, [lower_bound, upper_bound],
                                                    op.two_point_crossover, [2, 0.25],
                                                    condition_f, False, generations, init_sigma)

        pops.append(pop)

    plot_pareto_optimal(pops, problem_f, "NSGA II " + str(problem_f)[1:13], opt_fit)


if __name__ == '__main__':
    # for i in range(3):
    pop_size = 100
    max_iterations = 1000
    max_iterations_sr = 1000

    # problem g6
    bottom_limit = np.asarray([13.0, 0.0])
    top_limit = np.asarray([100.0, 100.0])
    perturb_sigma = [1.0, 0.998]  # sigma and decay
    opt_fit = -6961.8138
    stop_dst = 0.25
    run_storchastic_ranking(max_iterations_sr, pop_size, fit.g6, bottom_limit, top_limit, opt_fit, stop_dst, perturb_sigma)
    run_NSGA2(max_iterations, pop_size, fit.g6, bottom_limit, top_limit, opt_fit, stop_dst, perturb_sigma)

    # problem g8
    bottom_limit = np.asarray([0.1e-6, 0.1e-6])     # dividing by zero is not good idea ....
    top_limit = np.asarray([10.0, 10.0])
    perturb_sigma = [0.5, 0.998]
    opt_fit = -0.09582
    stop_dst = 0.01
    run_storchastic_ranking(max_iterations_sr, pop_size, fit.g8, bottom_limit, top_limit, opt_fit, stop_dst, perturb_sigma)
    run_NSGA2(max_iterations, pop_size, fit.g8, bottom_limit, top_limit, opt_fit, stop_dst, perturb_sigma)

    # problem g11
    bottom_limit = np.asarray([-1.0, -1.0])
    top_limit = np.asarray([1.0, 1.0])
    perturb_sigma = [0.1, 0.998]
    opt_fit = 0.7499
    stop_dst = 0.001
    run_storchastic_ranking(max_iterations_sr, pop_size, fit.g11, bottom_limit, top_limit, opt_fit, stop_dst, perturb_sigma)
    run_NSGA2(max_iterations, pop_size, fit.g11, bottom_limit, top_limit, opt_fit, stop_dst, perturb_sigma)

    # problem g24
    bottom_limit = np.asarray([0.0, 0.0])
    top_limit = np.asarray([3.0, 4.0])
    perturb_sigma = [0.25, 0.998]
    opt_fit = -5.508
    stop_dst = 0.0025
    run_storchastic_ranking(max_iterations_sr, pop_size, fit.g24, bottom_limit, top_limit, opt_fit, stop_dst, perturb_sigma)
    run_NSGA2(max_iterations, pop_size, fit.g24, bottom_limit, top_limit, opt_fit, stop_dst, perturb_sigma)

