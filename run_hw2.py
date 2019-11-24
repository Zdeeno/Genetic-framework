import executors as exec
import fitness as fit
import selects as sel
import operators as op
import inits as init
import utilities as ut


def run_storchastic_ranking(generations, pop_size):
    def condition_f(generation, fitness, error):
        return not ((fitness > -0.1 and error < 0.01) or generation > generations)

    exec.multiobjective_genetic_algorithm(init.init_uniform_real, [2, [-5, 5]], pop_size,
                                          fit.g6, [],
                                          ut.stochastic_ranking, [],
                                          sel.binary_tournament_sorted_selector, [],
                                          op.perturb_real_normal, [],
                                          op.two_point_crossover, [2],
                                          condition_f, True)


if __name__ == '__main__':
    run_storchastic_ranking(1000, 100)