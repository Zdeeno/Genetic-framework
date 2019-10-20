import numpy as np
import fitness as fit
import operators as op
import executors as ex
import selectors as sel
import inits


def run_binary_local_search():
    def final_condition(my_input):
        return fit.one_max(my_input) >= 5
    ret, stats = ex.local_search([0, 0, 0, 0, 0], fit.one_max, op.perturb_bin, final_condition)
    print(ret, stats)


def run_evolution_algorithm():
    def condition(generation, fitness):
        if generation > 500 or np.any(fitness == 0):
            return False
        else:
            return True
    ex.evolutionary_algorithm(inits.init_uniform_real, [5, (0, 5)], 10,
                              fit.sphere, [(10, 10, 5, 0, 0)],
                              sel.pointer_wheel_selector, [10, 1],
                              [op.two_point_crossover, op.perturb_real_normal], [[2], [1]],
                              sel.pointer_wheel_replacement, [10],
                              condition)


if __name__ == '__main__':
    run_evolution_algorithm()