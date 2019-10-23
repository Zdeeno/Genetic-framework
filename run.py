import numpy as np
import fitness as fit
import operators as op
import executors as ex
import selects as sel
import inits


def run_shifted_sphere():
    def condition(generation, fitness):
        if generation > 10000 or np.any(fitness > - 0.1):
            return False
        else:
            return True

    population = 20

    pop, fitness = ex.genetic_algorithm(inits.init_uniform_real, [5, (0, 10)], population,
                                        fit.sphere, [(10, 10, 5, 100, 1000)],
                                        sel.pointer_wheel_selector, [population, 1],
                                        [op.perturb_real_cauchy], [[100]], [[0]],
                                        [op.two_point_crossover], [[2]],
                                        sel.pointer_wheel_replacement, [population],
                                        condition)

    print(pop[:, fitness.argmax()])
    print(fitness.max())


if __name__ == '__main__':
    run_shifted_sphere()
