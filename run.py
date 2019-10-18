import numpy as np
import fitness as fit
import operators as op
import executors as ex


def run_binary_local_search():
    def final_condition(my_input):
        return fit.one_max(my_input) >= 5
    ret, stats = ex.local_search([0, 0, 0, 0, 0], fit.one_max, op.perturb_bin, final_condition)
    print(ret, stats)
