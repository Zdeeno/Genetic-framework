import numpy as np


DECAY_PARAMETER = 0.999
RULE = 1/5


def one_fifth_rule_decay(old_fitness, new_fitness, arg, decay_arg):
    for i in range(len(decay_arg)):
        for j in range(len(decay_arg[i])):
            if np.sum(((new_fitness > old_fitness).astype(int)))/old_fitness.size > RULE:
                arg[i][j] = arg[i][j] * DECAY_PARAMETER
            else:
                arg[i][j] = arg[i][j] / DECAY_PARAMETER
