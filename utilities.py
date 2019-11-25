import numpy as np


RULE = 1/5
SCALE = 1.75


def one_fifth_rule_decay(old_fitness, new_fitness, arg, decay_arg, decay_pace, max):
    for i in range(len(decay_arg)):
        for j in range(len(decay_arg[i])):
            tmp = np.sum(((new_fitness > old_fitness).astype(int)))/(old_fitness.size + new_fitness.size)
            if tmp > RULE:
                arg[i][j] = arg[i][j] * decay_pace
            else:
                if arg[i][j] < max[i][j]:
                    arg[i][j] = arg[i][j] / decay_pace


def my_print(string, verbose):
    if verbose:
        print(string)


def update_top_solution(top_solution, population, generation):
    if population.fitness.max() > top_solution["fitness"]:
        top_solution["fitness"] = population.fitness.max()
        top_solution["generation"] = generation
        top_solution["chromosome"] = population.population[:, population.fitness.argmax()]


def fitness_scaling(fitness):
    fit_min = fitness.min()
    my_fitness = fitness.copy()
    my_fitness = my_fitness - fit_min
    old_max = my_fitness.max()
    avg = np.mean(my_fitness)
    if old_max == avg:
        print("!all chromosomes have same fitness!")
    new_max = SCALE*avg
    new_min = (2-SCALE)*avg
    scale = new_max - new_min
    new_fitness = (my_fitness/old_max)*scale + new_min
    return new_fitness


def stochastic_ranking(population, fitness, fi, max_it=None, pf=0.5):
    pop_size = population.shape[1]
    if max_it is None:
        max_it = pop_size//20
    final_order = np.arange(pop_size)
    for i in range(max_it):
        swap_done = False
        for j in range(pop_size - 1):
            p = np.random.rand(1)
            if (fi[j] == 0 and fi[j + 1] == 0) or p < pf:
                if fitness[j] > fitness[j + 1]:
                    tmp = final_order[j]
                    final_order[j] = final_order[j + 1]
                    final_order[j + 1] = tmp
                    swap_done = True
            else:
                if fi[j] > fi[j + 1]:
                    tmp = final_order[j]
                    final_order[j] = final_order[j + 1]
                    final_order[j + 1] = tmp
                    swap_done = True
        if not swap_done:
            break
    return [population[:, final_order]]


def fronts_and_crowding(population, c1, c2):
    def dominators(c1, c2, indices):    # this is probably wrong!!!
        ret = []
        for idx in indices:
            if not np.sum((((c1[idx] <= c1[indices]).astype(int) + (c2[idx] <= c2[indices]).astype(int)) == 2).astype(int)) > 1:
                ret.append(idx)
        return ret

    def crowding_dist(c1, c2, fronts):
        ret_dists = np.zeros(c1.size)
        unique_fronts = np.unique(fronts)
        for front in unique_fronts:
            front_idxs = np.where(fronts == front)[0]
            c1_sorted_idxs = np.argsort(c1[front_idxs])
            c2_sorted_idxs = np.argsort(c2[front_idxs])
            for i, idx in enumerate(front_idxs):
                if c1_sorted_idxs[0] == i or c1_sorted_idxs[-1] == i or c2_sorted_idxs[0] == i or c2_sorted_idxs[-1] == i:
                    ret_dists[idx] = np.inf
                else:
                    index_c1 = np.where(c1_sorted_idxs == i)[0]
                    index_c2 = np.where(c2_sorted_idxs == i)[0]
                    c1_lower = c1[front_idxs[c1_sorted_idxs[index_c1 - 1]]]
                    c1_upper = c1[front_idxs[c1_sorted_idxs[index_c1 + 1]]]
                    c2_lower = c2[front_idxs[c2_sorted_idxs[index_c2 - 1]]]
                    c2_upper = c2[front_idxs[c2_sorted_idxs[index_c2 + 1]]]
                    ret_dists[idx] = abs(c1_upper - c1_lower) + abs(c2_upper - c2_lower)
        return ret_dists

    fronts = np.zeros(np.size(c1))
    front_num = 1
    while True:
        doms = dominators(c1, c2, np.where(fronts == 0)[0])
        if len(doms) == 0:
            fronts[fronts == 0] = front_num
            break
        else:
            fronts[doms] = front_num
            front_num += 1

    distances = crowding_dist(c1, c2, fronts)
    return population, fronts, distances





