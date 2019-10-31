import numpy as np
import torch


def one_max(population):  # OK

    def one_max_single(chromosome):
        return np.sum(chromosome)

    return np.apply_along_axis(one_max_single, 0, population)


def labs(population):  # Probably OK

    def labs_single(chromosome):
        if len(chromosome) == 1:
            return 0
        chrom = np.asarray(chromosome)
        chrom[chrom == 0] = -1
        h = 0
        for g in range(1, len(chrom) - 1):
            c = sum(a*b for a, b in zip(chrom, chrom[g:]))
            h += c**2
        return h + 1

    return np.apply_along_axis(labs_single, 0, population)


def sphere(population, minimum):  # OK

    def sphere_single(chromosome, minimum):
        chrom = np.asarray(chromosome)
        minim = np.asarray(minimum)
        return -np.sum(np.power((chrom - minim), 2))

    return np.apply_along_axis(sphere_single, 0, population, minimum)


def rosenbrock(population):  # OK

    def rosenbrock_single(chromosome):
        chrom = np.asarray(chromosome)
        tmp = 100*np.power((chrom[1:] - np.power(chrom[:-1], 2)), 2) + np.power((1 - chrom[:-1]), 2)
        return np.sum(tmp)

    return np.apply_along_axis(rosenbrock_single, 0, population)


def linear(population, sloap, bias):  # OK

    def linear_single(chromosome, sloap, bias):
        chrom = np.asarray(chromosome)
        a = np.asarray(sloap)
        return a.dot(chrom) + bias

    return np.apply_along_axis(linear_single, 0, population, sloap, bias)


def step(population, sloap, bias):  # OK

    def step_single(chromosome, sloap, bias):
        chrom = np.asarray(chromosome)
        a = np.asarray(sloap)
        tmp_vec = np.floor(a*chrom)
        return np.sum(tmp_vec) + bias

    return np.apply_along_axis(step_single, 0, population, sloap, bias)


def rastrigin(population):  # OK

    def rastrigin_single(chromosome):
        chrom = np.asarray(chromosome)
        chrom = chrom**2 - 10*np.cos(2*np.pi*chrom)
        return 10*len(chrom) + np.sum(chrom)

    return np.apply_along_axis(rastrigin_single, 0, population)


def griewank(population):  # OK

    def griewank_single(chromosome):
        chrom = np.asarray(chromosome)
        tmp = np.arange(len(chrom)) + 1
        tmp = np.prod(np.cos(chrom/np.sqrt(tmp)))
        return 1 + (np.sum(chrom**2)/4000) - tmp

    return np.apply_along_axis(griewank_single, 0, population)


def schwefel(population):  # OK

    def schwefel_single(chromosome):
        chrom = np.asarray(chromosome)
        norm_tmp = np.abs(chrom)
        return -np.sum(chrom * np.sin(np.sqrt(norm_tmp)))

    return np.apply_along_axis(schwefel_single, 0, population)


def traveled_distance(population, distance_matrix):  # for HW1

    def traveled_distance_single(chromosome, dist_matrix):
        from_idxs = chromosome[:-1].astype(int)
        to_idxs = chromosome[1:].astype(int)
        return -np.sum(dist_matrix[from_idxs, to_idxs]) - dist_matrix[int(chromosome[0]), int(chromosome[-1])]

    return np.apply_along_axis(traveled_distance_single, 0, population, distance_matrix)


def percent_earned(population, timeseries, price_ts, filter_per_ts, fee, device=None):
    """
    :param population: (length, population)
    :param timeseries: (data, N)
    :return: percent earned per chromosome
    """
    # setup many variables
    torch.no_grad()  # setup torch
    width = timeseries.shape[1] * filter_per_ts
    filter_rows = int(width*population.shape[1])
    filter_len = int(population.shape[0]/width)
    conv_filter = np.resize(np.transpose(population), (filter_rows, 1, filter_len))
    torch_filter = torch.nn.Conv1d(1, population.shape[1]*filter_per_ts, conv_filter.shape[2], bias=False)
    torch_ts = torch.from_numpy(np.resize(np.transpose(timeseries), (1, timeseries.shape[1], timeseries.shape[0])))
    torch_filter_vals = torch.from_numpy(conv_filter)
    fitness = torch.zeros(population.shape[1])

    # move variables to gpu if possible
    if device is not None and device != torch.device("cpu"):
        fitness = fitness.to(device)
        torch_ts = torch_ts.to(device)
        torch_filter = torch_filter.to(device)
        torch_filter_vals = torch_filter_vals.to(device)

    outputs = []
    for i in range(int(width/filter_per_ts)):
        idxs = []
        for j in range(i*filter_per_ts, filter_rows, width):
            for k in range(filter_per_ts):
                idxs.append(j + k)

        new_params = torch.nn.Parameter(torch_filter_vals[idxs, :, :], requires_grad=False)
        torch_filter.weight = new_params
        out = torch_filter(torch_ts[0, i, :-1].view(1, 1, -1))
        outputs.append(out)

    # sum with other timeseries (possible improvement - use sum method)
    out_sum = outputs[0]
    for i in range(1, len(outputs)):
        out_sum += outputs[i]

    # sum with other filters
    final_sum = out_sum[:, 0::filter_per_ts, :]
    for i in range(1, filter_per_ts):
        final_sum += out_sum[:, i::filter_per_ts, :]

    # find buy and sell signals
    buys = (final_sum > 1).double()
    sells = (final_sum < 1).double()
    actions = buys - sells

    # fill zeros with -1 and 1, count fees
    for i in range(1, actions.shape[2]):
        nulls = actions[0, :, i] == 0
        actions[0, nulls, i] = actions[0, nulls, i-1]
        diff = torch.abs(actions[0, :, i] - actions[0, :, i - 1])
        fitness[diff > 1] -= fee

    # use price increments to calculate profit
    price_incr = torch_ts[0, 0, filter_len:].view(1, 1, -1).repeat((1, population.shape[1], 1))
    fit_per_action = price_incr * actions
    fitness = fit_per_action.sum(2).view(-1) + fitness

    if device is not None and device != torch.device("cpu"):
        fitness.cpu()

    fitness = fitness.numpy()

    return fitness





