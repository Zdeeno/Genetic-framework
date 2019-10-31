from btc.dataset_parser import BTCBitstampNMin
from inits import init_conv_nn, sample_population
from fitness import percent_earned
from operators import perturb_real_normal, perturb_real_cauchy, trading_crossover
from selects import pointer_wheel_replacement
from tqdm import tqdm
from btc.visualisation import visualise
import matplotlib.pyplot as plt
import numpy as np
import torch


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    fee = 0.3
    conv_len = 24
    filters_per_ts = 2
    timeseries_num = 4
    width = filters_per_ts * timeseries_num
    pop_size = 100
    init_variance = 0.01
    perturb_variance = 0.001
    crossover_prob = 0.1
    variance_decay = 0.9999
    candle_min = 15
    candles_per_batch = 96 * 3  # 3 day
    generations = 500

    parser = BTCBitstampNMin(candle_min, candles_per_batch)
    population = sample_population(init_conv_nn, [conv_len, width, init_variance], pop_size)

    log1 = []
    log2 = []

    for i in tqdm(range(generations)):

        # get new data batch
        data_incr, price = parser.get_batch()
        # evaluate
        fitness = percent_earned(population, data_incr, price, filters_per_ts, fee, device)
        # mutation
        new_population = perturb_real_normal(population, perturb_variance)
        # crossover
        new_population = trading_crossover(new_population, width, crossover_prob)
        # evaluate children
        new_fitness = percent_earned(new_population, data_incr, price, filters_per_ts, fee, device)
        # create new generation
        population = pointer_wheel_replacement(population, fitness, new_population, new_fitness, pop_size)
        # decay parameter
        perturb_variance *= variance_decay
        # repeat

        # logs to track training progress
        log1.append(np.mean(fitness))
        log2.append(fitness.max())

    # validate
    val_incr, val_price = parser.get_whole_validation()
    val_fit = percent_earned(population, val_incr, val_price, filters_per_ts, fee)
    print("Mean on validation: ", np.mean(val_fit))
    print("Best on validation: ", val_fit.max())
    # visualise validation
    bias = visualise(population[:, val_fit.argmax()], val_incr, val_price, filters_per_ts, True)
    print("bias: ", bias)

    plt.plot(log1, "b")
    plt.plot(log2, "r")
    plt.grid()
    plt.savefig("progress.png")
