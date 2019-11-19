from btc.dataset_parser import BTCBitstampNMin
from inits import init_conv_nn, sample_population
from fitness import percent_earned
from operators import perturb_real_normal, perturb_real_cauchy, trading_crossover
from selects import pointer_wheel_replacement, pointer_wheel_selector
from tqdm import tqdm
from btc.visualisation import visualise
from utilities import fitness_scaling
import matplotlib.pyplot as plt
import numpy as np
import torch

fee = 0.3
conv_len = 18
filters_per_ts = 4
timeseries_num = 6
width = filters_per_ts * timeseries_num
pop_size = 500
init_variance = 0.1
perturb_variance = init_variance * 0.1
perturb_prob = 0.1
perturb_genes = 0.1
crossover_prob = 0.001
variance_decay = 0.9999
candle_min = 60 * 6
candles_per_batch = 4 * 28  # 4 weeks
generations = 5000
validation_step = 1000
bins = 5000
new_batch_step = 1


def validate(parser, population):
    # validate
    val_incr, val_price = parser.get_whole_validation()
    val_fit, bias = percent_earned(population, val_incr, val_price, filters_per_ts, fee)
    print("Mean on validation: ", np.mean(val_fit))
    idxs = val_fit.argsort()[-3:][::-1]
    print("Best 3 on validation: ", val_fit[idxs])
    # visualise validation
    _ = visualise(population[:, val_fit.argmax()], val_incr, val_price, filters_per_ts, True)
    print("Bias of top 3: ", bias[idxs])


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = BTCBitstampNMin(candle_min, candles_per_batch)
    population = sample_population(init_conv_nn, [conv_len, width, init_variance], pop_size)

    log1 = []

    for i in tqdm(range(generations)):

        # get new data batch
        if i % new_batch_step == 0:
            data_incr, price = parser.get_batch()
        # evaluate
        fitness, bias = percent_earned(population, data_incr, price, filters_per_ts, fee, device)
        # selection for breeding
        # new_population = pointer_wheel_selector(population, fitness_scaling(fitness), int(pop_size/2), 1)
        # mutation
        new_population = perturb_real_normal(population, perturb_variance, perturb_prob, perturb_genes)
        # crossover
        new_population = trading_crossover(new_population, filters_per_ts, timeseries_num, crossover_prob)
        # evaluate children
        new_fitness, new_bias = percent_earned(new_population, data_incr, price, filters_per_ts, fee, device)
        # create new generation
        population = pointer_wheel_replacement(population, fitness, new_population, new_fitness, pop_size,
                                               [np.hstack((bias, new_bias)), bins])
        # decay parameter
        perturb_variance *= variance_decay
        # repeat

        # logs to track training progress
        log1.append(np.mean(fitness))

        if i % validation_step == 0:
            validate(parser, population)

    # plot information about training
    idxs = np.arange(len(log1))
    coef = np.polyfit(idxs, log1, 1)
    poly1d_fn = np.poly1d(coef)
    # poly1d_fn is now a function which takes in x and returns an estimate for y
    plt.plot(idxs, log1, 'b+', idxs, poly1d_fn(idxs), '--k')
    plt.grid()
    plt.ylim(-20, 20)
    plt.savefig("progress.png")
