from btc.dataset_parser import BTCBitstampNMin
from inits import init_conv_nn, sample_population
from fitness import percent_earned
from operators import perturb_real_cauchy
from selects import pointer_wheel_replacement
from tqdm import tqdm
import matplotlib.pyplot as plt


if __name__ == '__main__':

    fee = 0.3
    conv_len = 12
    filters_per_ts = 2
    timeseries_num = 2
    width = filters_per_ts * timeseries_num
    pop_size = 1000
    init_variance = 0.1
    perturb_variance = 0.01
    variance_decay = 0.9999
    candle_min = 15
    candles_per_batch = 96 * 1  # 1 day
    generations = 1000

    parser = BTCBitstampNMin(candle_min, candles_per_batch)
    print("--- Data parsed ---")
    population = sample_population(init_conv_nn, [conv_len, width, init_variance], pop_size)

    log = []

    for i in tqdm(range(generations)):

        # get new data batch
        data_incr, price = parser.get_batch()
        # evaluate
        fitness = percent_earned(population, data_incr, price, filters_per_ts, fee)
        # perturb
        new_population = perturb_real_cauchy(population, perturb_variance)
        # evaluate children
        new_fitness = percent_earned(new_population, data_incr, price, filters_per_ts, fee)
        # create new generation
        population = pointer_wheel_replacement(population, fitness, new_population, new_fitness, pop_size)
        # decay parameter
        perturb_variance *= variance_decay
        # repeat

        log.append(fitness.max())

    plt.plot(log)
    plt.grid()
    plt.show()
