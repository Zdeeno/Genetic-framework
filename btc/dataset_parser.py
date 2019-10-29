import os.path
import numpy as np
import sys
from abc import ABCMeta, abstractmethod

# File for parsing historical data and passing them to environment in batches.
# All data should be located in 'datasets' in home folder

MINUTES_PER_MONTH = 60*24*30
MINUTES_PER_WEEK = 60 * 24 * 7


class Parser(metaclass=ABCMeta):
    def __init__(self, obs_length):
        self._src_folder = os.path.join(os.path.expanduser("~"), 'datasets')
        self._obs_length = obs_length

    def set_seed(self, seed):
        np.random.seed(seed)

    @abstractmethod
    def eval_batch(self):
        """
        sample random batch of testing data
        :return: generator
        """
        pass

    @abstractmethod
    def new_batch(self):
        """
        sample random batch of training data
        :return: generator
        """
        pass

    @abstractmethod
    def get_fee(self):
        pass


class BTCBitstampMin(Parser):
    """
    Available from https://www.kaggle.com/mczielinski/bitcoin-historical-data
    Data has 8 columns - last two volume, price
    """

    def __init__(self, obs_length):
        super().__init__(obs_length)
        self._src_file = os.path.join(self._src_folder, 'bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv')
        assert os.path.isfile(self._src_file), 'Download dataset from link above and place it into ~/datasets'
        self._data = np.genfromtxt(self._src_file, delimiter=',')
        self._data_length = np.shape(self._data)[0]
        # skip first two years due to low volume and last two months used for evaluation
        self._start = MINUTES_PER_MONTH * 12 * 2
        self._end = self._data_length - MINUTES_PER_MONTH*2
        self._curr_position = None
        self._curr_end = None
        self._batch_length = MINUTES_PER_WEEK
        self._repair_volume()

    def eval_batch(self):
        train_num = self._data_length - self._end - self._batch_length
        self._curr_position = int(np.random.random() * train_num) + self._data_length + self._obs_length
        while self._curr_position < self._data_length:
            ret = self._data[self._curr_position - self._obs_length:self._curr_position, 6:].T
            yield ret
            self._curr_position += 1
        yield None

    def new_batch(self):
        train_num = self._end - self._start - self._batch_length
        self._curr_position = int(np.random.random() * train_num) + self._start + self._obs_length
        self._curr_end = self._curr_position + self._batch_length
        while self._curr_position < self._curr_end:
            ret = self._data[self._curr_position - self._obs_length:self._curr_position, 6:].T
            yield ret
            self._curr_position += 1
        yield None

    def get_fee(self):
        return 0.3 / 100

    def _repair_volume(self):
        """
        Bitstamp dataset has corrupted volume. When no trade occur, there are two same records in row.
        """
        err = 1
        for i in range(1, self._data_length):
            if self._data[i-err, 6] == self._data[i, 6] or self._data[i, 6] == self._data[i, 7]:
                self._data[i, 6] = 1
                err += 1
            else:
                err = 1


class BTCBitstampNMin:

    def __init__(self, candle_min, batch_size):
        self._src_file = 'btc/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv'
        assert os.path.isfile(self._src_file), 'Download dataset from link above and place it into ~/datasets'
        self._data = np.genfromtxt(self._src_file, delimiter=',')
        self._data_length = np.shape(self._data)[0]
        # skip first four years due to low volume and last two months used for evaluation
        self._start = MINUTES_PER_MONTH * 12 * 4
        self._end = self._data_length - MINUTES_PER_MONTH*2
        self._candle_length = candle_min
        self._candles_per_batch = batch_size
        self._repair_volume()

        mod = self._data.shape[0] % candle_min

        # get timeseries
        self._price_ts = self._data[:-mod, 7]
        self._volume_ts = self._data[:-mod, 6]

        # min candles to N-min candles
        self._price_ts = self._price_ts[::candle_min]
        self._volume_ts = self._get_sum_ts(self._volume_ts)

        # get incremental timeseries
        self._price_incr_ts = self._get_incremental_ts(self._price_ts)
        self._volume_incr_ts = self._get_incremental_ts(self._volume_ts)

        # align data
        self._price_ts = self._price_ts[1:]
        self._volume_ts = self._volume_ts[1:]

        # delete unused data
        self._data = None

    def _repair_volume(self):
        """
        Bitstamp dataset has corrupted volume. When no trade occur, there are two same records in row.
        """
        err = 1
        for i in range(1, self._data_length):
            if self._data[i - err, 6] == self._data[i, 6] or self._data[i, 6] == self._data[i, 7]:
                self._data[i, 6] = 1
                err += 1
            else:
                err = 1

    def _get_incremental_ts(self, ts):
        ts_from = ts[:-1]
        ts_to = ts[1:]
        return (ts_to/ts_from - 1)*100

    def _get_sum_ts(self, ts):
        new_ts = np.reshape(ts, (int(ts.size/self._candle_length), self._candle_length))
        new_ts = np.sum(new_ts, axis=1)
        return new_ts

    def get_batch(self):
        batch_start = np.floor(np.random.rand(1) * (self._price_ts.size - self._candles_per_batch)).astype(int)[0]
        ret = np.empty((self._candles_per_batch, 2))
        ret[:, 0] = self._price_incr_ts[batch_start:(batch_start + self._candles_per_batch)]
        ret[:, 1] = self._volume_incr_ts[batch_start:(batch_start + self._candles_per_batch)]
        return ret, self._price_ts[batch_start:(batch_start + self._candles_per_batch)]


if __name__ == "__main__":
    print(sys.version)
    test = BTCBitstampMin(12)
    test.set_seed(7)
    generator = test.new_batch()
    while True:
        print(next(generator)[1, :])
